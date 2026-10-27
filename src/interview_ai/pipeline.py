"""Interview pipeline — structured LLM output with validated Pydantic models.

Production path:
  LLM JSON output → JSON parsing → Pydantic validation → typed result

Fallback (logged):
  if parsing/validation fails → weak rule-based result

This module must NOT depend directly on OpenRouterClient. It receives any
LLMProvider implementation, enabling clean provider swaps and unit testing
with mocks.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from interview_ai.io import read_any_text
from interview_ai.parsers import generate_questions_weak, match_resume_to_jd, parse_resume_weak
from interview_ai.prompts import (
    SYSTEM_PROMPT,
    build_answer_evaluation_prompt,
    build_follow_up_prompt,
    build_interview_plan_prompt,
    build_jd_extract_prompt,
    build_question_generation_prompt,
    build_resume_extract_prompt,
    build_resume_optimize_prompt,
)
from interview_ai.providers import LLMProvider, OpenRouterLLMProvider
from interview_ai.schemas import (
    AnswerEvaluation,
    CandidateProfile,
    FollowUpDecision,
    InterviewPlan,
    InterviewQuestion,
    InterviewTopicPlan,
    JobProfile,
    ResumeOptimization,
)

logger = logging.getLogger(__name__)


# ── JSON recovery helpers ─────────────────────────────────────────────────────

def _extract_json_block(text: str) -> str:
    """Strip markdown fences and extract the outermost JSON value.

    Strategy:
    1. Try parsing the cleaned text as-is (cheapest).
    2. Find the outermost { ... } — object pattern, searched from the first { to
       the matching closing } to avoid capturing mid-object arrays.
    3. Find the outermost [ ... ] as a fallback.
    This ordering prevents inner array values inside an object from being
    mistakenly extracted as the top-level value.
    """
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    # Fast path: already valid JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Find the outermost { } block by tracking brace depth
    for start_ch, end_ch in (('{', '}'), ('[', ']')):
        start = text.find(start_ch)
        if start == -1:
            continue
        depth = 0
        in_str = False
        escape = False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == '\\' and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_ch:
                depth += 1
            elif ch == end_ch:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text


def _try_parse_json(raw: str) -> Any:
    """Return parsed JSON or raise ValueError."""
    cleaned = _extract_json_block(raw)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON parse failed: {exc}\nRaw: {raw[:300]}") from exc


# ── Pipeline ──────────────────────────────────────────────────────────────────

class InterviewPipeline:
    """Runtime pipeline backed by any LLMProvider.

    Supply a provider instance explicitly, or let it create the default
    OpenRouter provider from environment variables.

    All public methods return typed Pydantic models (primary path) and log a
    warning when falling back to heuristic parsers.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        # Legacy compatibility: accept an OpenRouterClient-like object via `client=`
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        if provider is not None:
            self.provider = provider
        elif client is not None:
            # Wrap legacy client: delegate generate() to client.chat()
            self.provider = _LegacyClientAdapter(client)
        else:
            self.provider = OpenRouterLLMProvider(**client_kwargs)

        # Keep .client for code that still accesses pipeline.client directly
        # (AudioAnalyzer constructor, etc.)
        self._legacy_client = client

    @property
    def client(self) -> Any:
        """Backwards-compat property for AudioAnalyzer and other consumers."""
        return self._legacy_client

    def _generate(self, user_prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.provider.generate(messages, max_tokens=max_new_tokens)

    # ── Resume extraction ─────────────────────────────────────────────────────

    def extract_resume_profile(self, resume_file: str) -> CandidateProfile:
        """Extract CandidateProfile using LLM; fall back to heuristic parser."""
        resume_text = read_any_text(resume_file)
        prompt = build_resume_extract_prompt(resume_text)

        try:
            raw = self._generate(prompt, max_new_tokens=1024)
            data = _try_parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object for CandidateProfile")
            # Flatten nested candidate_profile key if present
            if "candidate_profile" in data and isinstance(data["candidate_profile"], dict):
                nested = data.pop("candidate_profile")
                data = {**nested, **data}
            profile = CandidateProfile.model_validate(data)
            logger.debug("extract_resume_profile: LLM path succeeded for %s", resume_file)
            return profile
        except Exception as exc:
            logger.warning(
                "extract_resume_profile: LLM path failed (%s); using weak fallback for %s",
                exc,
                resume_file,
            )
            return parse_resume_weak(resume_file)

    def extract_resume(self, resume_file: str) -> dict[str, Any]:
        """Legacy method: returns {'weak_label': ..., 'model_output': ..., 'profile': ...}."""
        weak = parse_resume_weak(resume_file)
        profile = self.extract_resume_profile(resume_file)
        raw = self._generate(build_resume_extract_prompt(read_any_text(resume_file)), max_new_tokens=1024)
        return {
            "weak_label": weak.model_dump(),
            "model_output": raw,
            "profile": profile,
        }

    # ── JD extraction ─────────────────────────────────────────────────────────

    def extract_job_profile(self, jd_file: str) -> JobProfile:
        """Extract JobProfile from a JD file using LLM; fall back to empty profile."""
        jd_text = read_any_text(jd_file)
        prompt = build_jd_extract_prompt(jd_text)

        try:
            raw = self._generate(prompt, max_new_tokens=1024)
            data = _try_parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object for JobProfile")
            job_profile = JobProfile.model_validate(data)
            logger.debug("extract_job_profile: LLM path succeeded for %s", jd_file)
            return job_profile
        except Exception as exc:
            logger.warning(
                "extract_job_profile: LLM path failed (%s); using empty JobProfile for %s",
                exc,
                jd_file,
            )
            return JobProfile()

    # ── Interview planning ────────────────────────────────────────────────────

    def plan_interview(
        self,
        candidate: CandidateProfile,
        job: JobProfile,
        session_id: str,
        max_questions: int = 5,
    ) -> InterviewPlan:
        """Build an InterviewPlan from CandidateProfile + JobProfile.

        Uses the planning LLM to generate grounded questions; falls back to
        generate_questions_weak if the LLM output is unparseable.
        """
        candidate_json = json.dumps(candidate.model_dump(), ensure_ascii=False, indent=2)
        job_json = json.dumps(job.model_dump(), ensure_ascii=False, indent=2)
        prompt = build_interview_plan_prompt(candidate_json, job_json)

        questions: list[InterviewQuestion] = []
        used_llm = True
        try:
            raw = self._generate(prompt, max_new_tokens=2048)
            data = _try_parse_json(raw)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array; got {type(data).__name__}")
            for item in data[:max_questions]:
                try:
                    questions.append(InterviewQuestion.model_validate(item))
                except Exception as item_err:
                    logger.warning("plan_interview: skipping invalid question item: %s", item_err)
            if not questions:
                raise ValueError("No valid questions parsed from LLM output")
        except Exception as exc:
            logger.warning(
                "plan_interview: LLM path failed (%s); using weak fallback.", exc
            )
            used_llm = False
            jd_text = job_json  # best we have without raw file
            questions = generate_questions_weak(candidate, jd_text)[:max_questions]

        if used_llm:
            logger.info(
                "plan_interview: %d questions from LLM (session %s)", len(questions), session_id
            )
        else:
            logger.info(
                "plan_interview: %d questions from FALLBACK (session %s)", len(questions), session_id
            )

        topics = [
            InterviewTopicPlan(
                skill=q.skill_target,
                jd_importance=0.8 if q.type == "technical" else 0.5,
                should_verify=True,
                question=q,
                expected_signals=q.expected_keywords,
            )
            for q in questions
        ]

        return InterviewPlan(
            session_id=session_id,
            candidate_name=candidate.name,
            target_role=job.role,
            topics=topics,
            planned_questions=questions,
        )

    # ── Question generation (legacy single-call path) ─────────────────────────

    def generate_questions(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        """Legacy API: returns {'weak_label': [...], 'model_output': str, 'questions': [...]}.

        Prefer plan_interview() for new code.
        """
        candidate = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        weak_questions = generate_questions_weak(candidate, jd_text)
        weak_json = [q.model_dump() for q in weak_questions]

        prompt = build_question_generation_prompt(
            json.dumps(candidate.model_dump(), ensure_ascii=False), jd_text
        )
        raw = self._generate(prompt, max_new_tokens=1500)

        questions: list[InterviewQuestion] = []
        try:
            data = _try_parse_json(raw)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array; got {type(data).__name__}")
            for item in data:
                try:
                    questions.append(InterviewQuestion.model_validate(item))
                except Exception as item_err:
                    logger.warning("generate_questions: skipping invalid item: %s", item_err)
            if not questions:
                raise ValueError("No valid questions parsed")
            logger.info("generate_questions: %d questions from LLM", len(questions))
        except Exception as exc:
            logger.warning(
                "generate_questions: LLM output invalid (%s); falling back to weak questions.", exc
            )
            questions = weak_questions

        return {
            "weak_label": weak_json,
            "model_output": raw,
            "questions": questions,
        }

    # ── Answer evaluation ─────────────────────────────────────────────────────

    def evaluate_answer_structured(
        self,
        question: str,
        answer: str,
        rubric: str,
        duration_seconds: float = 0.0,
        cv_evidence: str = "",
        jd_requirement: str = "",
    ) -> AnswerEvaluation:
        """Evaluate answer and return a validated AnswerEvaluation."""
        from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words

        prompt = build_answer_evaluation_prompt(
            question, answer, rubric,
            cv_evidence=cv_evidence,
            jd_requirement=jd_requirement,
        )
        raw = self._generate(prompt)

        try:
            data = _try_parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object for AnswerEvaluation")
            return AnswerEvaluation.model_validate(data)
        except Exception as exc:
            logger.warning(
                "evaluate_answer_structured: LLM output invalid (%s); using score heuristics.", exc
            )
            fluency = compute_fluency_score(answer, duration_seconds) if duration_seconds else 0.5
            return AnswerEvaluation(
                relevance_score=0.5,
                communication_score=fluency,
                confidence_score=0.5,
                overall_score=0.5,
                strengths=[],
                improvements=["Không thể đánh giá tự động — vui lòng xem xét thủ công."],
            )

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        rubric: str,
        duration_seconds: float,
    ) -> dict[str, Any]:
        """Legacy API — returns {'metrics': ..., 'model_output': str, 'evaluation': ...}."""
        from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words

        metrics = {
            "wpm": compute_wpm(answer, duration_seconds),
            "filler_count": count_filler_words(answer),
            "fluency_score": compute_fluency_score(answer, duration_seconds),
        }
        prompt = build_answer_evaluation_prompt(question, answer, rubric)
        raw = self._generate(prompt)

        evaluation = None
        try:
            data = _try_parse_json(raw)
            evaluation = AnswerEvaluation.model_validate(data)
        except Exception as exc:
            logger.warning("evaluate_answer: LLM output invalid (%s).", exc)

        return {"metrics": metrics, "model_output": raw, "evaluation": evaluation}

    # ── Follow-up decision ────────────────────────────────────────────────────

    def decide_follow_up(
        self,
        current_question: str,
        answer: str,
        transcript_history: str,
        cv_evidence: str = "",
        jd_requirement: str = "",
        expected_signals: list[str] | None = None,
    ) -> FollowUpDecision:
        """Return a validated FollowUpDecision from the LLM."""
        prompt = build_follow_up_prompt(
            current_question,
            answer,
            transcript_history,
            cv_evidence=cv_evidence,
            jd_requirement=jd_requirement,
            expected_signals=expected_signals,
        )
        raw = self._generate(prompt)

        try:
            data = _try_parse_json(raw)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object for FollowUpDecision")
            return FollowUpDecision.model_validate(data)
        except Exception as exc:
            logger.warning(
                "decide_follow_up: LLM output invalid (%s); defaulting to next_question.", exc
            )
            return FollowUpDecision(
                next_action="next_question",
                follow_up_question="",
                rationale="Không thể phân tích kết quả LLM — chuyển câu hỏi tiếp theo.",
            )

    def generate_follow_up_question(
        self,
        current_question: str,
        answer: str,
        transcript_history: str,
    ) -> dict[str, Any]:
        """Legacy API — returns {'model_output': str, 'decision': FollowUpDecision}."""
        decision = self.decide_follow_up(current_question, answer, transcript_history)
        prompt = build_follow_up_prompt(current_question, answer, transcript_history)
        raw = self._generate(prompt)
        return {"model_output": raw, "decision": decision}

    # ── Misc legacy methods ───────────────────────────────────────────────────

    def optimize_resume(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        weak_json = match_resume_to_jd(resume, jd_text).model_dump()
        prompt = build_resume_optimize_prompt(
            json.dumps(resume.model_dump(), ensure_ascii=False), jd_text
        )
        raw = self._generate(prompt)

        try:
            data = _try_parse_json(raw)
            optimization = ResumeOptimization.model_validate(data)
        except Exception as exc:
            logger.warning("optimize_resume: LLM output invalid (%s); using weak fallback.", exc)
            optimization = match_resume_to_jd(resume, jd_text)

        return {"weak_label": weak_json, "model_output": raw, "optimization": optimization}

    def build_full_context(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        return {
            "resume": resume.model_dump(),
            "jd_text": jd_text,
            "resume_optimization": match_resume_to_jd(resume, jd_text).model_dump(),
            "questions": [q.model_dump() for q in generate_questions_weak(resume, jd_text)],
        }


# ── Legacy client adapter ─────────────────────────────────────────────────────

class _LegacyClientAdapter(LLMProvider):
    """Wraps an OpenRouterClient so it satisfies the LLMProvider interface."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def model_id(self) -> str:
        return getattr(self._client, "chat_model", "unknown")

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        return self._client.chat(messages, max_tokens=max_tokens, temperature=temperature)
