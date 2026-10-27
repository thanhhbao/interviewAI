"""Turn-based interview conversation manager.

Uses InterviewPlan (structured LLM output) as the question source.
Raw CV/JD text is only read once during start_session — it is NOT resent
on every turn. Relevant structured context is passed per-turn instead.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from interview_ai.pipeline import InterviewPipeline
from interview_ai.providers import STTProvider, TTSProvider, StubTTSProvider
from interview_ai.schemas import (
    AudioAnalysis,
    ConversationState,
    FollowUpDecision,
    InterviewQuestion,
    InterviewTopicPlan,
    InterviewTurnRecord,
)

logger = logging.getLogger(__name__)

_DEFAULT_RUBRIC = (
    "Đánh giá dựa trên mức độ liên quan đến câu hỏi, kỹ năng giao tiếp, "
    "sự tự tin, và độ rõ ràng về chuyên môn kỹ thuật."
)


class TurnBasedConversationManager:
    def __init__(
        self,
        llm_pipeline: InterviewPipeline,
        stt_provider: STTProvider | None = None,
        tts_provider: TTSProvider | None = None,
        # Legacy compat: accept audio_analyzer and tts_service kwargs
        audio_analyzer: object | None = None,
        tts_service: object | None = None,
    ) -> None:
        self.llm_pipeline = llm_pipeline

        # STT: prefer stt_provider, then wrap audio_analyzer as legacy fallback
        if stt_provider is not None:
            self._stt = stt_provider
        elif audio_analyzer is not None:
            self._stt = _LegacyAudioAnalyzerAdapter(audio_analyzer)
        else:
            # Build from pipeline's client if available
            from interview_ai.audio import AudioAnalyzer
            self._stt = _LegacyAudioAnalyzerAdapter(
                AudioAnalyzer(client=llm_pipeline.client)
            )

        # TTS: prefer tts_provider, then wrap tts_service
        if tts_provider is not None:
            self._tts = tts_provider
        elif tts_service is not None:
            self._tts = _LegacyTTSAdapter(tts_service)
        else:
            self._tts = StubTTSProvider()

    # ── Session start ─────────────────────────────────────────────────────────

    def start_session(
        self,
        resume_file: str,
        jd_file: str,
        max_turns: int = 5,
        follow_up_budget: int = 2,
    ) -> ConversationState:
        """Build a ConversationState driven by an LLM InterviewPlan.

        1. Extract CandidateProfile from resume (LLM path, heuristic fallback)
        2. Extract JobProfile from JD (LLM path, heuristic fallback)
        3. Plan interview questions grounded in CV + JD evidence
        4. Store structured context in state — NOT raw text
        """
        session_id = datetime.utcnow().strftime("conversation-%Y%m%d-%H%M%S")

        candidate = self.llm_pipeline.extract_resume_profile(resume_file)
        logger.info("start_session [%s]: candidate=%s", session_id, candidate.name or "(unknown)")

        job = self.llm_pipeline.extract_job_profile(jd_file)
        logger.info("start_session [%s]: role=%s", session_id, job.role or "(unknown)")

        plan = self.llm_pipeline.plan_interview(
            candidate=candidate,
            job=job,
            session_id=session_id,
            max_questions=max_turns,
        )
        logger.info(
            "start_session [%s]: %d questions planned", session_id, len(plan.planned_questions)
        )

        questions = list(plan.planned_questions)
        current_q = questions[0] if questions else InterviewQuestion(
            question_id="q0",
            type="intro",
            question="Bạn có thể giới thiệu về bản thân và lý do ứng tuyển vị trí này không?",
        )

        return ConversationState(
            session_id=session_id,
            resume_file=resume_file,
            jd_file=jd_file,
            candidate_profile=candidate,
            job_profile=job,
            interview_plan=plan,
            current_question=current_q.question,
            current_question_obj=current_q,
            planned_questions=[q.question for q in questions[1:]],
            planned_question_objs=questions[1:],
            follow_up_budget=follow_up_budget,
            max_turns=max_turns,
        )

    # ── TTS ───────────────────────────────────────────────────────────────────

    def speak_current_question(self, state: ConversationState, output_file: str) -> str:
        return self._tts.synthesize(state.current_question, output_file)

    # ── Answer turn ───────────────────────────────────────────────────────────

    def process_answer_turn(
        self,
        state: ConversationState,
        answer_audio_file: str,
        rubric: str | None = None,
    ) -> ConversationState:
        """Process one answer turn and advance the conversation state."""
        # 1. Transcribe
        audio = self._transcribe(answer_audio_file)

        # 2. Gather per-turn context from structured plan (no raw CV/JD resent)
        cv_evidence = ""
        jd_requirement = ""
        expected_signals: list[str] = []
        current_q_obj = state.current_question_obj

        if current_q_obj and state.interview_plan:
            topic = _find_topic(state.interview_plan.topics, current_q_obj.question_id)
            if topic:
                cv_evidence = topic.cv_evidence
                expected_signals = topic.expected_signals
            if state.job_profile:
                jd_requirement = "; ".join(state.job_profile.important_requirements[:3])

        # 3. Evaluate answer
        evaluation = self.llm_pipeline.evaluate_answer_structured(
            question=state.current_question,
            answer=audio.transcript,
            rubric=rubric or _DEFAULT_RUBRIC,
            duration_seconds=audio.duration_seconds,
            cv_evidence=cv_evidence,
            jd_requirement=jd_requirement,
        )

        # 4. Follow-up / next-question decision
        transcript_history = _build_history(state)
        decision: FollowUpDecision = self.llm_pipeline.decide_follow_up(
            current_question=state.current_question,
            answer=audio.transcript,
            transcript_history=transcript_history,
            cv_evidence=cv_evidence,
            jd_requirement=jd_requirement,
            expected_signals=expected_signals or None,
        )

        # 5. Apply decision
        next_action, next_question, next_question_obj = _apply_decision(state, decision)

        # 6. Record turn
        turn = InterviewTurnRecord(
            turn_index=len(state.completed_turns) + 1,
            question=state.current_question,
            question_type=current_q_obj.type if current_q_obj else "technical",
            skill_target=current_q_obj.skill_target if current_q_obj else "",
            answer_audio_file=answer_audio_file,
            transcript=audio.transcript,
            audio_metrics=audio.model_dump(),
            llm_evaluation={
                "evaluation": evaluation.model_dump(),
                "follow_up_decision": decision.model_dump(),
            },
            next_question=next_question,
            next_action=next_action,
        )
        state.completed_turns.append(turn)
        state.current_question = next_question
        state.current_question_obj = next_question_obj
        return state

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, state: ConversationState, output_file: str) -> str:
        target = Path(output_file)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(target)

    def summarize_state(self, state: ConversationState) -> dict:
        return {
            "session_id": state.session_id,
            "candidate_name": state.candidate_profile.name if state.candidate_profile else "",
            "target_role": state.job_profile.role if state.job_profile else "",
            "completed_turns": len(state.completed_turns),
            "current_question": state.current_question,
            "remaining_planned_questions": len(state.planned_question_objs),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transcribe(self, audio_file: str) -> AudioAnalysis:
        """Transcribe audio through the STT provider and return AudioAnalysis."""
        from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words

        result = self._stt.transcribe(audio_file)
        # STTProvider returns STTResult; legacy adapter may return AudioAnalysis
        if isinstance(result, AudioAnalysis):
            return result

        transcript = result.transcript
        duration = result.duration
        return AudioAnalysis(
            transcript=transcript,
            duration_seconds=duration,
            wpm=compute_wpm(transcript, duration),
            filler_count=count_filler_words(transcript),
            fluency_score=compute_fluency_score(transcript, duration),
            pause_ratio=0.1,
            provider=result.provider,
        )


# ── Pure helper functions ─────────────────────────────────────────────────────

def _build_history(state: ConversationState) -> str:
    return "\n".join(
        f"Q: {t.question}\nA: {t.transcript}" for t in state.completed_turns[-4:]
    )


def _find_topic(
    topics: list[InterviewTopicPlan], question_id: str
) -> InterviewTopicPlan | None:
    for t in topics:
        if t.question and t.question.question_id == question_id:
            return t
    return None


def _apply_decision(
    state: ConversationState,
    decision: FollowUpDecision,
) -> tuple[str, str, InterviewQuestion | None]:
    """Return (next_action, next_question_text, next_question_obj)."""
    total_turns_after = len(state.completed_turns) + 1

    if (
        decision.next_action == "follow_up"
        and state.follow_up_budget > 0
        and decision.follow_up_question
    ):
        state.follow_up_budget -= 1
        fq = InterviewQuestion(
            question_id=f"fu{total_turns_after}",
            type="follow_up",
            question=decision.follow_up_question,
        )
        return "follow_up", decision.follow_up_question, fq

    if state.planned_question_objs and total_turns_after < state.max_turns:
        nxt = state.planned_question_objs.pop(0)
        if state.planned_questions:
            state.planned_questions.pop(0)
        return "next_question", nxt.question, nxt

    return "end_interview", "", None


# ── Legacy adapters ───────────────────────────────────────────────────────────

class _LegacyAudioAnalyzerAdapter:
    """Adapts the old AudioAnalyzer to the STTProvider interface."""

    def __init__(self, analyzer: object) -> None:
        self._analyzer = analyzer

    def transcribe(self, audio_path: str) -> AudioAnalysis:
        return self._analyzer.analyze(audio_path)  # type: ignore[attr-defined]


class _LegacyTTSAdapter:
    """Adapts LocalTTSService to the TTSProvider interface."""

    def __init__(self, service: object) -> None:
        self._service = service

    def synthesize(self, text: str, output_file: str) -> str:
        return self._service.synthesize(text, output_file)  # type: ignore[attr-defined]
