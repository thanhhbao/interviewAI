"""Unit tests for the refactored interview pipeline.

All tests mock the LLM/STT/TTS providers — no real API calls are made.

Coverage:
  1.  LLM-generated question JSON is actually used
  2.  Invalid question JSON falls back to weak questions
  3.  CandidateProfile validation
  4.  JobProfile validation
  5.  InterviewPlan validation
  6.  FollowUpDecision parsing
  7.  Same LLM provider can serve planning / realtime / evaluation roles
  8.  STT and TTS providers can be mocked
  9.  Vietnamese Unicode prompts are preserved
  10. Raw CV/JD are not resent on each interview turn
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from interview_ai.providers import LLMProvider, STTProvider, TTSProvider, StubTTSProvider
from interview_ai.schemas import (
    AudioAnalysis,
    CandidateProfile,
    ConversationState,
    FollowUpDecision,
    InterviewPlan,
    InterviewQuestion,
    JobProfile,
    STTResult,
)


# ── Shared fixtures & helpers ─────────────────────────────────────────────────

VALID_QUESTIONS_JSON = json.dumps([
    {
        "question_id": "q1",
        "type": "technical",
        "skill_target": "FastAPI",
        "difficulty": "hard",
        "question": "Bạn đã xử lý bottleneck trong hệ thống FastAPI như thế nào?",
        "expected_keywords": ["async", "profiling", "connection pool", "cache", "load test"],
    },
    {
        "question_id": "q2",
        "type": "behavioral",
        "skill_target": "Leadership",
        "difficulty": "medium",
        "question": "Kể về một lần bạn phải thuyết phục team thay đổi kiến trúc hệ thống.",
        "expected_keywords": ["trade-off", "communication", "stakeholder", "decision", "impact"],
    },
])

VALID_CANDIDATE_JSON = json.dumps({
    "name": "Nguyễn Văn A",
    "email": "a@example.com",
    "phone": "0912345678",
    "summary": "Backend engineer với 5 năm kinh nghiệm Python.",
    "total_years_experience": 5.0,
    "skills": ["Python", "FastAPI", "PostgreSQL"],
    "skills_with_evidence": [
        {"name": "FastAPI", "evidence": ["API phục vụ 20k req/ngày"], "inferred": False}
    ],
    "experience": [],
    "education": [],
    "projects": [],
})

VALID_JOB_JSON = json.dumps({
    "role": "Senior Backend Engineer",
    "seniority": "Senior",
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "preferred_skills": ["Docker", "Kubernetes"],
    "responsibilities": ["Design REST APIs", "Mentor junior devs"],
    "competencies": ["System design", "Code review"],
    "domain": "fintech",
    "important_requirements": ["5+ years experience", "distributed systems"],
})

VALID_FOLLOW_UP_JSON = json.dumps({
    "next_action": "follow_up",
    "follow_up_question": "Bạn có thể chia sẻ cụ thể hơn về cách bạn đo hiệu năng trước và sau khi tối ưu không?",
    "rationale": "Câu trả lời còn thiếu số liệu cụ thể về kết quả đạt được.",
})

VALID_EVAL_JSON = json.dumps({
    "relevance_score": 0.85,
    "communication_score": 0.75,
    "confidence_score": 0.80,
    "overall_score": 0.80,
    "strengths": ["Hiểu rõ vấn đề", "Đề xuất giải pháp cụ thể"],
    "improvements": ["Cần thêm số liệu định lượng"],
})


class MockLLMProvider(LLMProvider):
    """Configurable mock: returns responses in order, then repeats the last one."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[list[dict]] = []

    @property
    def model_id(self) -> str:
        return "mock/test-model"

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append(messages)
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        return response


class MockSTTProvider(STTProvider):
    def __init__(self, transcript: str = "Câu trả lời mẫu.") -> None:
        self.transcript = transcript
        self.call_count = 0

    def transcribe(self, audio_path: Any) -> STTResult:
        self.call_count += 1
        return STTResult(transcript=self.transcript, duration=30.0, provider="mock_stt")


class MockTTSProvider(TTSProvider):
    def __init__(self) -> None:
        self.synthesized: list[str] = []

    def synthesize(self, text: str, output_file: str) -> str:
        self.synthesized.append(text)
        return output_file


def _make_pipeline(responses: list[str]):
    from interview_ai.pipeline import InterviewPipeline
    return InterviewPipeline(provider=MockLLMProvider(responses))


def _resume_file(tmp_path: Path) -> str:
    p = tmp_path / "resume.txt"
    p.write_text(
        "Nguyễn Văn A\na@example.com\n0912345678\n"
        "Python developer with 5 years of experience in FastAPI and PostgreSQL.",
        encoding="utf-8",
    )
    return str(p)


def _jd_file(tmp_path: Path) -> str:
    p = tmp_path / "jd.txt"
    p.write_text(
        "Senior Backend Engineer — Fintech\n"
        "Required: Python, FastAPI, PostgreSQL, 5+ years experience.\n"
        "Preferred: Docker, Kubernetes.",
        encoding="utf-8",
    )
    return str(p)


# ── 1. LLM-generated questions are actually used ──────────────────────────────

def test_llm_questions_used_in_plan(tmp_path: Path) -> None:
    """plan_interview() must use LLM questions, not weak fallback."""
    # plan_interview() calls the LLM once — return the questions array directly
    pipeline = _make_pipeline([VALID_QUESTIONS_JSON])
    candidate = CandidateProfile.model_validate(json.loads(VALID_CANDIDATE_JSON))
    job = JobProfile.model_validate(json.loads(VALID_JOB_JSON))

    plan = pipeline.plan_interview(candidate, job, session_id="test-001")

    assert len(plan.planned_questions) == 2
    assert plan.planned_questions[0].question_id == "q1"
    assert "FastAPI" in plan.planned_questions[0].question


def test_generate_questions_uses_llm_output(tmp_path: Path) -> None:
    """generate_questions() must parse and return LLM questions, not weak_label."""
    pipeline = _make_pipeline([VALID_QUESTIONS_JSON])
    resume = _resume_file(tmp_path)
    jd = _jd_file(tmp_path)

    result = pipeline.generate_questions(resume, jd)

    assert "questions" in result
    llm_questions = result["questions"]
    assert isinstance(llm_questions[0], InterviewQuestion)
    assert llm_questions[0].question_id == "q1"
    # Must NOT be the weak English placeholder questions
    assert "Hãy" in llm_questions[0].question or "bạn" in llm_questions[0].question.lower()


def test_conversation_start_uses_llm_plan(tmp_path: Path) -> None:
    """TurnBasedConversationManager.start_session() must use LLM plan, not weak_label."""
    from interview_ai.conversation import TurnBasedConversationManager

    pipeline = _make_pipeline([
        VALID_CANDIDATE_JSON,   # extract_resume_profile
        VALID_JOB_JSON,         # extract_job_profile
        VALID_QUESTIONS_JSON,   # plan_interview
    ])
    manager = TurnBasedConversationManager(
        llm_pipeline=pipeline,
        stt_provider=MockSTTProvider(),
        tts_provider=MockTTSProvider(),
    )
    state = manager.start_session(_resume_file(tmp_path), _jd_file(tmp_path))

    assert state.interview_plan is not None
    assert len(state.interview_plan.planned_questions) == 2
    # Current question should be the first LLM question
    assert state.current_question_obj is not None
    assert state.current_question_obj.question_id == "q1"


# ── 2. Invalid JSON falls back to weak questions ──────────────────────────────

def test_invalid_json_falls_back_to_weak(tmp_path: Path) -> None:
    """generate_questions() must fall back gracefully on bad LLM output."""
    pipeline = _make_pipeline(["This is not JSON at all ¯\\_(ツ)_/¯"])
    result = pipeline.generate_questions(_resume_file(tmp_path), _jd_file(tmp_path))

    assert "questions" in result
    # Weak fallback returns InterviewQuestion objects too
    assert all(isinstance(q, InterviewQuestion) for q in result["questions"])
    # There should still be questions (weak generator produces at least 1)
    assert len(result["questions"]) >= 1


def test_partial_json_array_falls_back(tmp_path: Path) -> None:
    """An invalid array item is skipped; if nothing valid remains, use weak fallback."""
    bad_array = json.dumps([{"bad_key": "no question or question_id"}])
    pipeline = _make_pipeline([bad_array])
    result = pipeline.generate_questions(_resume_file(tmp_path), _jd_file(tmp_path))

    assert len(result["questions"]) >= 1


# ── 3. CandidateProfile validation ───────────────────────────────────────────

def test_candidate_profile_valid() -> None:
    data = json.loads(VALID_CANDIDATE_JSON)
    profile = CandidateProfile.model_validate(data)
    assert profile.name == "Nguyễn Văn A"
    assert profile.total_years_experience == 5.0
    assert len(profile.skills_with_evidence) == 1
    assert profile.skills_with_evidence[0].name == "FastAPI"


def test_candidate_profile_minimal() -> None:
    profile = CandidateProfile()
    assert profile.name == ""
    assert profile.skills == []


def test_extract_resume_profile_uses_llm(tmp_path: Path) -> None:
    pipeline = _make_pipeline([VALID_CANDIDATE_JSON])
    profile = pipeline.extract_resume_profile(_resume_file(tmp_path))
    assert isinstance(profile, CandidateProfile)
    assert profile.name == "Nguyễn Văn A"


def test_extract_resume_profile_fallback(tmp_path: Path) -> None:
    """Falls back to heuristic parser when LLM output is invalid JSON."""
    pipeline = _make_pipeline(["not json"])
    profile = pipeline.extract_resume_profile(_resume_file(tmp_path))
    assert isinstance(profile, CandidateProfile)
    # Weak parser extracts something (at least returns a CandidateProfile)


# ── 4. JobProfile validation ──────────────────────────────────────────────────

def test_job_profile_valid() -> None:
    data = json.loads(VALID_JOB_JSON)
    job = JobProfile.model_validate(data)
    assert job.role == "Senior Backend Engineer"
    assert "Python" in job.required_skills
    assert job.seniority == "Senior"


def test_job_profile_empty() -> None:
    job = JobProfile()
    assert job.role == ""
    assert job.required_skills == []


def test_extract_job_profile_uses_llm(tmp_path: Path) -> None:
    pipeline = _make_pipeline([VALID_JOB_JSON])
    job = pipeline.extract_job_profile(_jd_file(tmp_path))
    assert isinstance(job, JobProfile)
    assert job.role == "Senior Backend Engineer"


def test_extract_job_profile_fallback(tmp_path: Path) -> None:
    """Falls back to empty JobProfile on invalid LLM output."""
    pipeline = _make_pipeline(["not json"])
    job = pipeline.extract_job_profile(_jd_file(tmp_path))
    assert isinstance(job, JobProfile)


# ── 5. InterviewPlan validation ───────────────────────────────────────────────

def test_interview_plan_valid() -> None:
    questions = [
        InterviewQuestion(
            question_id="q1", type="technical", question="Câu hỏi 1", skill_target="Python"
        ),
        InterviewQuestion(
            question_id="q2", type="behavioral", question="Câu hỏi 2"
        ),
    ]
    plan = InterviewPlan(
        session_id="s1",
        candidate_name="Nguyễn Văn A",
        target_role="Backend Engineer",
        planned_questions=questions,
    )
    assert plan.total_questions == 2
    assert plan.question_texts() == ["Câu hỏi 1", "Câu hỏi 2"]


def test_interview_plan_empty() -> None:
    plan = InterviewPlan(session_id="s0")
    assert plan.total_questions == 0
    assert plan.question_texts() == []


# ── 6. FollowUpDecision parsing ───────────────────────────────────────────────

def test_follow_up_decision_parsed(tmp_path: Path) -> None:
    pipeline = _make_pipeline([VALID_FOLLOW_UP_JSON])
    decision = pipeline.decide_follow_up(
        current_question="Bạn đã tối ưu hệ thống như thế nào?",
        answer="Tôi dùng caching.",
        transcript_history="",
    )
    assert isinstance(decision, FollowUpDecision)
    assert decision.next_action == "follow_up"
    assert "hiệu năng" in decision.follow_up_question or len(decision.follow_up_question) > 0


def test_follow_up_decision_invalid_json_defaults_to_next(tmp_path: Path) -> None:
    pipeline = _make_pipeline(["!!!not json!!!"])
    decision = pipeline.decide_follow_up("Q", "A", "")
    assert decision.next_action == "next_question"


def test_follow_up_decision_end_interview(tmp_path: Path) -> None:
    payload = json.dumps({
        "next_action": "end_interview",
        "follow_up_question": "",
        "rationale": "Phỏng vấn đã đủ thông tin.",
    })
    pipeline = _make_pipeline([payload])
    decision = pipeline.decide_follow_up("Q", "A long and thorough answer.", "")
    assert decision.next_action == "end_interview"


# ── 7. Same LLM provider for planning / realtime / evaluation ─────────────────

def test_single_provider_for_all_roles() -> None:
    """All three roles can use the same LLMProvider instance."""
    from interview_ai.config import AppConfig, ModelConfig, ModelsConfig

    cfg = AppConfig(
        models=ModelsConfig(
            planning=ModelConfig(provider="openrouter", model="openai/gpt-oss-20b"),
            realtime=ModelConfig(provider="openrouter", model="openai/gpt-oss-20b"),
            evaluation=ModelConfig(provider="openrouter", model="openai/gpt-oss-20b"),
        )
    )
    assert cfg.models.planning.model == cfg.models.realtime.model == cfg.models.evaluation.model

    # Create one provider and verify it can be reused for all roles
    provider = MockLLMProvider([VALID_EVAL_JSON])
    from interview_ai.pipeline import InterviewPipeline
    pipeline = InterviewPipeline(provider=provider)
    assert pipeline.provider is provider


def test_config_loads_default() -> None:
    from interview_ai.config import AppConfig, load_config
    cfg = load_config(path=None)
    assert isinstance(cfg, AppConfig)
    # Default provider is openrouter
    assert cfg.models.planning.provider == "openrouter"


# ── 8. STT and TTS providers can be mocked ───────────────────────────────────

def test_mock_stt_provider(tmp_path: Path) -> None:
    mock_stt = MockSTTProvider(transcript="Đây là câu trả lời mẫu.")
    result = mock_stt.transcribe(str(tmp_path / "fake.wav"))
    assert result.transcript == "Đây là câu trả lời mẫu."
    assert result.provider == "mock_stt"
    assert mock_stt.call_count == 1


def test_mock_tts_provider() -> None:
    mock_tts = MockTTSProvider()
    out = mock_tts.synthesize("Câu hỏi đầu tiên.", "/tmp/q.wav")
    assert out == "/tmp/q.wav"
    assert mock_tts.synthesized == ["Câu hỏi đầu tiên."]


def test_stub_tts_writes_text(tmp_path: Path) -> None:
    tts = StubTTSProvider()
    out_path = str(tmp_path / "out.txt")
    tts.synthesize("Xin chào!", out_path)
    assert Path(out_path).read_text(encoding="utf-8") == "Xin chào!"


def test_conversation_uses_mock_stt_tts(tmp_path: Path) -> None:
    from interview_ai.conversation import TurnBasedConversationManager

    mock_stt = MockSTTProvider(transcript="Tôi có 5 năm kinh nghiệm với Python.")
    mock_tts = MockTTSProvider()

    pipeline = _make_pipeline([
        VALID_CANDIDATE_JSON,
        VALID_JOB_JSON,
        VALID_QUESTIONS_JSON,
        VALID_EVAL_JSON,
        VALID_FOLLOW_UP_JSON,
    ])
    manager = TurnBasedConversationManager(
        llm_pipeline=pipeline,
        stt_provider=mock_stt,
        tts_provider=mock_tts,
    )
    state = manager.start_session(_resume_file(tmp_path), _jd_file(tmp_path))

    # Speak current question
    manager.speak_current_question(state, str(tmp_path / "q.txt"))
    assert state.current_question in mock_tts.synthesized

    # Process one answer turn
    audio_file = tmp_path / "answer.wav"
    audio_file.write_bytes(b"")
    state = manager.process_answer_turn(state, str(audio_file))
    assert mock_stt.call_count == 1
    assert len(state.completed_turns) == 1


# ── 9. Vietnamese Unicode prompts are preserved ───────────────────────────────

def test_vietnamese_prompts_contain_diacritics() -> None:
    from interview_ai.prompts import (
        SYSTEM_PROMPT,
        build_answer_evaluation_prompt,
        build_follow_up_prompt,
        build_interview_plan_prompt,
        build_jd_extract_prompt,
        build_question_generation_prompt,
        build_resume_extract_prompt,
    )

    prompts = [
        SYSTEM_PROMPT,
        build_resume_extract_prompt("CV text"),
        build_jd_extract_prompt("JD text"),
        build_interview_plan_prompt("{}", "{}"),
        build_question_generation_prompt("{}", "JD text"),
        build_answer_evaluation_prompt("Q", "A", "rubric"),
        build_follow_up_prompt("Q", "A", "history"),
    ]

    vietnamese_chars = set("àáảãạăắặẳẵặâấầẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")
    for i, prompt in enumerate(prompts):
        has_diacritics = any(c in vietnamese_chars for c in prompt.lower())
        assert has_diacritics, (
            f"Prompt {i} contains no Vietnamese diacritics — "
            f"may still use ASCII transliteration. First 100 chars: {prompt[:100]!r}"
        )


def test_no_telex_transliteration_in_prompts() -> None:
    """Prompts must NOT contain common un-accented Vietnamese patterns."""
    from interview_ai.prompts import build_question_generation_prompt, build_follow_up_prompt

    prompt1 = build_question_generation_prompt("{}", "JD text")
    prompt2 = build_follow_up_prompt("Q", "A", "history")

    banned_phrases = [
        "Hay dong vai",
        "nguoi phong van",
        "Danh gia dua tren",
        "Duoi vao",
    ]
    for phrase in banned_phrases:
        assert phrase not in prompt1, f"Found transliterated phrase in question prompt: {phrase!r}"
        assert phrase not in prompt2, f"Found transliterated phrase in follow-up prompt: {phrase!r}"


# ── 10. Raw CV/JD not resent on each turn ─────────────────────────────────────

def test_raw_cv_jd_not_in_turn_prompts(tmp_path: Path) -> None:
    """After start_session, process_answer_turn must NOT re-read raw CV/JD files."""
    from interview_ai.conversation import TurnBasedConversationManager
    from interview_ai.io import read_any_text

    resume = _resume_file(tmp_path)
    jd = _jd_file(tmp_path)
    resume_text = Path(resume).read_text(encoding="utf-8")
    jd_text = Path(jd).read_text(encoding="utf-8")

    capture: list[list[dict]] = []

    class CapturingProvider(LLMProvider):
        @property
        def model_id(self) -> str:
            return "mock/capture"

        def generate(self, messages, max_tokens=512, temperature=0.7) -> str:
            capture.extend(messages)
            # Return appropriate mock responses in order
            call_n = len(capture) // 2  # rough index
            responses = [
                VALID_CANDIDATE_JSON,
                VALID_JOB_JSON,
                VALID_QUESTIONS_JSON,
                VALID_EVAL_JSON,
                VALID_FOLLOW_UP_JSON,
            ]
            return responses[min(call_n, len(responses) - 1)]

    from interview_ai.pipeline import InterviewPipeline
    pipeline = InterviewPipeline(provider=CapturingProvider())

    manager = TurnBasedConversationManager(
        llm_pipeline=pipeline,
        stt_provider=MockSTTProvider(),
        tts_provider=MockTTSProvider(),
    )
    state = manager.start_session(resume, jd)
    capture.clear()  # reset after setup

    audio_file = tmp_path / "answer.wav"
    audio_file.write_bytes(b"")
    manager.process_answer_turn(state, str(audio_file))

    # Check that no turn-time prompt contains the raw full CV or JD text
    all_prompt_content = " ".join(
        m.get("content", "") for m in capture if isinstance(m, dict)
    )
    assert resume_text not in all_prompt_content, (
        "Raw resume text was resent in a turn-time LLM call"
    )
    assert jd_text not in all_prompt_content, (
        "Raw JD text was resent in a turn-time LLM call"
    )
