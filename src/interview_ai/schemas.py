from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Resume / Candidate ────────────────────────────────────────────────────────

class ExperienceItem(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    achievements: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""
    year: str = ""


class SkillWithEvidence(BaseModel):
    name: str
    evidence: list[str] = Field(default_factory=list)
    inferred: bool = False


class CandidateProfile(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    total_years_experience: float | None = None
    skills: list[str] = Field(default_factory=list)
    skills_with_evidence: list[SkillWithEvidence] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


# ── Job Description ───────────────────────────────────────────────────────────

class JobProfile(BaseModel):
    role: str = ""
    seniority: str = ""
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    competencies: list[str] = Field(default_factory=list)
    domain: str = ""
    important_requirements: list[str] = Field(default_factory=list)


# ── Interview Planning ────────────────────────────────────────────────────────

class InterviewQuestion(BaseModel):
    question_id: str
    type: Literal["intro", "technical", "behavioral", "follow_up"]
    skill_target: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question: str
    expected_keywords: list[str] = Field(default_factory=list)


class InterviewTopicPlan(BaseModel):
    skill: str
    jd_importance: float = 0.5
    cv_evidence: str = ""
    should_verify: bool = True
    question: InterviewQuestion | None = None
    expected_signals: list[str] = Field(default_factory=list)
    rubric: dict[str, Any] = Field(default_factory=dict)


class InterviewPlan(BaseModel):
    session_id: str
    candidate_name: str = ""
    target_role: str = ""
    topics: list[InterviewTopicPlan] = Field(default_factory=list)
    planned_questions: list[InterviewQuestion] = Field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return len(self.planned_questions)

    def question_texts(self) -> list[str]:
        return [q.question for q in self.planned_questions]


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvaluationOutput(BaseModel):
    relevance_score: float
    communication_score: float
    confidence_score: float
    overall_score: float
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


# Alias for explicit naming
AnswerEvaluation = EvaluationOutput


class FollowUpDecision(BaseModel):
    next_action: Literal["follow_up", "next_question", "end_interview"]
    follow_up_question: str = ""
    rationale: str = ""


# ── Audio / Vision ────────────────────────────────────────────────────────────

class AudioAnalysis(BaseModel):
    transcript: str = ""
    duration_seconds: float = 0.0
    wpm: float = 0.0
    filler_count: int = 0
    fluency_score: float = 0.0
    pause_ratio: float = 0.0
    provider: str = ""


class STTResult(BaseModel):
    transcript: str
    duration: float = 0.0
    provider: str = ""
    confidence: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class VisionFrameMetric(BaseModel):
    timestamp: float
    eye_open_ratio: float = 0.0
    smile_score: float = 0.0
    gaze_forward_score: float = 0.0
    head_down_score: float = 0.0
    face_present: bool = True


class VisionAnalysis(BaseModel):
    frames: list[VisionFrameMetric] = Field(default_factory=list)
    attention_score: float = 0.0
    eye_contact_score: float = 0.0
    smile_frequency: float = 0.0
    head_down_ratio: float = 0.0
    face_presence_ratio: float = 0.0


class FusionScore(BaseModel):
    content_score: float
    behavior_score: float
    speaking_score: float
    final_score: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


# ── Session / Report ──────────────────────────────────────────────────────────

class ResumeOptimization(BaseModel):
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    suggested_resume_improvements: list[str] = Field(default_factory=list)


class ResumeExtractionOutput(BaseModel):
    candidate_profile: CandidateProfile
    jd_match: ResumeOptimization | None = None


class InterviewSessionReport(BaseModel):
    session_id: str
    candidate_profile: dict[str, Any]
    questions: list[dict[str, Any]]
    audio_analysis: AudioAnalysis
    vision_analysis: VisionAnalysis
    llm_evaluation: dict[str, Any]
    fusion_score: FusionScore
    report_summary: dict[str, Any] = Field(default_factory=dict)


class InterviewTurnRecord(BaseModel):
    turn_index: int
    question: str
    question_type: str = "technical"
    skill_target: str = ""
    answer_audio_file: str = ""
    transcript: str = ""
    audio_metrics: dict[str, Any] = Field(default_factory=dict)
    llm_evaluation: dict[str, Any] = Field(default_factory=dict)
    next_question: str = ""
    next_action: Literal["next_question", "follow_up", "end_interview"] = "next_question"


class ConversationState(BaseModel):
    session_id: str
    resume_file: str
    jd_file: str
    candidate_profile: CandidateProfile | None = None
    job_profile: JobProfile | None = None
    interview_plan: InterviewPlan | None = None
    current_question: str = ""
    current_question_obj: InterviewQuestion | None = None
    planned_questions: list[str] = Field(default_factory=list)
    planned_question_objs: list[InterviewQuestion] = Field(default_factory=list)
    completed_turns: list[InterviewTurnRecord] = Field(default_factory=list)
    follow_up_budget: int = 2
    max_turns: int = 5


# ── Training / SFT ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTRecord(BaseModel):
    messages: list[ChatMessage]
    task: str
    meta: dict[str, Any] = Field(default_factory=dict)
