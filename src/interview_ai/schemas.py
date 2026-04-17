from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperienceItem(BaseModel):
    company: str = ""
    role: str = ""
    duration: str = ""
    highlights: list[str] = Field(default_factory=list)


class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""
    year: str = ""


class CandidateProfile(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[ExperienceItem] = Field(default_factory=list)
    education: list[EducationItem] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


class ResumeOptimization(BaseModel):
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    suggested_resume_improvements: list[str] = Field(default_factory=list)


class ResumeExtractionOutput(BaseModel):
    candidate_profile: CandidateProfile
    jd_match: ResumeOptimization | None = None


class InterviewQuestion(BaseModel):
    question_id: str
    type: Literal["intro", "technical", "behavioral", "follow_up"]
    skill_target: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    question: str
    expected_keywords: list[str] = Field(default_factory=list)


class EvaluationOutput(BaseModel):
    relevance_score: float
    communication_score: float
    confidence_score: float
    overall_score: float
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


class AudioAnalysis(BaseModel):
    transcript: str = ""
    duration_seconds: float = 0.0
    wpm: float = 0.0
    filler_count: int = 0
    fluency_score: float = 0.0
    pause_ratio: float = 0.0


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


class InterviewSessionReport(BaseModel):
    session_id: str
    candidate_profile: dict[str, Any]
    questions: list[dict[str, Any]]
    audio_analysis: AudioAnalysis
    vision_analysis: VisionAnalysis
    llm_evaluation: dict[str, Any]
    fusion_score: FusionScore
    report_summary: dict[str, Any] = Field(default_factory=dict)


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTRecord(BaseModel):
    messages: list[ChatMessage]
    task: str
    meta: dict[str, Any] = Field(default_factory=dict)
