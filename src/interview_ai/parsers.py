from __future__ import annotations

import json
import re
from pathlib import Path

from interview_ai.io import read_any_text
from interview_ai.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    InterviewQuestion,
    ResumeOptimization,
)


SKILL_KEYWORDS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "react",
    "next.js",
    "node.js",
    "sql",
    "postgresql",
    "mongodb",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "gcp",
    "machine learning",
    "deep learning",
    "pytorch",
    "tensorflow",
    "fastapi",
    "django",
    "flask",
]


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _find_email(text: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return match.group(0) if match else ""


def _find_phone(text: str) -> str:
    match = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    return match.group(0) if match else ""


def _guess_name(text: str) -> str:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line[:80]


def _extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    return [skill for skill in SKILL_KEYWORDS if skill in lowered]


def parse_resume_weak(path: str | Path) -> CandidateProfile:
    text = normalize_whitespace(read_any_text(path))
    return CandidateProfile(
        name=_guess_name(text),
        email=_find_email(text),
        phone=_find_phone(text),
        summary=text[:500],
        skills=_extract_skills(text),
        experience=[
            ExperienceItem(
                company="Unknown",
                role="Candidate role",
                duration="Unknown",
                highlights=text[:300].split(". ")[:3],
            )
        ],
        education=[EducationItem(institution="Unknown", degree="Unknown", year="")],
        projects=[],
    )


def match_resume_to_jd(resume: CandidateProfile, jd_text: str) -> ResumeOptimization:
    jd_lower = jd_text.lower()
    matched = [skill for skill in resume.skills if skill.lower() in jd_lower]
    missing = [skill for skill in SKILL_KEYWORDS if skill in jd_lower and skill not in resume.skills]
    suggestions = []
    for skill in missing[:5]:
        suggestions.append(f"Add measurable evidence for {skill} in projects or experience.")
    if not resume.summary:
        suggestions.append("Add a professional summary tailored to the job description.")
    return ResumeOptimization(
        matched_skills=matched,
        missing_skills=missing,
        suggested_resume_improvements=suggestions,
    )


def generate_questions_weak(resume: CandidateProfile, jd_text: str) -> list[InterviewQuestion]:
    topics = resume.skills[:3] or _extract_skills(jd_text)[:3] or ["problem solving", "communication"]
    questions: list[InterviewQuestion] = [
        InterviewQuestion(
            question_id="q1",
            type="intro",
            skill_target="communication",
            difficulty="easy",
            question="Introduce yourself and summarize the most relevant experience for this role.",
            expected_keywords=["experience", "impact", "role"],
        )
    ]
    for index, topic in enumerate(topics, start=2):
        questions.append(
            InterviewQuestion(
                question_id=f"q{index}",
                type="technical",
                skill_target=topic,
                difficulty="medium",
                question=f"Explain a project where you used {topic} and describe the trade-offs you handled.",
                expected_keywords=[topic, "trade-off", "result"],
            )
        )
    questions.append(
        InterviewQuestion(
            question_id=f"q{len(questions) + 1}",
            type="behavioral",
            skill_target="teamwork",
            difficulty="medium",
            question="Describe a conflict in a team project and how you resolved it.",
            expected_keywords=["conflict", "resolution", "communication"],
        )
    )
    return questions[:5]


def load_answer_training_examples(answer_file: str | Path | None) -> list[dict]:
    if answer_file is None:
        return []
    records = []
    with Path(answer_file).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
