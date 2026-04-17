from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from interview_ai.augmentation import augment_resume_text, build_question_prompt_variants
from interview_ai.prompts import SYSTEM_PROMPT, build_resume_extract_prompt
from interview_ai.schemas import (
    CandidateProfile,
    ChatMessage,
    EducationItem,
    ExperienceItem,
    InterviewQuestion,
    SFTRecord,
)


RESUME_LABEL_MAP = {
    "name": "name",
    "college name": "education",
    "degree": "education",
    "graduation year": "education",
    "years of experience": "summary",
    "companies worked at": "experience",
    "designation": "experience",
    "skills": "skills",
    "location": "summary",
    "email address": "email",
}


QUESTION_COLUMN_CANDIDATES = {
    "question": ["question", "Question", "question_text"],
    "brief_answer": ["brief answer", "Brief Answer", "answer", "Answer", "brief_answer"],
    "category": ["category", "Category", "topic", "Topic"],
    "difficulty": ["difficulty", "Difficulty", "level"],
    "question_number": ["question number", "Question Number", "id", "ID", "No"],
}


def build_chat_record(task: str, user_prompt: str, assistant_content: str, meta: dict[str, Any] | None = None) -> dict:
    record = SFTRecord(
        task=task,
        messages=[
            ChatMessage(role="system", content=SYSTEM_PROMPT),
            ChatMessage(role="user", content=user_prompt),
            ChatMessage(role="assistant", content=assistant_content),
        ],
        meta=meta or {},
    )
    return record.model_dump()


def _extract_text_from_resume_item(item: dict[str, Any]) -> str:
    for key in ["content", "text", "document", "resume", "data"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_text_spans(text: str, annotation: dict[str, Any]) -> list[str]:
    results: list[str] = []
    labels = annotation.get("label") or annotation.get("labels") or []
    points = annotation.get("points") or []
    if not isinstance(points, list):
        return results

    for point in points:
        if not isinstance(point, dict):
            continue
        start = point.get("start")
        end = point.get("end")
        text_value = point.get("text")
        if isinstance(text_value, str) and text_value.strip():
            results.append(text_value.strip())
            continue
        if isinstance(start, int) and isinstance(end, int) and 0 <= start <= end < len(text):
            results.append(text[start : end + 1].strip())
    if not results and isinstance(labels, list) and labels:
        if isinstance(annotation.get("text"), str):
            results.append(annotation["text"].strip())
    return [item for item in results if item]


def _profile_from_resume_annotations(text: str, annotations: list[dict[str, Any]]) -> CandidateProfile:
    profile = CandidateProfile(summary=text[:500])
    experience_company: list[str] = []
    experience_role: list[str] = []
    education_institution: list[str] = []
    education_degree: list[str] = []
    education_year: list[str] = []

    for annotation in annotations:
        labels = annotation.get("label") or annotation.get("labels") or []
        if isinstance(labels, str):
            labels = [labels]
        values = _extract_text_spans(text, annotation)
        for label in labels:
            field = RESUME_LABEL_MAP.get(str(label).strip().lower())
            if not field:
                continue
            if field == "name" and values:
                profile.name = profile.name or values[0]
            elif field == "email" and values:
                profile.email = profile.email or values[0]
            elif field == "skills":
                for value in values:
                    split_values = re.split(r"[,/;\n]", value)
                    for skill in split_values:
                        skill = skill.strip()
                        if skill and skill not in profile.skills:
                            profile.skills.append(skill)
            elif field == "experience":
                normalized_label = str(label).strip().lower()
                if normalized_label == "companies worked at":
                    experience_company.extend(values)
                elif normalized_label == "designation":
                    experience_role.extend(values)
            elif field == "education":
                normalized_label = str(label).strip().lower()
                if normalized_label == "college name":
                    education_institution.extend(values)
                elif normalized_label == "degree":
                    education_degree.extend(values)
                elif normalized_label == "graduation year":
                    education_year.extend(values)
            elif field == "summary" and values:
                snippet = ", ".join(values)
                if snippet and snippet.lower() not in profile.summary.lower():
                    profile.summary = f"{profile.summary}\n{snippet}".strip()

    max_experience = max(len(experience_company), len(experience_role), 1)
    for index in range(max_experience):
        company = experience_company[index] if index < len(experience_company) else "Unknown"
        role = experience_role[index] if index < len(experience_role) else "Unknown"
        profile.experience.append(
            ExperienceItem(
                company=company,
                role=role,
                duration="Unknown",
                highlights=[],
            )
        )

    max_education = max(len(education_institution), len(education_degree), len(education_year), 1)
    for index in range(max_education):
        institution = education_institution[index] if index < len(education_institution) else "Unknown"
        degree = education_degree[index] if index < len(education_degree) else "Unknown"
        year = education_year[index] if index < len(education_year) else ""
        profile.education.append(EducationItem(institution=institution, degree=degree, year=year))

    if not profile.name:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        profile.name = first_line[:80]
    return profile


def load_dataturks_resume_records(
    dataset_file: str | Path,
    augment: bool = False,
    num_augments: int = 2,
) -> list[dict]:
    path = Path(dataset_file)
    raw = _read_text_maybe_zipped(path, preferred_encodings=["utf-8"])
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            raise ValueError("Resume dataset JSON must be a list or object.")
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in raw.splitlines() if line.strip()]

    records: list[dict] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        text = _extract_text_from_resume_item(item)
        annotations = item.get("annotation") or item.get("annotations") or []
        if not text or not isinstance(annotations, list):
            continue
        profile = _profile_from_resume_annotations(text, annotations)
        prompt = build_resume_extract_prompt(text)
        records.append(
            build_chat_record(
                task="resume_extract",
                user_prompt=prompt,
                assistant_content=json.dumps(profile.model_dump(), ensure_ascii=False, indent=2),
                meta={"dataset": "resume_entities_for_ner", "row_index": index},
            )
        )
        if augment:
            for aug_index, variant_text in enumerate(augment_resume_text(text, seed=index + 13, num_variants=num_augments)):
                records.append(
                    build_chat_record(
                        task="resume_extract",
                        user_prompt=build_resume_extract_prompt(variant_text),
                        assistant_content=json.dumps(profile.model_dump(), ensure_ascii=False, indent=2),
                        meta={
                            "dataset": "resume_entities_for_ner",
                            "row_index": index,
                            "augmented": True,
                            "aug_index": aug_index,
                        },
                    )
                )
    return records


def _read_table_rows(dataset_file: str | Path) -> list[dict[str, str]]:
    path = Path(dataset_file)
    suffix = path.suffix.lower()
    if suffix == ".zip":
        inner_name, raw = _read_first_file_from_zip(path)
        inner_suffix = Path(inner_name).suffix.lower()
        if inner_suffix == ".csv":
            text = _decode_with_fallback(raw, ["utf-8-sig", "cp1252", "latin-1"])
            return list(csv.DictReader(text.splitlines()))
        if inner_suffix == ".json":
            payload = json.loads(_decode_with_fallback(raw, ["utf-8", "utf-8-sig", "latin-1"]))
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        if inner_suffix == ".jsonl":
            text = _decode_with_fallback(raw, ["utf-8", "utf-8-sig", "latin-1"])
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        raise ValueError(f"Unsupported question dataset format inside zip: {inner_name}")
    if suffix == ".csv":
        text = _decode_with_fallback(path.read_bytes(), ["utf-8-sig", "cp1252", "latin-1"])
        return list(csv.DictReader(text.splitlines()))
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported question dataset format: {path}")


def _decode_with_fallback(raw: bytes, encodings: list[str]) -> str:
    last_error = None
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No encodings provided.")


def _read_first_file_from_zip(path: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise FileNotFoundError(f"No files found inside zip: {path}")
        name = names[0]
        return name, archive.read(name)


def _read_text_maybe_zipped(path: Path, preferred_encodings: list[str]) -> str:
    if path.suffix.lower() == ".zip":
        _, raw = _read_first_file_from_zip(path)
        return _decode_with_fallback(raw, preferred_encodings)
    return _decode_with_fallback(path.read_bytes(), preferred_encodings)


def _pick_value(row: dict[str, Any], candidates: list[str]) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _keywords_from_answer(text: str, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9\-\+\.#]{2,}", text.lower())
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "using",
        "used",
        "your",
        "have",
        "will",
        "what",
        "when",
        "where",
        "which",
        "about",
        "their",
        "there",
        "they",
    }
    unique: list[str] = []
    for token in tokens:
        if token in stopwords or token in unique:
            continue
        unique.append(token)
        if len(unique) >= limit:
            break
    return unique


def load_interview_question_records(
    dataset_file: str | Path,
    augment: bool = False,
    num_augments: int = 2,
) -> list[dict]:
    rows = _read_table_rows(dataset_file)
    records: list[dict] = []

    for index, row in enumerate(rows):
        question = _pick_value(row, QUESTION_COLUMN_CANDIDATES["question"])
        brief_answer = _pick_value(row, QUESTION_COLUMN_CANDIDATES["brief_answer"])
        category = _pick_value(row, QUESTION_COLUMN_CANDIDATES["category"]) or "software engineering"
        difficulty = (_pick_value(row, QUESTION_COLUMN_CANDIDATES["difficulty"]) or "hard").lower()
        difficulty = difficulty if difficulty in {"easy", "medium", "hard"} else "hard"
        question_number = _pick_value(row, QUESTION_COLUMN_CANDIDATES["question_number"]) or str(index + 1)
        if not question:
            continue

        synthetic_resume = {
            "candidate_profile": {
                "skills": [category],
                "experience": [{"role": "Software Engineer", "highlights": [brief_answer[:160] or category]}],
            }
        }
        synthetic_jd = (
            f"Role: Software Engineer\n"
            f"Target category: {category}\n"
            f"Difficulty: {difficulty}\n"
            f"Generate one interview question tailored to this topic."
        )
        interview_question = InterviewQuestion(
            question_id=f"q{question_number}",
            type="technical",
            skill_target=category,
            difficulty=difficulty,
            question=question,
            expected_keywords=_keywords_from_answer(brief_answer or category),
        )
        user_prompt = (
            "Generate 1 interview question in JSON. "
            "Each item must include question_id, type, skill_target, difficulty, question, expected_keywords.\n\n"
            f"Resume JSON:\n{json.dumps(synthetic_resume, ensure_ascii=False, indent=2)}\n\n"
            f"Job Description:\n{synthetic_jd}"
        )
        records.append(
            build_chat_record(
                task="question_generation",
                user_prompt=user_prompt,
                assistant_content=json.dumps([interview_question.model_dump()], ensure_ascii=False, indent=2),
                meta={"dataset": "software_engineering_interview_questions", "row_index": index, "category": category},
            )
        )
        if augment:
            prompt_variants = build_question_prompt_variants(
                resume_json=synthetic_resume,
                jd_text=synthetic_jd,
                seed=index + 7,
                num_variants=num_augments,
            )
            for aug_index, prompt_variant in enumerate(prompt_variants):
                records.append(
                    build_chat_record(
                        task="question_generation",
                        user_prompt=prompt_variant,
                        assistant_content=json.dumps([interview_question.model_dump()], ensure_ascii=False, indent=2),
                        meta={
                            "dataset": "software_engineering_interview_questions",
                            "row_index": index,
                            "category": category,
                            "augmented": True,
                            "aug_index": aug_index,
                        },
                    )
                )
    return records
