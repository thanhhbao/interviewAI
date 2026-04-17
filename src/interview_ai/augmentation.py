from __future__ import annotations

import json
import random
import re
from copy import deepcopy


RESUME_SECTION_VARIANTS = {
    "Skills": ["Skills", "Technical Skills", "Core Skills", "Competencies"],
    "Experience": ["Experience", "Work Experience", "Employment History", "Professional Experience"],
    "Education": ["Education", "Academic Background", "Education History"],
    "Projects": ["Projects", "Selected Projects", "Project Experience"],
}


QUESTION_PROMPT_STYLES = [
    "Generate 1 interview question in JSON. Each item must include question_id, type, skill_target, difficulty, question, expected_keywords.",
    "Create 1 technical interview question in JSON for the candidate profile and job description below.",
    "Produce 1 structured interview question in JSON. Keep the question aligned with the target skill and difficulty.",
]


def _swap_resume_headers(text: str, rng: random.Random) -> str:
    updated = text
    for base_header, variants in RESUME_SECTION_VARIANTS.items():
        replacement = rng.choice(variants)
        updated = re.sub(rf"\b{re.escape(base_header)}\b", replacement, updated, flags=re.IGNORECASE)
    return updated


def _normalize_phone_variants(text: str) -> list[str]:
    variants = []
    phone_match = re.search(r"(\+?\d[\d\s\-]{8,}\d)", text)
    if not phone_match:
        return variants
    raw_phone = phone_match.group(0)
    digits = re.sub(r"\D", "", raw_phone)
    if len(digits) >= 9:
        variants.append(text.replace(raw_phone, " ".join([digits[:3], digits[3:6], digits[6:]])))
        variants.append(text.replace(raw_phone, f"+{digits[:2]}-{digits[2:5]}-{digits[5:8]}-{digits[8:]}"))
    return variants


def _shuffle_resume_blocks(text: str) -> str:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    if len(blocks) < 3:
        return text
    first = blocks[:1]
    remaining = blocks[1:]
    remaining.reverse()
    return "\n\n".join(first + remaining)


def augment_resume_text(text: str, seed: int = 13, num_variants: int = 3) -> list[str]:
    rng = random.Random(seed)
    variants = []

    base = _swap_resume_headers(text, rng)
    if base != text:
        variants.append(base)

    shuffled = _shuffle_resume_blocks(text)
    if shuffled != text:
        variants.append(shuffled)

    variants.extend(_normalize_phone_variants(text))

    condensed = re.sub(r"\n{2,}", "\n", text)
    if condensed != text:
        variants.append(condensed)

    deduped = []
    seen = {text.strip()}
    for variant in variants:
        key = variant.strip()
        if key and key not in seen:
            seen.add(key)
            deduped.append(variant)
        if len(deduped) >= num_variants:
            break
    return deduped


def build_question_prompt_variants(resume_json: dict, jd_text: str, seed: int = 7, num_variants: int = 3) -> list[str]:
    rng = random.Random(seed)
    variants = []
    styles = deepcopy(QUESTION_PROMPT_STYLES)
    rng.shuffle(styles)

    jd_variants = [
        jd_text,
        jd_text + "\nFocus on problem solving and communication trade-offs.",
        jd_text.replace("Generate one interview question tailored to this topic.", "Ask a realistic interview question for this target role."),
    ]
    for index in range(min(num_variants, len(styles))):
        style = styles[index]
        jd_variant = jd_variants[index % len(jd_variants)]
        variants.append(
            f"{style}\n\nResume JSON:\n{json.dumps(resume_json, ensure_ascii=False, indent=2)}\n\nJob Description:\n{jd_variant}"
        )
    return variants
