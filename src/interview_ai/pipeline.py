from __future__ import annotations

import json
from typing import Any

from interview_ai.io import read_any_text
from interview_ai.openrouter_client import OpenRouterClient
from interview_ai.parsers import generate_questions_weak, match_resume_to_jd, parse_resume_weak
from interview_ai.prompts import (
    SYSTEM_PROMPT,
    build_answer_evaluation_prompt,
    build_follow_up_prompt,
    build_question_generation_prompt,
    build_resume_extract_prompt,
    build_resume_optimize_prompt,
)
from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words


class InterviewPipeline:
    """LLM pipeline for the runtime conversation flow, backed by an OpenRouter chat
    model (default: GPT-OSS-20B). Replaces local fine-tuning with a hosted API call —
    see local_pipeline.py for the transformers/peft-based alternative used in training."""

    def __init__(self, client: OpenRouterClient | None = None, **client_kwargs: Any) -> None:
        self.client = client or OpenRouterClient(**client_kwargs)

    def _generate(self, user_prompt: str, max_new_tokens: int = 512) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return self.client.chat(messages, max_tokens=max_new_tokens)

    def extract_resume(self, resume_file: str) -> dict[str, Any]:
        weak_json = parse_resume_weak(resume_file).model_dump()
        prompt = build_resume_extract_prompt(read_any_text(resume_file))
        return {"weak_label": weak_json, "model_output": self._generate(prompt)}

    def optimize_resume(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        weak_json = match_resume_to_jd(resume, jd_text).model_dump()
        prompt = build_resume_optimize_prompt(json.dumps(resume.model_dump(), ensure_ascii=False), jd_text)
        return {"weak_label": weak_json, "model_output": self._generate(prompt)}

    def generate_questions(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        weak_json = [item.model_dump() for item in generate_questions_weak(resume, jd_text)]
        prompt = build_question_generation_prompt(json.dumps(resume.model_dump(), ensure_ascii=False), jd_text)
        return {"weak_label": weak_json, "model_output": self._generate(prompt, max_new_tokens=1024)}

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        rubric: str,
        duration_seconds: float,
    ) -> dict[str, Any]:
        prompt = build_answer_evaluation_prompt(question, answer, rubric)
        metrics = {
            "wpm": compute_wpm(answer, duration_seconds),
            "filler_count": count_filler_words(answer),
            "fluency_score": compute_fluency_score(answer, duration_seconds),
        }
        return {"metrics": metrics, "model_output": self._generate(prompt)}

    def build_full_context(self, resume_file: str, jd_file: str) -> dict[str, Any]:
        resume = parse_resume_weak(resume_file)
        jd_text = read_any_text(jd_file)
        return {
            "resume": resume.model_dump(),
            "jd_text": jd_text,
            "resume_optimization": match_resume_to_jd(resume, jd_text).model_dump(),
            "questions": [item.model_dump() for item in generate_questions_weak(resume, jd_text)],
        }

    def generate_follow_up_question(
        self,
        current_question: str,
        answer: str,
        transcript_history: str,
    ) -> dict[str, Any]:
        prompt = build_follow_up_prompt(current_question, answer, transcript_history)
        return {"model_output": self._generate(prompt, max_new_tokens=512)}
