from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.pipeline import InterviewPipeline


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def try_parse_json(text: str):
    try:
        cleaned = strip_code_fence(text)
        return True, json.loads(cleaned)
    except Exception:
        return False, None


def question_summary(payload):
    if isinstance(payload, list):
        return {
            "question_output_type": "list",
            "question_count": len(payload),
            "question_items_valid": sum(1 for item in payload if isinstance(item, dict) and "question" in item),
        }
    if isinstance(payload, dict):
        if isinstance(payload.get("questions"), list):
            questions = payload["questions"]
            return {
                "question_output_type": "dict.questions_list",
                "question_count": len(questions),
                "question_items_valid": sum(1 for item in questions if isinstance(item, dict) and "question" in item),
            }
        return {
            "question_output_type": "dict",
            "question_count": 1 if "question" in payload else 0,
            "question_items_valid": 1 if "question" in payload else 0,
        }
    return {
        "question_output_type": type(payload).__name__ if payload is not None else "none",
        "question_count": 0,
        "question_items_valid": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick evaluation script for trained InterviewAI adapters.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-dir")
    parser.add_argument("--resume-file", required=True)
    parser.add_argument("--jd-file", required=True)
    parser.add_argument("--output-file")
    args = parser.parse_args()

    pipeline = InterviewPipeline(model_name=args.model_name, adapter_dir=args.adapter_dir)

    resume_result = pipeline.extract_resume(args.resume_file)
    question_result = pipeline.generate_questions(args.resume_file, args.jd_file)

    resume_ok, resume_json = try_parse_json(resume_result["model_output"])
    question_ok, question_json = try_parse_json(question_result["model_output"])
    question_info = question_summary(question_json)

    summary = {
        "resume_extract_json_valid": resume_ok,
        "question_generation_json_valid": question_ok,
        "resume_extract_preview": strip_code_fence(resume_result["model_output"])[:500],
        "question_generation_preview": strip_code_fence(question_result["model_output"])[:500],
        "resume_extract_keys": sorted(list(resume_json.keys())) if isinstance(resume_json, dict) else [],
        "question_output_type": question_info["question_output_type"],
        "question_count": question_info["question_count"],
        "question_items_valid": question_info["question_items_valid"],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
