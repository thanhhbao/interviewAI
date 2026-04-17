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


def try_parse_json(text: str):
    try:
        return True, json.loads(text)
    except Exception:
        return False, None


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

    summary = {
        "resume_extract_json_valid": resume_ok,
        "question_generation_json_valid": question_ok,
        "resume_extract_preview": resume_result["model_output"][:500],
        "question_generation_preview": question_result["model_output"][:500],
        "resume_extract_keys": sorted(list(resume_json.keys())) if isinstance(resume_json, dict) else [],
        "question_count": len(question_json) if isinstance(question_json, list) else 0,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_file:
        Path(args.output_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
