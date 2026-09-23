from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.openrouter_client import DEFAULT_CHAT_MODEL, OpenRouterClient
from interview_ai.pipeline import InterviewPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run demo inference for the interview AI pipeline.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="OpenRouter model id for the LLM (default: %(default)s)")
    parser.add_argument("--provider", action="append", help="Pin OpenRouter provider order, e.g. --provider groq (repeatable)")
    parser.add_argument("--resume-file", required=True)
    parser.add_argument("--jd-file", required=True)
    args = parser.parse_args()

    client = OpenRouterClient(chat_model=args.chat_model, provider_order=args.provider)
    pipeline = InterviewPipeline(client=client)

    resume_result = pipeline.extract_resume(args.resume_file)
    optimization_result = pipeline.optimize_resume(args.resume_file, args.jd_file)
    question_result = pipeline.generate_questions(args.resume_file, args.jd_file)

    print("=== Resume Extraction ===")
    print(json.dumps(resume_result, ensure_ascii=False, indent=2))

    print("\n=== Resume Optimization ===")
    print(json.dumps(optimization_result, ensure_ascii=False, indent=2))

    print("\n=== Question Generation ===")
    print(json.dumps(question_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
