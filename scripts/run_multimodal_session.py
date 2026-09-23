from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interview_ai.openrouter_client import DEFAULT_CHAT_MODEL, DEFAULT_STT_MODEL, OpenRouterClient
from interview_ai.session import InterviewSessionRunner
from interview_ai.pipeline import InterviewPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full multimodal interview session.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="OpenRouter model id for the LLM (default: %(default)s)")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="OpenRouter model id for transcription (default: %(default)s)")
    parser.add_argument("--provider", action="append", help="Pin OpenRouter provider order, e.g. --provider groq (repeatable)")
    parser.add_argument("--resume-file", required=True)
    parser.add_argument("--jd-file", required=True)
    parser.add_argument("--answer-audio", required=True)
    parser.add_argument("--vision-file", required=True)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()

    client = OpenRouterClient(chat_model=args.chat_model, stt_model=args.stt_model, provider_order=args.provider)
    llm_pipeline = InterviewPipeline(client=client)
    runner = InterviewSessionRunner(llm_pipeline=llm_pipeline)
    outputs = runner.run(
        resume_file=args.resume_file,
        jd_file=args.jd_file,
        answer_audio_file=args.answer_audio,
        vision_file=args.vision_file,
        report_dir=args.report_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
