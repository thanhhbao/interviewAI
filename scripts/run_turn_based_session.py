from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    pass

from interview_ai.config import load_config
from interview_ai.conversation import TurnBasedConversationManager
from interview_ai.pipeline import InterviewPipeline
from interview_ai.providers import (
    OpenRouterLLMProvider,
    make_tts_provider,
)

# Legacy import kept for backwards compat with existing scripts that import
# DEFAULT_CHAT_MODEL / DEFAULT_STT_MODEL from here.
try:
    from interview_ai.openrouter_client import (
        DEFAULT_CHAT_MODEL,
        DEFAULT_STT_MODEL,
        OpenRouterClient,
    )
except ImportError:
    DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
    DEFAULT_STT_MODEL = "openai/whisper-large-v3-turbo"
    OpenRouterClient = None  # type: ignore[assignment]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a turn-based interview session with push-to-talk style audio files."
    )
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL)
    parser.add_argument("--provider", action="append", help="Pin OpenRouter provider order (repeatable)")
    parser.add_argument("--resume-file", required=True)
    parser.add_argument("--jd-file", required=True)
    parser.add_argument("--answer-audio", required=True, nargs="+")
    parser.add_argument("--session-output", required=True)
    parser.add_argument("--tts-output-dir", required=True)
    parser.add_argument(
        "--tts-provider",
        default=None,
        help="Override TTS provider (stub | openrouter). Defaults to value in configs/models.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config(ROOT_DIR / "configs" / "models.yaml")

    # Build LLM pipeline — prefer new provider pattern, fall back to legacy client
    llm_provider = OpenRouterLLMProvider(
        model=args.chat_model,
        provider_order=args.provider,
    )
    pipeline = InterviewPipeline(provider=llm_provider)

    # Build TTS provider from config (or CLI override)
    tts_cfg = cfg.speech.tts
    tts_provider_name = args.tts_provider or tts_cfg.provider
    extra = dict(tts_cfg.extra or {})
    tts_provider = make_tts_provider(
        provider=tts_provider_name,
        model=tts_cfg.model,
        **extra,
    )

    manager = TurnBasedConversationManager(
        llm_pipeline=pipeline,
        tts_provider=tts_provider,
    )
    state = manager.start_session(args.resume_file, args.jd_file)

    for turn_index, audio_file in enumerate(args.answer_audio, start=1):
        ext = ".mp3" if tts_provider_name == "openrouter" else ".txt"
        tts_file = Path(args.tts_output_dir) / f"turn_{turn_index:02d}_question{ext}"
        manager.speak_current_question(state, str(tts_file))
        state = manager.process_answer_turn(state, audio_file)
        print(json.dumps(manager.summarize_state(state), ensure_ascii=False, indent=2))
        if not state.current_question:
            break

    output_file = manager.save_state(state, args.session_output)
    print(f"Saved conversation state to {output_file}")


if __name__ == "__main__":
    main()
