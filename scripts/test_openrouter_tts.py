#!/usr/bin/env python3
"""One-command Vietnamese TTS smoke test via OpenRouter.

Usage:
    PYTHONPATH=src python3 scripts/test_openrouter_tts.py
    PYTHONPATH=src python3 scripts/test_openrouter_tts.py --text "Xin chào"
    PYTHONPATH=src python3 scripts/test_openrouter_tts.py --model microsoft/mai-voice-2.1 --voice vi-VN-Harper:MAI-Voice-2.1
    PYTHONPATH=src python3 scripts/test_openrouter_tts.py --output /tmp/test.mp3 --response-format mp3
"""
from __future__ import annotations

import argparse
import os
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
from interview_ai.providers import OpenRouterTTSProvider

DEFAULT_TEXT = (
    "Chào bạn, hôm nay chúng ta sẽ bắt đầu buổi phỏng vấn vị trí Backend Engineer. "
    "Trong CV bạn có đề cập đến PostgreSQL và Redis. Bạn có thể giải thích bottleneck "
    "lớn nhất của hệ thống đó là gì không?"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenRouter Vietnamese TTS smoke test")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--model", help="Override model from config")
    parser.add_argument("--voice", help="Override voice from config")
    parser.add_argument("--output", help="Output MP3 path (default: artifacts/tts-smoke/smoke.mp3)")
    parser.add_argument("--response-format", dest="response_format", default=None)
    parser.add_argument("--speed", type=float, default=None, help="Speech speed (only when supported)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "1. Open .env in the repo root.\n"
            "2. Set OPENROUTER_API_KEY=sk-or-...\n"
            "3. Re-run this script."
        )

    cfg = load_config(ROOT_DIR / "configs" / "models.yaml")
    tts_cfg = cfg.speech.tts

    model = args.model or tts_cfg.model or "microsoft/mai-voice-2.1-flash"
    extra = tts_cfg.extra or {}
    voice = args.voice or extra.get("voice", "vi-VN-Harper:MAI-Voice-2.1-Flash")
    response_format = args.response_format or extra.get("response_format", "mp3")
    timeout = float(extra.get("timeout", 60))

    output_path = args.output or str(ROOT_DIR / "artifacts" / "tts-smoke" / "smoke.mp3")

    provider = OpenRouterTTSProvider(
        model=model,
        voice=voice,
        response_format=response_format,
        api_key=api_key,
        timeout=timeout,
        speed=args.speed,
    )

    print(f"Model  : {model}")
    print(f"Voice  : {voice}")
    print(f"Format : {response_format}")
    print(f"Chars  : {len(args.text)}")
    print(f"Output : {output_path}")
    print("Synthesizing...", flush=True)

    try:
        result = provider.synthesize_with_meta(args.text, output_path)
    except Exception as exc:
        sys.exit(f"Error: {exc}")

    print()
    print(f"✓ Output path     : {result.output_path}")
    print(f"  Byte count      : {result.byte_count:,}")
    print(f"  Content-Type    : {result.content_type}")
    print(f"  Total time      : {result.total_elapsed:.2f}s")
    if result.time_to_first_byte is not None:
        print(f"  Time-to-1st-byte: {result.time_to_first_byte:.2f}s")
    if result.generation_id:
        print(f"  Generation ID   : {result.generation_id}")

    print()
    print(f"Play with:  open \"{result.output_path}\"   # macOS")
    print(f"            mpv \"{result.output_path}\"    # Linux")


if __name__ == "__main__":
    main()
