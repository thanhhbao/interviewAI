#!/usr/bin/env python3
"""List TTS models available on OpenRouter.

Queries GET /api/v1/models?output_modalities=speech and prints a table.

Usage:
    PYTHONPATH=src python3 scripts/list_openrouter_tts_models.py
    PYTHONPATH=src python3 scripts/list_openrouter_tts_models.py --vietnamese-only
    PYTHONPATH=src python3 scripts/list_openrouter_tts_models.py --json
    PYTHONPATH=src python3 scripts/list_openrouter_tts_models.py --model microsoft/mai-voice-2.1-flash
"""
from __future__ import annotations

import argparse
import json
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

import requests


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def fetch_tts_models(api_key: str) -> list[dict]:
    resp = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        params={"output_modalities": "speech"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def extract_voices(model: dict) -> list[str]:
    """Try common locations where voice lists appear in the schema."""
    # architecture.voices or output_parameters.voice.enum
    arch = model.get("architecture", {})
    voices = arch.get("voices", [])
    if voices:
        return voices

    output_params = model.get("output_parameters", {})
    voice_param = output_params.get("voice", {})
    enum = voice_param.get("enum", [])
    if enum:
        return enum

    # Some models list voices at top level
    return model.get("voices", [])


def has_vietnamese_voice(voices: list[str]) -> bool:
    return any("vi-VN" in v or "vi_VN" in v for v in voices)


def format_pricing(model: dict) -> str:
    pricing = model.get("pricing", {})
    tts_cost = pricing.get("completion", pricing.get("input", "?"))
    return f"${tts_cost}/1K chars" if tts_cost != "?" else "?"


def print_table(models: list[dict]) -> None:
    col_id = max(len(m.get("id", "")) for m in models) if models else 20
    col_id = max(col_id, 10)
    col_name = 30
    print(
        f"{'Model ID':<{col_id}}  {'Name':<{col_name}}  {'Pricing':<16}  {'vi-VN?':<7}  Voices"
    )
    print("-" * (col_id + col_name + 60))
    for m in models:
        mid = m.get("id", "")
        name = (m.get("name", "") or "")[:col_name]
        pricing = format_pricing(m)
        voices = extract_voices(m)
        vi = "✓" if has_vietnamese_voice(voices) else ""
        voices_str = ", ".join(voices[:4])
        if len(voices) > 4:
            voices_str += f" (+{len(voices) - 4})"
        print(f"{mid:<{col_id}}  {name:<{col_name}}  {pricing:<16}  {vi:<7}  {voices_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="List OpenRouter TTS models")
    parser.add_argument("--vietnamese-only", action="store_true", help="Show only models with vi-VN voices")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output raw JSON")
    parser.add_argument("--model", help="Show detailed info for one model ID")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "Add it to .env or: export OPENROUTER_API_KEY=sk-or-..."
        )

    print("Fetching TTS models from OpenRouter...", flush=True)
    try:
        models = fetch_tts_models(api_key)
    except requests.HTTPError as exc:
        sys.exit(f"HTTP error: {exc}")

    if args.model:
        found = [m for m in models if m.get("id") == args.model]
        if not found:
            sys.exit(f"Model {args.model!r} not found in TTS model list.")
        print(json.dumps(found[0], indent=2, ensure_ascii=False))
        return

    if args.vietnamese_only:
        models = [m for m in models if has_vietnamese_voice(extract_voices(m))]

    if args.as_json:
        print(json.dumps(models, indent=2, ensure_ascii=False))
        return

    if not models:
        print("No models found.")
        return

    print(f"\nFound {len(models)} TTS model(s):\n")
    print_table(models)


if __name__ == "__main__":
    main()
