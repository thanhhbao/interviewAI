#!/usr/bin/env python3
"""Repeatable multi-model/multi-voice Vietnamese TTS benchmark.

Usage:
    PYTHONPATH=src python3 scripts/benchmark_openrouter_tts.py --dry-run
    PYTHONPATH=src python3 scripts/benchmark_openrouter_tts.py --models microsoft/mai-voice-2.1-flash --yes
    PYTHONPATH=src python3 scripts/benchmark_openrouter_tts.py --yes
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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

from interview_ai.providers import OpenRouterTTSProvider

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Default benchmark config ──────────────────────────────────────────────────

DEFAULT_SENTENCES = [
    "Chào bạn, hôm nay chúng ta sẽ bắt đầu buổi phỏng vấn vị trí Backend Engineer.",
    "Trong CV bạn có đề cập đến PostgreSQL và Redis. Bạn có thể giải thích bottleneck lớn nhất của hệ thống đó là gì không?",
    "Nếu lượng request tăng từ một nghìn lên một trăm nghìn mỗi phút, bottleneck đầu tiên bạn dự đoán sẽ nằm ở đâu?",
    "Bạn vừa đề cập tới CI/CD. Pipeline của bạn kiểm soát rollback như thế nào khi deployment thất bại?",
]

# Default matrix: model → [voice, ...]
DEFAULT_MATRIX: dict[str, list[str]] = {
    "microsoft/mai-voice-2.1-flash": [
        "vi-VN-Harper:MAI-Voice-2.1-Flash",
        "vi-VN-Grant:MAI-Voice-2.1-Flash",
    ],
    "microsoft/mai-voice-2.1": [
        "vi-VN-Harper:MAI-Voice-2.1",
        "vi-VN-Grant:MAI-Voice-2.1",
    ],
}


@dataclass
class BenchmarkRecord:
    model: str
    voice: str
    sentence_index: int
    sentence: str
    char_count: int
    output_path: str
    byte_count: int
    content_type: str
    time_to_first_byte: float | None
    total_latency: float
    generation_id: str | None
    success: bool
    error: str = ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_filename(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^\w\-]", "_", s)
    return s[:max_len]


def fetch_tts_models(api_key: str) -> dict[str, Any]:
    """Return {model_id: model_data} for all TTS-capable models."""
    resp = requests.get(
        f"{OPENROUTER_BASE_URL}/models",
        params={"output_modalities": "speech"},
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    return {m["id"]: m for m in data}


def extract_voices(model_data: dict) -> list[str]:
    arch = model_data.get("architecture", {})
    voices = arch.get("voices", [])
    if voices:
        return voices
    output_params = model_data.get("output_parameters", {})
    enum = output_params.get("voice", {}).get("enum", [])
    if enum:
        return enum
    return model_data.get("voices", [])


def estimate_cost(model_data: dict, total_chars: int) -> str | None:
    pricing = model_data.get("pricing", {})
    # TTS pricing is typically per character in the completion/input field
    cost_per_1k = pricing.get("completion") or pricing.get("input")
    if cost_per_1k is None:
        return None
    try:
        cost = float(cost_per_1k) * total_chars / 1000
        return f"${cost:.4f}"
    except (TypeError, ValueError):
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark OpenRouter Vietnamese TTS models")
    parser.add_argument("--models", nargs="+", help="Limit to these model IDs")
    parser.add_argument("--yes", action="store_true", help="Execute paid requests without prompting")
    parser.add_argument("--dry-run", action="store_true", help="Plan requests without sending any")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        sys.exit(
            "Error: OPENROUTER_API_KEY is not set.\n"
            "Open .env and set OPENROUTER_API_KEY=sk-or-..."
        )

    # Build candidate matrix
    matrix = DEFAULT_MATRIX
    if args.models:
        matrix = {m: DEFAULT_MATRIX.get(m, []) for m in args.models}
        # For unknown models include a default vi-VN voice
        for m in args.models:
            if not matrix[m]:
                matrix[m] = ["vi-VN-Harper:MAI-Voice-2.1-Flash"]

    # Query live API
    print("Querying OpenRouter models API...", flush=True)
    try:
        live_models = fetch_tts_models(api_key)
    except Exception as exc:
        sys.exit(f"Failed to query models API: {exc}")

    # Validate candidates
    candidates: list[tuple[str, str]] = []  # (model, voice)
    skipped: list[str] = []

    for model_id, voices in matrix.items():
        if model_id not in live_models:
            skipped.append(f"  SKIP {model_id!r} — not found in live TTS models")
            continue
        live_voices = extract_voices(live_models[model_id])
        for voice in voices:
            # If live API has voice list, validate; otherwise accept (API may not expose list)
            if live_voices and voice not in live_voices:
                skipped.append(f"  SKIP {model_id!r} voice={voice!r} — not in live voice list ({live_voices[:3]}...)")
            else:
                candidates.append((model_id, voice))

    if skipped:
        print("\nSkipped candidates:")
        for s in skipped:
            print(s)

    n_requests = len(candidates) * len(DEFAULT_SENTENCES)
    print(f"\nPlanned: {len(candidates)} model/voice pair(s) × {len(DEFAULT_SENTENCES)} sentences = {n_requests} requests")

    # Estimate cost
    for model_id in {m for m, _ in candidates}:
        total_chars = sum(len(s) for s in DEFAULT_SENTENCES)
        est = estimate_cost(live_models[model_id], total_chars)
        if est:
            print(f"  Est. cost for {model_id}: {est} (for {total_chars} chars × all voices)")

    if not candidates:
        sys.exit("No valid candidates remain. Check model IDs against --dry-run output.")

    if args.dry_run:
        print("\n[dry-run] No requests sent.")
        for m, v in candidates:
            for i, s in enumerate(DEFAULT_SENTENCES):
                print(f"  Would synthesize: model={m!r} voice={v!r} sent={i+1}")
        return

    if not args.yes:
        sys.exit(
            f"\nThis will make {n_requests} paid API request(s).\n"
            "Re-run with --yes to proceed, or --dry-run to preview."
        )

    # Set up output directory
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else ROOT_DIR / "artifacts" / "tts-benchmarks" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}\n")

    records: list[BenchmarkRecord] = []

    for model_id, voice in candidates:
        for sent_idx, sentence in enumerate(DEFAULT_SENTENCES, start=1):
            model_slug = safe_filename(model_id)
            voice_slug = safe_filename(voice, 30)
            filename = f"{model_slug}__{voice_slug}__s{sent_idx}.mp3"
            output_path = str(out_dir / filename)

            print(f"  [{sent_idx}/{len(DEFAULT_SENTENCES)}] {model_id} / {voice}")

            try:
                provider = OpenRouterTTSProvider(
                    model=model_id,
                    voice=voice,
                    response_format="mp3",
                    api_key=api_key,
                    timeout=60,
                )
                result = provider.synthesize_with_meta(sentence, output_path)
                records.append(BenchmarkRecord(
                    model=model_id,
                    voice=voice,
                    sentence_index=sent_idx,
                    sentence=sentence,
                    char_count=len(sentence),
                    output_path=result.output_path,
                    byte_count=result.byte_count,
                    content_type=result.content_type,
                    time_to_first_byte=result.time_to_first_byte,
                    total_latency=result.total_elapsed,
                    generation_id=result.generation_id,
                    success=True,
                ))
                print(f"    ✓ {result.byte_count:,} bytes in {result.total_elapsed:.2f}s")
            except Exception as exc:
                err_msg = str(exc)[:300]
                print(f"    ✗ {err_msg}")
                records.append(BenchmarkRecord(
                    model=model_id,
                    voice=voice,
                    sentence_index=sent_idx,
                    sentence=sentence,
                    char_count=len(sentence),
                    output_path=output_path,
                    byte_count=0,
                    content_type="",
                    time_to_first_byte=None,
                    total_latency=0.0,
                    generation_id=None,
                    success=False,
                    error=err_msg,
                ))

    # Write manifest.json
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write summary.csv
    summary_path = out_dir / "summary.csv"
    summary_fields = [
        "model", "voice", "sentence_index", "char_count",
        "byte_count", "time_to_first_byte", "total_latency",
        "generation_id", "content_type", "output_path", "success", "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        for r in records:
            row = {k: getattr(r, k, "") for k in summary_fields}
            writer.writerow(row)

    # Write ratings.csv (blank scoring columns)
    ratings_path = out_dir / "ratings.csv"
    rating_fields = [
        "model", "voice", "sentence_index", "output_path",
        "vietnamese_naturalness", "accent_pronunciation",
        "technical_english_pronunciation", "prosody",
        "professional_interviewer_feel", "perceived_latency",
        "overall_score", "notes",
    ]
    with ratings_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rating_fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "model": r.model,
                "voice": r.voice,
                "sentence_index": r.sentence_index,
                "output_path": r.output_path,
                "vietnamese_naturalness": "",
                "accent_pronunciation": "",
                "technical_english_pronunciation": "",
                "prosody": "",
                "professional_interviewer_feel": "",
                "perceived_latency": "",
                "overall_score": "",
                "notes": "",
            })

    n_ok = sum(1 for r in records if r.success)
    n_fail = len(records) - n_ok
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed")
    print(f"  manifest.json  → {manifest_path}")
    print(f"  summary.csv    → {summary_path}")
    print(f"  ratings.csv    → {ratings_path}")


if __name__ == "__main__":
    main()
