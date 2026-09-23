from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interview_ai.io import read_text_file
from interview_ai.openrouter_client import OpenRouterClient
from interview_ai.schemas import AudioAnalysis
from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words


def _load_duration_from_sidecar(path: Path) -> float:
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return float(payload.get("duration_seconds", 60.0))
    return 60.0


class AudioAnalyzer:
    """Transcribes push-to-talk answers via OpenRouter (default: Whisper Large V3
    Turbo). .txt/.md files bypass transcription entirely — used by data/sample/ fixtures."""

    def __init__(self, client: OpenRouterClient | None = None, **client_kwargs: Any) -> None:
        self.client = client or OpenRouterClient(**client_kwargs)

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        path = Path(audio_path)
        if path.suffix.lower() in {".txt", ".md"}:
            transcript = read_text_file(path)
            return {
                "text": transcript,
                "duration_seconds": _load_duration_from_sidecar(path),
                "source": "text_fallback",
            }

        result = self.client.transcribe(path)
        return {
            "text": result["text"],
            "duration_seconds": result["duration_seconds"],
            "source": "openrouter",
        }

    def analyze(self, audio_path: str) -> AudioAnalysis:
        transcription = self.transcribe(audio_path)
        transcript = transcription["text"]
        duration_seconds = float(transcription.get("duration_seconds", 60.0))
        # OpenRouter's transcription response doesn't include per-segment timing,
        # so approximate speech time as 90% of clip duration (same fallback the
        # previous local-Whisper path used when segments were unavailable).
        speech_duration = duration_seconds * 0.9
        pause_ratio = max(0.0, min(1.0, 1.0 - (speech_duration / duration_seconds if duration_seconds else 1.0)))

        return AudioAnalysis(
            transcript=transcript,
            duration_seconds=duration_seconds,
            wpm=compute_wpm(transcript, duration_seconds),
            filler_count=count_filler_words(transcript),
            fluency_score=compute_fluency_score(transcript, duration_seconds),
            pause_ratio=round(pause_ratio, 2),
        )
