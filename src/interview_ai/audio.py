"""Audio analysis — now delegates transcription through STTProvider.

AudioAnalyzer is kept for backwards compatibility with existing scripts and
with the AudioAnalyzer(client=...) constructor pattern.  New code should use
OpenRouterSTTProvider directly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interview_ai.schemas import AudioAnalysis, STTResult
from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words


def _load_duration_from_sidecar(path: Path) -> float:
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return float(payload.get("duration_seconds", 60.0))
    return 60.0


def _audio_analysis_from_result(result: STTResult) -> AudioAnalysis:
    transcript = result.transcript
    duration = result.duration or 60.0
    speech_duration = duration * 0.9
    pause_ratio = max(0.0, min(1.0, 1.0 - (speech_duration / duration if duration else 1.0)))
    return AudioAnalysis(
        transcript=transcript,
        duration_seconds=duration,
        wpm=compute_wpm(transcript, duration),
        filler_count=count_filler_words(transcript),
        fluency_score=compute_fluency_score(transcript, duration),
        pause_ratio=round(pause_ratio, 2),
        provider=result.provider,
    )


class AudioAnalyzer:
    """Backwards-compat wrapper: accepts an OpenRouterClient or STTProvider.

    New code: pass stt_provider= to TurnBasedConversationManager instead.
    """

    def __init__(self, client: Any = None, stt_provider: Any = None, **client_kwargs: Any) -> None:
        # Prefer an explicit STTProvider
        if stt_provider is not None:
            self._stt = stt_provider
            self.client = None
        elif client is not None:
            # Wrap legacy OpenRouterClient
            from interview_ai.providers import OpenRouterSTTProvider
            import os
            self._stt = OpenRouterSTTProvider(
                model=getattr(client, "stt_model", "openai/whisper-large-v3-turbo"),
                api_key=getattr(client, "api_key", None),
            )
            self.client = client
        else:
            from interview_ai.providers import OpenRouterSTTProvider
            self._stt = OpenRouterSTTProvider(**client_kwargs)
            self.client = None

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        result = self._stt.transcribe(audio_path)
        return {
            "text": result.transcript,
            "duration_seconds": result.duration,
            "source": result.provider,
        }

    def analyze(self, audio_path: str) -> AudioAnalysis:
        result = self._stt.transcribe(audio_path)
        return _audio_analysis_from_result(result)
