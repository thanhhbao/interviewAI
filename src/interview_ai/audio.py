from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from interview_ai.io import read_text_file
from interview_ai.schemas import AudioAnalysis
from interview_ai.scoring import compute_fluency_score, compute_wpm, count_filler_words


def _load_duration_from_sidecar(path: Path) -> float:
    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return float(payload.get("duration_seconds", 60.0))
    return 60.0


class AudioAnalyzer:
    def __init__(self, whisper_model_name: str = "base") -> None:
        self.whisper_model_name = whisper_model_name
        self._whisper_model = None

    def _get_whisper_model(self):
        if self._whisper_model is None:
            import whisper

            self._whisper_model = whisper.load_model(self.whisper_model_name)
        return self._whisper_model

    def transcribe(self, audio_path: str) -> dict[str, Any]:
        path = Path(audio_path)
        if path.suffix.lower() in {".txt", ".md"}:
            transcript = read_text_file(path)
            return {
                "text": transcript,
                "duration_seconds": _load_duration_from_sidecar(path),
                "source": "text_fallback",
            }

        model = self._get_whisper_model()
        result = model.transcribe(str(path))
        duration_seconds = float(result.get("segments", [{}])[-1].get("end", 60.0)) if result.get("segments") else 60.0
        return {
            "text": result.get("text", "").strip(),
            "duration_seconds": duration_seconds,
            "segments": result.get("segments", []),
            "source": "whisper",
        }

    def analyze(self, audio_path: str) -> AudioAnalysis:
        transcription = self.transcribe(audio_path)
        transcript = transcription["text"]
        duration_seconds = float(transcription.get("duration_seconds", 60.0))
        segments = transcription.get("segments", [])

        speech_duration = 0.0
        if segments:
            speech_duration = sum(float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)) for segment in segments)
        else:
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
