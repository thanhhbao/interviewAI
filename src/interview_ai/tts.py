from __future__ import annotations

from pathlib import Path


class LocalTTSService:
    def __init__(self, engine: str = "stub") -> None:
        self.engine = engine

    def synthesize(self, text: str, output_file: str) -> str:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Stub implementation: save the text that should be spoken.
        # Replace this with Piper/Coqui integration when you add real TTS.
        output_path.write_text(text, encoding="utf-8")
        return str(output_path)
