from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import requests

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_STT_MODEL = "openai/whisper-large-v3-turbo"


class OpenRouterClient:
    """Thin client over OpenRouter's chat completions and audio transcription endpoints.

    Routes chat (default: GPT-OSS-20B) and speech-to-text (default: Whisper Large V3
    Turbo) through OpenRouter. Pass provider_order=["groq"] to pin inference to Groq's
    LPU backend for lower latency instead of OpenRouter's cheapest-provider routing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        stt_model: str = DEFAULT_STT_MODEL,
        provider_order: list[str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        self.chat_model = chat_model
        self.stt_model = stt_model
        self.provider_order = provider_order
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _provider_payload(self) -> dict[str, Any]:
        if not self.provider_order:
            return {}
        return {"provider": {"order": self.provider_order, "allow_fallbacks": True}}

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self._provider_payload(),
        }
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        path = Path(audio_path)
        audio_format = path.suffix.lstrip(".").lower() or "wav"
        audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        payload: dict[str, Any] = {
            "model": self.stt_model,
            "input_audio": {"data": audio_b64, "format": audio_format},
            **self._provider_payload(),
        }
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/audio/transcriptions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return {
            "text": data.get("text", "").strip(),
            "duration_seconds": float(usage.get("seconds", 0.0)) or 60.0,
            "raw": data,
        }
