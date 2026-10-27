"""Provider abstractions for LLM, STT, and TTS.

Business logic must depend only on these ABCs, never directly on a concrete
HTTP client. Swap implementations by passing a different provider instance.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from interview_ai.schemas import STTResult

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

logger = logging.getLogger(__name__)


# ── Abstract interfaces ───────────────────────────────────────────────────────

class LLMProvider(ABC):
    """Generate a chat completion given a list of messages."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str: ...

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class STTProvider(ABC):
    """Transcribe audio to text."""

    @abstractmethod
    def transcribe(self, audio_path: str | Path) -> STTResult: ...


class TTSProvider(ABC):
    """Synthesize text to spoken audio."""

    @abstractmethod
    def synthesize(self, text: str, output_file: str) -> str:
        """Write audio to *output_file* and return the path."""
        ...


# ── OpenRouter LLM ───────────────────────────────────────────────────────────

class OpenRouterLLMProvider(LLMProvider):
    """Chat completions via OpenRouter (default: GPT-OSS-20B)."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
        api_key: str | None = None,
        provider_order: list[str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._provider_order = provider_order
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _provider_payload(self) -> dict[str, Any]:
        if not self._provider_order:
            return {}
        return {"provider": {"order": self._provider_order, "allow_fallbacks": True}}

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **self._provider_payload(),
        }
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()


# ── OpenRouter STT ────────────────────────────────────────────────────────────

class OpenRouterSTTProvider(STTProvider):
    """Speech-to-text via OpenRouter (default: Whisper Large V3 Turbo)."""

    def __init__(
        self,
        model: str = "openai/whisper-large-v3-turbo",
        api_key: str | None = None,
        provider_order: list[str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._model = model
        self._provider_order = provider_order
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _provider_payload(self) -> dict[str, Any]:
        if not self._provider_order:
            return {}
        return {"provider": {"order": self._provider_order, "allow_fallbacks": True}}

    def transcribe(self, audio_path: str | Path) -> STTResult:
        path = Path(audio_path)

        # Text files bypass HTTP — used in tests and for pre-transcribed input.
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8").strip()
            return STTResult(
                transcript=text,
                duration=max(len(text.split()) / 2.5, 1.0),
                provider="text_fallback",
            )

        audio_format = path.suffix.lstrip(".").lower() or "wav"
        audio_b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        payload: dict[str, Any] = {
            "model": self._model,
            "input_audio": {"data": audio_b64, "format": audio_format},
            **self._provider_payload(),
        }
        response = requests.post(
            f"{OPENROUTER_BASE_URL}/audio/transcriptions",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return STTResult(
            transcript=data.get("text", "").strip(),
            duration=float(usage.get("seconds", 0.0)) or 60.0,
            provider=f"openrouter/{self._model}",
            raw=data,
        )


# ── OpenRouter TTS ────────────────────────────────────────────────────────────

@dataclass
class TTSSynthesisResult:
    """Non-secret metadata returned by OpenRouterTTSProvider.synthesize_with_meta()."""
    output_path: str
    model: str
    voice: str
    response_format: str
    byte_count: int
    content_type: str
    total_elapsed: float
    time_to_first_byte: float | None = None
    generation_id: str | None = None
    char_count: int = 0


class OpenRouterTTSProvider(TTSProvider):
    """Text-to-speech via OpenRouter /v1/audio/speech.

    Implements the TTSProvider.synthesize() compatibility contract and also
    exposes synthesize_with_meta() for richer benchmark/logging data.
    """

    def __init__(
        self,
        model: str = "microsoft/mai-voice-2.1-flash",
        voice: str = "vi-VN-Harper:MAI-Voice-2.1-Flash",
        response_format: str = "mp3",
        api_key: str | None = None,
        timeout: float = 60.0,
        speed: float | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to .env or export it before running."
            )
        self._api_key = key
        self._model = model
        self._voice = voice
        self._response_format = response_format
        self._timeout = timeout
        self._speed = speed

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def synthesize(self, text: str, output_file: str) -> str:
        """TTSProvider contract: write audio to output_file, return path."""
        result = self.synthesize_with_meta(text, output_file)
        return result.output_path

    def synthesize_with_meta(self, text: str, output_file: str) -> TTSSynthesisResult:
        """Synthesize and return rich metadata for benchmarking."""
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(out.suffix + ".part")

        payload: dict[str, Any] = {
            "model": self._model,
            "input": text,
            "voice": self._voice,
            "response_format": self._response_format,
        }
        if self._speed is not None:
            payload["speed"] = self._speed

        t_start = time.monotonic()
        t_first_byte: float | None = None
        byte_count = 0
        content_type = ""
        generation_id: str | None = None

        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/audio/speech",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout,
                stream=True,
            )

            if not response.ok:
                # Safe error: status + truncated body, never auth header
                try:
                    err_body = response.json()
                    msg = str(err_body.get("error", err_body))[:200]
                except Exception:
                    msg = response.text[:200]
                raise RuntimeError(
                    f"OpenRouter TTS HTTP {response.status_code}: {msg}"
                )

            content_type = response.headers.get("Content-Type", "")
            generation_id = response.headers.get("X-Generation-Id")

            with part.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        if t_first_byte is None:
                            t_first_byte = time.monotonic() - t_start
                        fh.write(chunk)
                        byte_count += len(chunk)

        except Exception:
            if part.exists():
                part.unlink()
            raise

        if byte_count == 0:
            part.unlink()
            raise RuntimeError(
                f"OpenRouter TTS returned an empty audio body for model={self._model!r}"
            )

        part.rename(out)
        total_elapsed = time.monotonic() - t_start
        logger.info(
            "TTS synthesized %d chars → %d bytes in %.2fs (model=%s)",
            len(text), byte_count, total_elapsed, self._model,
        )
        return TTSSynthesisResult(
            output_path=str(out),
            model=self._model,
            voice=self._voice,
            response_format=self._response_format,
            byte_count=byte_count,
            content_type=content_type,
            total_elapsed=total_elapsed,
            time_to_first_byte=t_first_byte,
            generation_id=generation_id,
            char_count=len(text),
        )


# ── Stub TTS (for tests / MVP) ────────────────────────────────────────────────

class StubTTSProvider(TTSProvider):
    """Writes the text to a .txt file instead of producing real audio.

    Suitable for tests and for local development when a real TTS API is not
    configured. Replace with a real provider (e.g. ElevenLabs, Edge-TTS,
    Zalo TTS) for production.
    """

    def synthesize(self, text: str, output_file: str) -> str:
        out = Path(output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return str(out)


# ── Factory helpers ────────────────────────────────────────────────────────────

def make_llm_provider(provider: str, model: str, **kwargs: Any) -> LLMProvider:
    if provider == "openrouter":
        return OpenRouterLLMProvider(model=model, **kwargs)
    raise ValueError(f"Unknown LLM provider: {provider!r}. Supported: 'openrouter'")


def make_stt_provider(provider: str, model: str, **kwargs: Any) -> STTProvider:
    if provider == "openrouter":
        return OpenRouterSTTProvider(model=model, **kwargs)
    raise ValueError(f"Unknown STT provider: {provider!r}. Supported: 'openrouter'")


def make_tts_provider(provider: str, model: str = "", **kwargs: Any) -> TTSProvider:
    if provider in {"stub", ""}:
        return StubTTSProvider()
    if provider == "openrouter":
        # Pull voice/format/speed/timeout from extra kwargs; fail loudly on bad config
        init_kwargs: dict[str, Any] = {}
        if model:
            init_kwargs["model"] = model
        for key in ("voice", "response_format", "timeout", "speed", "api_key"):
            if key in kwargs:
                init_kwargs[key] = kwargs[key]
        return OpenRouterTTSProvider(**init_kwargs)
    raise ValueError(f"Unknown TTS provider: {provider!r}. Supported: 'stub', 'openrouter'")
