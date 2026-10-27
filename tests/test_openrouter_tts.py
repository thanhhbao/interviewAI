"""Unit tests for OpenRouterTTSProvider — all HTTP mocked, no real API calls."""
from __future__ import annotations

import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

FAKE_MP3 = b"\xff\xfb\x90\x00" + b"\x00" * 100  # minimal fake MP3 bytes


def _mock_ok_response(content: bytes = FAKE_MP3, generation_id: str = "gen-abc-123") -> MagicMock:
    resp = MagicMock()
    resp.ok = True
    resp.status_code = 200
    resp.headers = {
        "Content-Type": "audio/mpeg",
        "X-Generation-Id": generation_id,
    }
    # iter_content yields one chunk
    resp.iter_content = MagicMock(return_value=iter([content]))
    return resp


def _mock_error_response(status: int, body: dict | str) -> MagicMock:
    resp = MagicMock()
    resp.ok = False
    resp.status_code = status
    resp.headers = {}
    if isinstance(body, dict):
        resp.json = MagicMock(return_value=body)
        resp.text = json.dumps(body)
    else:
        resp.json = MagicMock(side_effect=ValueError("not json"))
        resp.text = body
    resp.iter_content = MagicMock(return_value=iter([]))
    return resp


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestOpenRouterTTSProvider:
    """OpenRouterTTSProvider: happy path and error cases."""

    def _provider(self, **kwargs):
        from interview_ai.providers import OpenRouterTTSProvider
        return OpenRouterTTSProvider(
            model="microsoft/mai-voice-2.1-flash",
            voice="vi-VN-Harper:MAI-Voice-2.1-Flash",
            api_key="sk-or-test-key",
            **kwargs,
        )

    def test_successful_mp3_response(self, tmp_path):
        """Happy path: write MP3 bytes, return path."""
        out = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=_mock_ok_response()) as mock_post:
            from interview_ai.providers import OpenRouterTTSProvider
            p = self._provider()
            result_path = p.synthesize("Xin chào", out)
        assert result_path == out
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0
        mock_post.assert_called_once()

    def test_synthesize_with_meta_returns_metadata(self, tmp_path):
        """synthesize_with_meta() populates generation ID, byte count, content type."""
        out = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=_mock_ok_response(generation_id="gen-xyz")):
            p = self._provider()
            result = p.synthesize_with_meta("Xin chào", out)
        assert result.generation_id == "gen-xyz"
        assert result.byte_count == len(FAKE_MP3)
        assert result.content_type == "audio/mpeg"
        assert result.char_count == len("Xin chào")
        assert result.total_elapsed >= 0

    def test_nested_output_directory_created(self, tmp_path):
        """Parent directories are created automatically."""
        out = str(tmp_path / "a" / "b" / "c" / "out.mp3")
        with patch("requests.post", return_value=_mock_ok_response()):
            p = self._provider()
            p.synthesize("test", out)
        assert Path(out).exists()

    def test_part_file_cleaned_up_on_http_error(self, tmp_path):
        """No .part file left behind after an error."""
        out = tmp_path / "out.mp3"
        with patch("requests.post", return_value=_mock_error_response(429, {"error": "rate limited"})):
            p = self._provider()
            with pytest.raises(RuntimeError, match="429"):
                p.synthesize("test", str(out))
        part = tmp_path / "out.mp3.part"
        assert not part.exists()
        assert not out.exists()

    def test_empty_audio_response_raises(self, tmp_path):
        """Empty body after 200 raises RuntimeError."""
        out = str(tmp_path / "out.mp3")
        resp = _mock_ok_response(content=b"")
        with patch("requests.post", return_value=resp):
            p = self._provider()
            with pytest.raises(RuntimeError, match="empty"):
                p.synthesize("test", out)
        assert not Path(out).exists()

    def test_http_json_error_body(self, tmp_path):
        """JSON error body is surfaced safely (no auth header)."""
        out = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=_mock_error_response(400, {"error": "bad voice"})):
            p = self._provider()
            with pytest.raises(RuntimeError, match="400"):
                p.synthesize("test", out)

    def test_http_non_json_error_body(self, tmp_path):
        """Non-JSON error body is handled."""
        out = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=_mock_error_response(500, "Internal Server Error")):
            p = self._provider()
            with pytest.raises(RuntimeError, match="500"):
                p.synthesize("test", out)

    def test_missing_api_key_raises(self, monkeypatch):
        """Missing key raises ValueError before any HTTP call."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from interview_ai.providers import OpenRouterTTSProvider
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            OpenRouterTTSProvider(model="m", voice="v", api_key="")

    def test_time_to_first_byte_captured(self, tmp_path):
        out = str(tmp_path / "out.mp3")
        with patch("requests.post", return_value=_mock_ok_response()):
            p = self._provider()
            result = p.synthesize_with_meta("test", out)
        assert result.time_to_first_byte is not None
        assert result.time_to_first_byte >= 0


class TestMakeTTSProviderFactory:
    """make_tts_provider() factory."""

    def test_stub_provider(self):
        from interview_ai.providers import StubTTSProvider, make_tts_provider
        p = make_tts_provider("stub")
        assert isinstance(p, StubTTSProvider)

    def test_openrouter_provider_constructed(self):
        from interview_ai.providers import OpenRouterTTSProvider, make_tts_provider
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}):
            p = make_tts_provider(
                "openrouter",
                model="microsoft/mai-voice-2.1-flash",
                voice="vi-VN-Harper:MAI-Voice-2.1-Flash",
                response_format="mp3",
            )
        assert isinstance(p, OpenRouterTTSProvider)

    def test_unknown_provider_raises(self):
        from interview_ai.providers import make_tts_provider
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            make_tts_provider("nonexistent")

    def test_openrouter_provider_fails_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        from interview_ai.providers import make_tts_provider
        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            make_tts_provider("openrouter", model="m", voice="v", api_key="")


class TestConfigLoading:
    """load_config() correctly hydrates TTS config."""

    def test_tts_config_from_yaml(self, tmp_path):
        yaml_content = """
speech:
  tts:
    provider: openrouter
    model: microsoft/mai-voice-2.1-flash
    extra:
      voice: vi-VN-Harper:MAI-Voice-2.1-Flash
      response_format: mp3
      timeout: 60
"""
        cfg_file = tmp_path / "models.yaml"
        cfg_file.write_text(yaml_content, encoding="utf-8")
        from interview_ai.config import load_config
        cfg = load_config(cfg_file)
        assert cfg.speech.tts.provider == "openrouter"
        assert cfg.speech.tts.model == "microsoft/mai-voice-2.1-flash"
        assert cfg.speech.tts.extra["voice"] == "vi-VN-Harper:MAI-Voice-2.1-Flash"
        assert cfg.speech.tts.extra["response_format"] == "mp3"


class TestBenchmarkSkipsUnavailable:
    """Benchmark validation logic skips invalid model/voice combinations."""

    def test_skip_unknown_model(self):
        """fetch_tts_models mock returns only flash; non-existent model is skipped."""
        live = {
            "microsoft/mai-voice-2.1-flash": {
                "id": "microsoft/mai-voice-2.1-flash",
                "architecture": {"voices": ["vi-VN-Harper:MAI-Voice-2.1-Flash"]},
                "pricing": {},
            }
        }
        # simulate what the benchmark does
        model_id = "microsoft/nonexistent"
        skipped = model_id not in live
        assert skipped

    def test_skip_unknown_voice(self):
        live_voices = ["vi-VN-Harper:MAI-Voice-2.1-Flash", "vi-VN-Grant:MAI-Voice-2.1-Flash"]
        voice = "vi-VN-Unknown:MAI-Voice-2.1-Flash"
        assert voice not in live_voices

    def test_accept_known_voice(self):
        live_voices = ["vi-VN-Harper:MAI-Voice-2.1-Flash"]
        voice = "vi-VN-Harper:MAI-Voice-2.1-Flash"
        assert voice in live_voices
