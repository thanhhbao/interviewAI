"""TTS module — backwards-compatibility shim.

Production TTS is now handled through TTSProvider in providers.py.
LocalTTSService is kept as an alias for StubTTSProvider so that existing
scripts that import it continue to work without modification.
"""
from __future__ import annotations

from interview_ai.providers import StubTTSProvider

# Alias for existing code that imports LocalTTSService
LocalTTSService = StubTTSProvider

__all__ = ["LocalTTSService", "StubTTSProvider"]
