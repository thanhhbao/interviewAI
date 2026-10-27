"""Runtime configuration loaded from configs/models.yaml.

All three text tasks (planning, realtime, evaluation) can point to the same
model during MVP — they are separate only to allow future optimisation without
changing the business logic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

try:
    import yaml  # pyyaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

_DEFAULT_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "models.yaml"


class ModelConfig(BaseModel):
    provider: str = "openrouter"
    model: str = "openai/gpt-oss-20b"
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    planning: ModelConfig = Field(default_factory=ModelConfig)
    realtime: ModelConfig = Field(default_factory=ModelConfig)
    evaluation: ModelConfig = Field(default_factory=ModelConfig)


class STTConfig(BaseModel):
    provider: str = "openrouter"
    model: str = "openai/whisper-large-v3-turbo"
    extra: dict[str, Any] = Field(default_factory=dict)


class TTSConfig(BaseModel):
    provider: str = "stub"
    model: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class SpeechConfig(BaseModel):
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)


class AppConfig(BaseModel):
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load AppConfig from a YAML file.

    Falls back to sensible defaults when:
    - no path is given and the default file does not exist;
    - pyyaml is not installed;
    - the file is empty.
    """
    target = Path(path) if path else _DEFAULT_CONFIG_PATH

    if not target.exists() or not _YAML_AVAILABLE:
        return AppConfig()

    with open(target, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    return AppConfig.model_validate(data)
