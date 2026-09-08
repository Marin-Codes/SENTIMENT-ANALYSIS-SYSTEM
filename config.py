"""Minimal NVIDIA NIM config used by main.py."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True)
class NvidiaModelConfig:
    name: str
    env_var: str
    model_id: str
    description: str = ""


NVIDIA_MODELS: tuple[NvidiaModelConfig, ...] = (
    NvidiaModelConfig(
        name="nemotron_3_nano_30b",
        env_var="NVIDIA_API_KEY_NEMOTRON_3_NANO_30B_A3B",
        model_id="nvidia/nemotron-3-nano-30b-a3b",
        description="Efficient MoE LLM for chat and reasoning",
    ),
)


def _load_env() -> None:
    project_root = Path(__file__).resolve().parent
    for candidate in (project_root / ".env", project_root / "venv" / ".env"):
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
    load_dotenv(override=False)


_load_env()


def get_api_key(model_name: str) -> str | None:
    for model in NVIDIA_MODELS:
        if model.name == model_name:
            return os.getenv(model.env_var) or None
    return None


def get_model_config(model_name: str) -> NvidiaModelConfig | None:
    for model in NVIDIA_MODELS:
        if model.name == model_name:
            return model
    return None


def configured_models() -> list[NvidiaModelConfig]:
    return [model for model in NVIDIA_MODELS if os.getenv(model.env_var)]