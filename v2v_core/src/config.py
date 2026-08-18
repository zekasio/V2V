"""
Centralized configuration using pydantic-settings.

All settings are loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    """Return the project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings — values sourced from .env or environment."""

    model_config = SettingsConfigDict(
        env_file=str(_project_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Murf Dub ──────────────────────────────────────────────
    murfdub_api_key: str = Field(default="", description="Murf Dub API key")

    # ── LLM Translation (Gemini) ──────────────────────────────
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-3.1-flash-lite", description="Gemini model name")

    # ── Language ──────────────────────────────────────────────
    source_language: str = Field(default="tr", description="Source language ISO code")
    target_locale: str = Field(default="en_US", description="Target locale for Murf")
    target_language: str = Field(default="English", description="Target language name")

    # ── OCR ───────────────────────────────────────────────────
    ocr_confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    ocr_sample_fps: int = Field(default=2, ge=1, le=30)
    ocr_use_scene_detection: bool = Field(default=True)

    # ── Inpainting ────────────────────────────────────────────
    inpainting_method: Literal["lama", "telea", "ns"] = Field(default="lama")
    inpainting_mask_dilation: int = Field(default=15, ge=0)

    # ── Subtitles ─────────────────────────────────────────────
    subtitle_mode: Literal["external", "burned"] = Field(default="external")
    subtitle_font: str = Field(default="Arial")
    subtitle_font_size: int = Field(default=24, ge=8, le=72)

    # ── Video Output ──────────────────────────────────────────
    output_video_crf: int = Field(default=20, ge=0, le=51)
    output_video_codec: str = Field(default="libx264")
    output_audio_codec: str = Field(default="aac")
    fast_template_mode: bool = Field(default=False, description="Enable ultra-fast static template overlay mode")

    # ── Directories ───────────────────────────────────────────
    input_dir: str = Field(default="input")
    output_dir: str = Field(default="output")
    temp_dir: str = Field(default="temp")
    log_dir: str = Field(default="logs")

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(default="INFO")

    # ── Helpers ───────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return _project_root()

    def resolve_dir(self, name: str) -> Path:
        """Return an absolute path for a directory setting, creating it if needed."""
        raw = getattr(self, name)
        p = Path(raw) if os.path.isabs(raw) else self.project_root / raw
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def input_path(self) -> Path:
        return self.resolve_dir("input_dir")

    @property
    def output_path(self) -> Path:
        return self.resolve_dir("output_dir")

    @property
    def temp_path(self) -> Path:
        return self.resolve_dir("temp_dir")

    @property
    def log_path(self) -> Path:
        return self.resolve_dir("log_dir")


# Singleton instance
settings = Settings()
