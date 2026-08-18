"""
Subtitle Handler — external SRT management and burn-in support.
"""
from __future__ import annotations
from pathlib import Path
import pysrt
from src.config import settings
from src.logger import get_logger

logger = get_logger("subtitle")


class SubtitleHandler:
    """Manages English subtitle files (SRT) — external and burned modes."""

    def __init__(self) -> None:
        self._mode = settings.subtitle_mode
        self._font = settings.subtitle_font
        self._font_size = settings.subtitle_font_size

    def load_srt(self, srt_path: Path) -> pysrt.SubRipFile:
        """Load and parse an SRT file."""
        if not srt_path.exists():
            raise FileNotFoundError(f"SRT file not found: {srt_path}")
        subs = pysrt.open(str(srt_path), encoding="utf-8")
        logger.info("Loaded %d subtitle entries from %s", len(subs), srt_path.name)
        return subs

    def save_srt(self, subs: pysrt.SubRipFile, output_path: Path) -> Path:
        """Save subtitles to an SRT file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        subs.save(str(output_path), encoding="utf-8")
        logger.info("Saved SRT → %s", output_path)
        return output_path

    def copy_external(self, srt_path: Path, output_dir: Path, video_stem: str) -> Path:
        """Copy SRT as external subtitle file alongside the output video."""
        subs = self.load_srt(srt_path)
        out = output_dir / f"{video_stem}_EN.srt"
        return self.save_srt(subs, out)

    def get_ffmpeg_subtitle_filter(self, srt_path: Path) -> str:
        """Return an FFmpeg subtitle filter string for burn-in."""
        escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        return (
            f"subtitles='{escaped}'"
            f":force_style='FontName={self._font},"
            f"FontSize={self._font_size},"
            f"PrimaryColour=&H00FFFFFF,"
            f"OutlineColour=&H00000000,"
            f"BorderStyle=3,"
            f"Outline=2,"
            f"Shadow=1,"
            f"MarginV=30'"
        )

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("external", "burned"):
            raise ValueError(f"Invalid subtitle mode: {value}")
        self._mode = value
