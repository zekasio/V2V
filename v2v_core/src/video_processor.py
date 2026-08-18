"""
Video Processor — FFmpeg-based final assembly:
- Combine processed frames
- Replace audio with Murf dubbed track
- Burn or attach subtitles
"""
from __future__ import annotations
import subprocess
import shutil
from pathlib import Path
from src.config import settings
from src.logger import get_logger
from src.subtitle import SubtitleHandler

logger = get_logger("video_processor")


class VideoProcessor:
    """Assemble the final localized video using FFmpeg."""

    def __init__(self) -> None:
        self._crf = settings.output_video_crf
        self._vcodec = settings.output_video_codec
        self._acodec = settings.output_audio_codec
        self._subtitle = SubtitleHandler()
        self._ffmpeg = self._find_ffmpeg()

    def _find_ffmpeg(self) -> str:
        path = shutil.which("ffmpeg")
        if not path:
            raise RuntimeError("ffmpeg not found in PATH. Please install FFmpeg.")
        logger.info("FFmpeg found: %s", path)
        return path

    def _run_ffmpeg(self, args: list[str], desc: str = "") -> None:
        cmd = [self._ffmpeg] + args
        logger.info("FFmpeg %s: %s", desc, " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            logger.error("FFmpeg stderr:\n%s", result.stderr[-2000:] if result.stderr else "")
            raise RuntimeError(f"FFmpeg failed ({desc}): {result.stderr[-500:]}")

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Extract audio track from a video."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_ffmpeg([
            "-i", str(video_path),
            "-vn", "-acodec", "copy",
            "-y", str(output_path),
        ], "extract audio")
        return output_path

    def get_video_info(self, video_path: Path) -> dict:
        """Get video metadata via ffprobe."""
        ffprobe = shutil.which("ffprobe") or self._ffmpeg.replace("ffmpeg", "ffprobe")
        cmd = [
            ffprobe, "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            import json
            return json.loads(result.stdout)
        return {}

    def frames_to_video(
        self, frames_dir: Path, fps: float, output_path: Path,
    ) -> Path:
        """Combine numbered frame PNGs into a video (no audio)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pattern = str(frames_dir / "frame_%08d.png")
        self._run_ffmpeg([
            "-framerate", str(fps),
            "-i", pattern,
            "-c:v", self._vcodec,
            "-crf", str(self._crf),
            "-pix_fmt", "yuv420p",
            "-y", str(output_path),
        ], "frames→video")
        return output_path

    def merge_video_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path,
        srt_path: Path | None = None,
        burn_subtitles: bool = False,
    ) -> Path:
        """Merge video + dubbed audio, optionally burn subtitles."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        args = [
            "-i", str(video_path),
            "-i", str(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
        ]
        if burn_subtitles and srt_path and srt_path.exists():
            sub_filter = self._subtitle.get_ffmpeg_subtitle_filter(srt_path)
            args.extend(["-vf", sub_filter])
            args.extend(["-c:v", self._vcodec, "-crf", str(self._crf)])
        else:
            args.extend(["-c:v", "copy"])
        args.extend([
            "-c:a", self._acodec,
            "-shortest",
            "-y", str(output_path),
        ])
        self._run_ffmpeg(args, "merge video+audio")
        return output_path

    def replace_audio_direct(
        self,
        original_video: Path,
        dubbed_audio: Path,
        output_path: Path,
        srt_path: Path | None = None,
        burn_subtitles: bool = False,
    ) -> Path:
        """
        Replace original audio with dubbed audio directly
        (when no text processing is needed).
        """
        return self.merge_video_audio(
            original_video, dubbed_audio, output_path,
            srt_path, burn_subtitles,
        )

    def assemble_final(
        self,
        processed_frames_dir: Path | None,
        original_video: Path,
        dubbed_audio_path: Path,
        srt_path: Path | None,
        output_path: Path,
        fps: float = 30.0,
        burn_subtitles: bool | None = None,
    ) -> Path:
        """
        Full assembly pipeline:
        1. If processed frames exist → reconstruct video from frames
        2. Merge with dubbed audio
        3. Skip subtitles (disabled)
        """
        burn_subtitles = False

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = settings.temp_path

        if processed_frames_dir and processed_frames_dir.exists():
            # Build video from processed frames
            temp_video = temp_dir / f"{output_path.stem}_frames.mp4"
            self.frames_to_video(processed_frames_dir, fps, temp_video)
            video_source = temp_video
        else:
            video_source = original_video

        # Merge video + audio (subtitles disabled)
        self.merge_video_audio(
            video_source, dubbed_audio_path, output_path,
            srt_path=None, burn_subtitles=False,
        )

        logger.info("Subtitle processing is disabled. Skipping external and burned subtitles.")
        logger.info("✅ Final video assembled → %s", output_path)
        return output_path
