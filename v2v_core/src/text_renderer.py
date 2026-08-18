"""
Text Renderer — renders translated English text back onto video frames,
matching original position, size, color, and style.
"""
from __future__ import annotations
import textwrap
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.config import settings
from src.logger import get_logger
from src.ocr_engine import DetectedText

logger = get_logger("text_renderer")


class TextRenderer:
    """Render translated text onto video frames matching original style."""

    def __init__(self) -> None:
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}
        self._style_cache: dict[tuple[str, int, int], tuple[int, list[str], int, int, int]] = {}
        self._color_cache: dict[tuple[str, int, int, int, int], tuple[tuple[int, int, int], tuple[int, int, int]]] = {}

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        size = max(8, size)
        if size not in self._font_cache:
            paths = [
                "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
                "/data/data/com.termux/files/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/system/fonts/Roboto-Regular.ttf",
                "arialbd.ttf", "arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
            for path in paths:
                try:
                    self._font_cache[size] = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def escape_ffmpeg_text(self, text: str) -> str:
        """Escape special characters for FFmpeg drawtext filter."""
        return text.replace('\\', '\\\\').replace(':', '\\:').replace(',', '\\,')

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
        """Wrap text into lines that fit within max_width pixels."""
        words = text.split()
        if not words:
            return [text]
        lines = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else [text]

    def _measure_lines(self, lines: list[str], font: ImageFont.FreeTypeFont) -> tuple[int, int]:
        """Return (total_width, total_height) for a list of wrapped lines."""
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        spacing = max(2, line_height // 6)
        total_h = len(lines) * line_height + (len(lines) - 1) * spacing
        total_w = max((font.getbbox(l)[2] - font.getbbox(l)[0]) for l in lines)
        return total_w, total_h

    def estimate_font_size(self, text: str, target_width: int, target_height: int) -> int:
        """Font size 10-22."""
        effective_height = int(target_height * 1.5)
        allowed_width = target_width * 1.3
        for size in range(22, 9, -1):
            font = self._get_font(size)
            lines = self._wrap_text(text, font, allowed_width)
            _, total_h = self._measure_lines(lines, font)
            if total_h <= effective_height:
                return size
        return 10

    def extract_dominant_color(
        self, frame: np.ndarray, det: DetectedText,
    ) -> tuple[int, int, int]:
        """Sample the text color from the original detection region."""
        x, y, w, h = det.x, det.y, det.width, det.height
        fh, fw = frame.shape[:2]
        cy = min(max(y + h // 2, 0), fh - 1)
        cx = min(max(x + w // 2, 0), fw - 1)
        sample_h = max(1, h // 4)
        sample_w = max(1, w // 2)
        y1, y2 = max(0, cy - sample_h // 2), min(fh, cy + sample_h // 2)
        x1, x2 = max(0, cx - sample_w // 2), min(fw, cx + sample_w // 2)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return (255, 255, 255)
        pixels = region.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            text_idx = np.argmin(counts)
            color = centers[text_idx].astype(int)
            return (int(color[2]), int(color[1]), int(color[0]))  # BGR→RGB
        except Exception:
            avg = np.mean(pixels, axis=0).astype(int)
            return (int(avg[2]), int(avg[1]), int(avg[0]))

    def estimate_bg_color(
        self, frame: np.ndarray, det: DetectedText,
    ) -> tuple[int, int, int]:
        """Estimate background color from the inpainted (cleaned) frame region."""
        x, y, w, h = det.x, det.y, det.width, det.height
        fh, fw = frame.shape[:2]
        # Sample the center of the cleaned region (background should be visible now)
        y1 = max(0, y)
        y2 = min(fh, y + h)
        x1 = max(0, x)
        x2 = min(fw, x + w)
        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return (0, 0, 0)
        pixels = region.reshape(-1, 3).astype(np.float32)
        # Use the most frequent color cluster as background
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        try:
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            bg_idx = np.argmax(counts)
            color = centers[bg_idx].astype(int)
            return (int(color[2]), int(color[1]), int(color[0]))
        except Exception:
            avg = np.mean(pixels, axis=0).astype(int)
            return (int(avg[2]), int(avg[1]), int(avg[0]))

    def render_text_on_frame(
        self,
        frame: np.ndarray,
        det: DetectedText,
        translated_text: str,
        original_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render translated text onto a frame at the detection position."""
        return self.render_all_texts(frame, [(det, translated_text)], original_frame)

    def render_all_texts(
        self,
        frame: np.ndarray,
        translations: list[tuple[DetectedText, str]],
        original_frame: np.ndarray | None = None,
        zone_bbox: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Render multiple translated texts onto a frame using localized high-performance crops."""
        if zone_bbox:
            zx, zy, zw, zh = zone_bbox
            translations = [
                (det, trans) for det, trans in translations
                if (det.x >= zx and det.y >= zy and
                    (det.x + det.width) <= (zx + zw) and
                    (det.y + det.height) <= (zy + zh))
            ]

        if not translations:
            return frame.copy()

        result = frame.copy()
        fh, fw = frame.shape[:2]
        ref = original_frame if original_frame is not None else frame

        for det, translated_text in translations:
            # Clamp detection box to frame bounds
            x = max(0, min(det.x, fw - 1))
            y = max(0, min(det.y, fh - 1))
            w = max(1, min(det.width, fw - x))
            h = max(1, min(det.height, fh - y))

            # Style cache lookup
            cache_key = (translated_text, w, h)
            if cache_key in self._style_cache:
                font_size, lines, total_text_h, line_h, spacing = self._style_cache[cache_key]
            else:
                font_size = self.estimate_font_size(translated_text, w, h)
                font = self._get_font(font_size)
                lines = self._wrap_text(translated_text, font, w - 4)
                line_h = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
                spacing = max(2, line_h // 6)
                total_text_h = len(lines) * line_h + (len(lines) - 1) * spacing
                self._style_cache[cache_key] = (font_size, lines, total_text_h, line_h, spacing)

            # Dynamic text/outline based on background luminance
            ref_crop = result[y:y+h, x:x+w]
            bg_lum = np.mean(cv2.cvtColor(ref_crop, cv2.COLOR_BGR2GRAY))
            is_light = bg_lum > 100
            if is_light:
                text_color = (0, 0, 0)
                outline_color = (255, 255, 255)
            else:
                text_color = (255, 255, 255)
                outline_color = (0, 0, 0)

            # Crop localized sub-region to prevent expensive full-frame conversion
            outline_width = max(2, font_size // 10)
            margin = font_size // 2  # small margin to avoid overlapping crops
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(fw, x + w + margin)
            y2 = min(fh, y + h + margin)

            crop_w = x2 - x1
            crop_h = y2 - y1
            if crop_w <= 0 or crop_h <= 0:
                continue

            crop = result[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            draw = ImageDraw.Draw(pil_img)

            # Cover old text with estimated background color
            bg_color = self.estimate_bg_color(ref, det)
            draw.rectangle([x - x1, y - y1, x + w - x1, y + h - y1], fill=bg_color)

            font = self._get_font(font_size)
            text_y = y - y1 + (h - total_text_h) // 2

            for line in lines:
                line_bbox = font.getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                text_x = x - x1 + (w - line_w) // 2

                # Clamp locally
                text_x = max(x - x1, min(text_x, crop_w - line_w - 1))
                ty = max(0, min(text_y, crop_h - line_h - 1))

                # Draw outline
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx == 0 and dy == 0:
                            continue
                        draw.text((text_x + dx, ty + dy), line, font=font, fill=outline_color)

                # Draw text
                draw.text((text_x, ty), line, font=font, fill=text_color)
                text_y += line_h + spacing

            # Paste back the rendered crop
            result[y1:y2, x1:x2] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        return result
