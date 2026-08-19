"""
OCR Engine — Gemini-based text detection and translation with scene-aware keyframe extraction.
"""
from __future__ import annotations
import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal
import cv2
import numpy as np
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.config import settings
from src.logger import get_logger

logger = get_logger("ocr_engine")

class TextRegionSchema(BaseModel):
    box_2d: list[int] = Field(description="Bounding box coordinates [ymin, xmin, ymax, xmax] normalized to 0-1000")
    text: str = Field(description="Original Turkish text transcribed from the image")
    translation: str = Field(description="English translation of the text")
    category: Literal["subtitle", "overlay"] = Field(
        description="Choose 'subtitle' if the text is a speech subtitle or transcription caption at the bottom of the screen. Choose 'overlay' for other text overlays like titles, banners, presentation slides, and names."
    )

class OCRResponseSchema(BaseModel):
    regions: list[TextRegionSchema] = Field(description="List of detected text regions")

@dataclass
class DetectedText:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    timestamp: float
    frame_index: int = 0
    bbox_points: list[list[int]] = field(default_factory=list)
    translation: str = ""
    category: str = "overlay"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def fingerprint(self) -> str:
        return f"{self.text}:{self.x//20}:{self.y//20}"

@dataclass
class TextDetectionResult:
    detections: list[DetectedText] = field(default_factory=list)
    keyframe_count: int = 0
    total_frames: int = 0
    video_fps: float = 0.0
    video_duration: float = 0.0
    keyframes: list[int] = field(default_factory=list)

class OCREngine:
    def __init__(self) -> None:
        self._client = None
        self._model = settings.gemini_model
        self._confidence_threshold = settings.ocr_confidence_threshold
        self._sample_fps = settings.ocr_sample_fps
        self._use_scene_detection = settings.ocr_use_scene_detection

    def _get_client(self):
        if self._client is None:
            api_key = settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set. Provide it via .env or environment.")
            self._client = genai.Client(api_key=api_key)
            logger.info("Gemini client initialized for OCR (model=%s)", self._model)
        return self._client

    def detect_text_in_video(self, video_path: Path) -> TextDetectionResult:
        logger.info("Starting Gemini-based OCR on %s", video_path.name)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        result = TextDetectionResult(video_fps=fps, total_frames=total_frames, video_duration=duration)
        keyframes = self._select_keyframes(video_path, fps, total_frames)
        result.keyframe_count = len(keyframes)
        result.keyframes = keyframes
        logger.info("Processing %d keyframes (%.1fs @ %.1f fps)", len(keyframes), duration, fps)
        
        last_frame = None
        last_detections = []
        
        for idx, frame_idx in enumerate(keyframes):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = frame_idx / fps
            
            # Run detection on keyframe
            detections = self._detect_text_in_frame(frame, timestamp, frame_idx)
            result.detections.extend(detections)
            
            last_frame = frame
            last_detections = detections
            
            # Small rate-limit courtesy sleep
            if len(keyframes) > 1:
                time.sleep(0.3)
                
        cap.release()
        result.detections = self._deduplicate(result.detections)
        logger.info("OCR complete: %d unique text regions", len(result.detections))
        return result

    def _compute_frame_diff(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute average absolute pixel difference between two resized grayscale frames."""
        gray1 = cv2.cvtColor(cv2.resize(f1, (128, 128)), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cv2.resize(f2, (128, 128)), cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(np.mean(diff))

    def _select_keyframes(self, video_path: Path, fps: float, total_frames: int) -> list[int]:
        frames: set[int] = set()

        if settings.fast_template_mode:
            count = min(6, max(3, int(total_frames / max(1, fps * 15.0))))
            indices = np.linspace(0, total_frames - 1, max(3, min(6, count)), dtype=int)
            for idx in indices:
                frames.add(int(idx))
            logger.info("Fast template mode: selected %d keyframes %s", len(frames), sorted(list(frames)))
            return sorted(list(frames))

        if self._use_scene_detection:
            try:
                from scenedetect import detect, ContentDetector
                scene_list = detect(str(video_path), ContentDetector(threshold=30.0))
                num_scenes = len(scene_list)
                logger.info("Scene detection found %d scenes", num_scenes)
                
                if num_scenes <= 3:
                    # Static video (e.g. single camera view, talking head)
                    # Sample once every 8 seconds to check for text overlay changes
                    interval = max(1, int(fps * 8.0))
                    logger.info("Video is highly static. Sampling every 8 seconds (interval=%d frames)", interval)
                    for f in range(0, total_frames, interval):
                        frames.add(f)
                else:
                    # Dynamic video with scene cuts
                    for scene in scene_list:
                        s, e = scene[0].get_frames(), scene[1].get_frames()
                        duration_sec = (e - s) / fps
                        if duration_sec <= 6.0:
                            # Short scene: take middle frame
                            frames.add(s + (e - s) // 2)
                        else:
                            # Long scene: sample every 5 seconds
                            step = int(fps * 5.0)
                            for f in range(s, e, step):
                                sub_end = min(e, f + step)
                                frames.add(f + (sub_end - f) // 2)
            except Exception as e:
                logger.warning("Scene detection failed: %s", e)

        # Fallback if no frames were selected
        if not frames:
            # Sample one frame every 6 seconds
            interval = max(1, int(fps * 6.0))
            logger.info("Sampling keyframes every 6 seconds (interval=%d frames)", interval)
            for f in range(0, total_frames, interval):
                frames.add(f)
        else:
            logger.info("Keyframes selected: %d frames", len(frames))

        return sorted(list(frames))

    def _detect_text_in_frame(self, frame: np.ndarray, timestamp: float, frame_idx: int) -> list[DetectedText]:
        """Run Gemini API OCR on a single frame with rate limit handling and retry backoff."""
        client = self._get_client()
        detections: list[DetectedText] = []
        fh, fw = frame.shape[:2]

        try:
            # Encode frame to JPEG
            _, encoded_img = cv2.imencode(".jpg", frame)
            image_bytes = encoded_img.tobytes()

            # Retry with exponential backoff on rate limits
            max_attempts = 6
            backoff = 3.0
            response = None
            for attempt in range(max_attempts):
                try:
                    response = client.models.generate_content(
                        model=self._model,
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                            "Detect all Turkish text overlays, titles, captions, and speech subtitles in the image. Include speech subtitles at the bottom of the screen. Categorize each text as either 'subtitle' (speech subtitles/captions at the bottom) or 'overlay' (titles, banners, slides, names, etc.). Provide English translations of the texts, but keep proper nouns, brand names, names of political parties, and people's names in their original Turkish form (do NOT translate them, e.g., keep 'Büyük Medeniyet Partisi' as 'Büyük Medeniyet Partisi', keep 'Kemal Güçlü' as 'Kemal Güçlü')."
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=OCRResponseSchema,
                            temperature=0.0
                        )
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "quota" in err_str.lower() or "limit" in err_str.lower():
                        logger.warning(
                            "Gemini rate limit (429) hit on frame %d. Retrying in %.1fs... (Attempt %d/%d)",
                            frame_idx, backoff, attempt + 1, max_attempts
                        )
                        time.sleep(backoff)
                        backoff *= 2.0
                    else:
                        logger.error("Gemini API error on frame %d: %s", frame_idx, e)
                        break

            if response is None or not response.text:
                return detections

            data = json.loads(response.text)
            for region in data.get("regions", []):
                box = region.get("box_2d")
                if not box or len(box) != 4:
                    continue
                ymin, xmin, ymax, xmax = box

                # Convert normalized (0-1000) coordinates to absolute pixels
                x = int(xmin * fw / 1000)
                y = int(ymin * fh / 1000)
                w = int((xmax - xmin) * fw / 1000)
                h = int((ymax - ymin) * fh / 1000)

                bbox_points = [
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]

                text = region.get("text", "").strip()
                translation = region.get("translation", "").strip()
                category = region.get("category", "overlay").strip().lower()

                if not text:
                    continue

                if self._is_logo_or_watermark(bbox_points, fw, fh, text):
                    continue

                detections.append(DetectedText(
                    text=text,
                    confidence=0.95,
                    x=x, y=y, width=w, height=h,
                    timestamp=round(timestamp, 3), frame_index=frame_idx,
                    bbox_points=bbox_points,
                    translation=translation,
                    category=category
                ))

        except Exception as e:
            logger.warning("Gemini OCR failed on frame %d: %s", frame_idx, e)

        return detections

    def _is_logo_or_watermark(self, bbox: list, fw: int, fh: int, text: str) -> bool:
        pts = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        if w * h < fw * fh * 0.005:
            return True
        cmx, cmy = fw * 0.05, fh * 0.05
        corners = (
            (x < cmx and y < cmy) or
            (x + w > fw - cmx and y < cmy) or
            (x + w > fw - cmx and y + h > fh - cmy)
        )
        if corners and len(text) < 15:
            return True
        brands = ["(C)", "(R)", "(TM)", "www.", "http", ".com", ".org", "instagram", "youtube", "twitter", "tiktok"]
        if any(p in text.lower() for p in brands):
            return True
        return False

    def _deduplicate(self, detections: list[DetectedText]) -> list[DetectedText]:
        fp_map: dict[str, DetectedText] = {}
        for d in detections:
            fp = d.fingerprint
            if fp not in fp_map or d.confidence > fp_map[fp].confidence:
                fp_map[fp] = d
        removed = len(detections) - len(fp_map)
        if removed > 0:
            logger.info("Dedup removed %d redundant detections", removed)
        return list(fp_map.values())

    def detect_text_in_image(self, image: np.ndarray) -> list[DetectedText]:
        return self._detect_text_in_frame(image, 0.0, 0)
