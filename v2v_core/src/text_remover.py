"""
Text Remover — LaMa inpainting (primary) with OpenCV fallback.
Removes detected Turkish text from video frames.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from src.config import settings
from src.logger import get_logger
from src.ocr_engine import DetectedText

logger = get_logger("text_remover")


class TextRemover:
    """Remove text regions from frames using inpainting."""

    def __init__(self) -> None:
        self._method = settings.inpainting_method
        self._dilation = settings.inpainting_mask_dilation
        self._lama_model = None
        self._inpainting_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _load_lama(self):
        if self._lama_model is None:
            try:
                from simple_lama_inpainting import SimpleLama
                self._lama_model = SimpleLama()
                logger.info("LaMa inpainting model loaded")
            except Exception as e:
                logger.warning("LaMa unavailable (%s), will use OpenCV fallback", e)
                self._method = "telea"
        return self._lama_model

    def create_text_mask(
        self, frame: np.ndarray, detections: list[DetectedText], zone_bbox: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Create a binary mask covering all detected text regions, restricted to a zone if provided."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        # If a zone is defined, create a zone mask to restrict operations
        zone_mask = None
        if zone_bbox:
            zx, zy, zw, zh = zone_bbox
            zone_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.rectangle(zone_mask, (zx, zy), (zx + zw, zy + zh), 255, -1)

        for det in detections:
            # Skip if zone is defined and detection is outside
            if zone_bbox:
                zx, zy, zw, zh = zone_bbox
                if not (det.x >= zx and det.y >= zy and (det.x + det.width) <= (zx + zw) and (det.y + det.height) <= (zy + zh)):
                    continue

            if det.bbox_points and len(det.bbox_points) >= 4:
                pts = np.array(det.bbox_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            else:
                x, y, w, h = det.x, det.y, det.width, det.height
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Dilate to cover edges
        if self._dilation > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self._dilation * 2, self._dilation * 2)
            )
            mask = cv2.dilate(mask, kernel, iterations=1)
            
        # Restrict mask to zone if provided
        if zone_mask is not None:
            mask = cv2.bitwise_and(mask, zone_mask)
            
        return mask

    def remove_text(
        self, frame: np.ndarray, detections: list[DetectedText], zone_bbox: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Remove text regions from a frame by running inpainting on cropped areas for maximum performance."""
        if zone_bbox:
            zx, zy, zw, zh = zone_bbox
            detections = [
                d for d in detections 
                if d.x >= zx and d.y >= zy and (d.x + d.width) <= (zx + zw) and (d.y + d.height) <= (zy + zh)
            ]
            
        if not detections:
            return frame.copy()

        result = frame.copy()
        fh, fw = frame.shape[:2]
        margin = max(3, self._dilation) + 8

        for det in detections:
            if det.bbox_points and len(det.bbox_points) >= 4:
                pts = np.array(det.bbox_points, dtype=np.int32)
                x, y, w, h = cv2.boundingRect(pts)
            else:
                x, y, w, h = det.x, det.y, det.width, det.height

            # Crop coords
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(fw, x + w + margin)
            y2 = min(fh, y + h + margin)

            if x2 <= x1 or y2 <= y1:
                continue

            # Localized crop mask
            crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
            if det.bbox_points and len(det.bbox_points) >= 4:
                offset_pts = pts - np.array([x1, y1])
                cv2.fillPoly(crop_mask, [offset_pts], 255)
            else:
                cv2.rectangle(crop_mask, (x - x1, y - y1), (x + w - x1, y + h - y1), 255, -1)

            if self._dilation > 0:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (self._dilation * 2, self._dilation * 2)
                )
                crop_mask = cv2.dilate(crop_mask, kernel, iterations=1)

            crop_frame = result[y1:y2, x1:x2]
            if crop_frame.size == 0 or np.sum(crop_mask) == 0:
                continue

            fingerprint = getattr(det, "fingerprint", None)
            crop_inpainted = None
            use_cached = False

            if fingerprint and fingerprint in self._inpainting_cache:
                cached_crop, cached_mask = self._inpainting_cache[fingerprint]
                if cached_crop.shape == crop_frame.shape:
                    diff = cv2.absdiff(crop_frame, cached_crop)
                    compare_mask = cv2.bitwise_not(crop_mask)
                    mean_diff = sum(cv2.mean(diff, mask=compare_mask)[:3]) / 3.0
                    if mean_diff < 12.0:  # Allow some compression/sensor noise
                        crop_frame[crop_mask > 0] = cached_crop[crop_mask > 0]
                        crop_inpainted = crop_frame
                        use_cached = True

            if not use_cached:
                radius = 3
                # Run inpaint on crop
                if self._method == "lama":
                    model = self._load_lama()
                    if model is not None:
                        try:
                            from PIL import Image
                            img_pil = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
                            mask_pil = Image.fromarray(crop_mask)
                            result_pil = model(img_pil, mask_pil)
                            crop_inpainted = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                        except Exception:
                            crop_inpainted = cv2.inpaint(crop_frame, crop_mask, radius, cv2.INPAINT_TELEA)
                    else:
                        crop_inpainted = cv2.inpaint(crop_frame, crop_mask, radius, cv2.INPAINT_TELEA)
                elif self._method == "ns":
                    crop_inpainted = cv2.inpaint(crop_frame, crop_mask, radius, cv2.INPAINT_NS)
                else:
                    crop_inpainted = cv2.inpaint(crop_frame, crop_mask, radius, cv2.INPAINT_TELEA)

                if fingerprint:
                    self._inpainting_cache[fingerprint] = (crop_inpainted.copy(), crop_mask.copy())

            result[y1:y2, x1:x2] = crop_inpainted

        return result

    def process_video_frames(
        self, video_path: Path, detections: list[DetectedText], output_dir: Path,
    ) -> Path:
        """
        Process all frames of a video, removing text where detected.
        Writes cleaned frames to output_dir as numbered PNGs.
        Returns output_dir.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Index detections by frame range
        det_by_frame = self._index_detections(detections, fps, total)
        logger.info("Removing text from %d frames (%.1f fps)", total, fps)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_dets = det_by_frame.get(frame_idx, [])
            if frame_dets:
                frame = self.remove_text(frame, frame_dets)
            out_path = output_dir / f"frame_{frame_idx:08d}.png"
            cv2.imwrite(str(out_path), frame)
            if frame_idx % 500 == 0:
                logger.debug("Processed frame %d/%d", frame_idx, total)
            frame_idx += 1
        cap.release()
        logger.info("Text removal complete: %d frames", frame_idx)
        return output_dir

    def _index_detections(
        self, detections: list[DetectedText], fps: float, total_frames: int,
    ) -> dict[int, list[DetectedText]]:
        """Map detections to frame ranges (extend each detection ±1 second)."""
        by_frame: dict[int, list[DetectedText]] = {}
        spread = int(fps)  # ±1 second
        for det in detections:
            center = det.frame_index
            for f in range(max(0, center - spread), min(total_frames, center + spread + 1)):
                by_frame.setdefault(f, []).append(det)
        return by_frame
