import os
import asyncio
import json
import shutil
import time
import numpy as np
from pathlib import Path
from src.config import settings
from src.logger import get_logger
from src.murf_client import MurfDubClient
from src.ocr_engine import OCREngine, TextDetectionResult, DetectedText
from src.text_remover import TextRemover
from src.text_renderer import TextRenderer
from src.translator import Translator
from src.video_processor import VideoProcessor

logger = get_logger("pipeline")


class LocalizationPipeline:
    """Full TR→EN video localization pipeline with single-pass processing."""

    def __init__(self) -> None:
        self._murf = MurfDubClient()
        self._ocr = OCREngine()
        self._remover = TextRemover()
        self._translator = Translator()
        self._renderer = TextRenderer()
        self._video = VideoProcessor()

    async def process_video(self, video_path: Path) -> Path:
        """
        Main pipeline entry point.

        Steps:
        1. Murf dubbing (audio + SRT)
        2. OCR text detection & translation (in one step)
        3. Detections post-translation (if needed)
        4. Single-pass text removal & replacement
        5. Final video assembly
        """
        video_path = Path(video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        stem = video_path.stem
        temp_dir = settings.temp_path / stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = settings.output_path / f"{stem}_EN.mp4"

        logger.info("═" * 60)
        logger.info("🚀 Starting localization: %s", video_path.name)
        logger.info("═" * 60)
        t0 = time.monotonic()

        # ── Step 1: Murf Dubbing ──────────────────────────────
        expected_dubbed = temp_dir / f"{stem}_dubbed.mp4"
        expected_srt = temp_dir / f"{stem}_en.srt"
        if expected_dubbed.exists():
            logger.info("Found cached dubbing files under %s. Skipping Murf Dubbing API call.", temp_dir)
            dubbed_audio = expected_dubbed
            srt_path = expected_srt if expected_srt.exists() else None
        else:
            logger.info("━━ Step 1/6: Murf Dubbing ━━")
            murf_output = await self._step_murf_dubbing(video_path, temp_dir)
            dubbed_audio = murf_output.get("video")  # dubbed video contains the audio
            srt_path = murf_output.get("srt")

        # ── Step 2: OCR Text Detection ────────────────────────
        dets_cache = temp_dir / "detections.json"
        trans_cache = temp_dir / "translations.json"
        if dets_cache.exists():
            logger.info("Found cached detections. Skipping OCR.")
            ocr_result = self._load_detections(dets_cache)
            has_dynamic_subtitles = self._has_dynamic_subtitles(ocr_result)
        else:
            settings.fast_template_mode = True
            logger.info("━━ Step 2/6: Keyframe OCR detection ━━")
            ocr_result = await asyncio.get_running_loop().run_in_executor(
                None, self._ocr.detect_text_in_video, video_path,
            )
            has_dynamic_subtitles = self._has_dynamic_subtitles(ocr_result)
            self._save_detections(ocr_result, dets_cache)

        logger.info("Auto-detection: dynamic subtitles = %s", has_dynamic_subtitles)

        processed_frames_dir = None

        if ocr_result.detections:
            # ── Step 3: Translation ──────────────────────────
            if trans_cache.exists():
                logger.info("Found cached translations. Skipping translation API.")
                translations = self._load_translations(trans_cache, ocr_result.detections)
            else:
                logger.info("━━ Step 3/6: Translation ━━")
                translations = await self._translator.translate_detections(
                    ocr_result.detections,
                )
                self._save_translations(translations, trans_cache)

            # ── Step 4 & 5: Single-pass Inpaint & Render (piped directly to video) ─────
            logger.info("━━ Step 4 & 5/6: Combined Text Removal & replacement ━━")
            processed_video = settings.temp_path / f"{stem}_EN_processed.mp4"
            await asyncio.get_running_loop().run_in_executor(
                None, self._process_and_render_video_fast,
                video_path, processed_video, translations, ocr_result, srt_path,
            )
        else:
            logger.info("No Turkish text detected — skipping steps 3-5")
            processed_video = None

        # ── Step 6: Final Assembly ────────────────────────────
        logger.info("━━ Step 6/6: Final Assembly ━━")
        audio_source = dubbed_audio or video_path
        video_source = processed_video if processed_video and processed_video.exists() else video_path
        self._video.merge_video_audio(
            video_path=video_source,
            audio_path=audio_source,
            output_path=output_path,
        )
        logger.info("✅ Final video assembled → %s", output_path)

        # Cleanup temp processed video
        if processed_video and processed_video.exists():
            try:
                processed_video.unlink()
            except Exception:
                pass

        elapsed = time.monotonic() - t0
        logger.info("═" * 60)
        logger.info("✅ Localization complete in %.1fs → %s", elapsed, output_path)
        logger.info("═" * 60)

        return output_path

    async def _step_murf_dubbing(
        self, video_path: Path, temp_dir: Path,
    ) -> dict[str, Path]:
        try:
            return await self._murf.dub_video(video_path, temp_dir)
        except Exception as e:
            logger.error("Murf dubbing failed: %s", e)
            logger.warning("Continuing without dubbing — original audio will be used")
            return {}

    def _has_dynamic_subtitles(self, ocr_result: TextDetectionResult) -> bool:
        """Check if subtitle text actually changes across frames (dynamic) or stays the same (static)."""
        import unicodedata

        subtitle_dets = [d for d in ocr_result.detections if getattr(d, "category", "overlay") == "subtitle"]
        if not subtitle_dets:
            return False

        def normalize(text: str) -> str:
            mapped = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
            normalized = unicodedata.normalize('NFKD', mapped)
            return "".join(c for c in normalized if not unicodedata.combining(c)).strip().lower()

        texts_by_frame: dict[int, list[str]] = {}
        for d in subtitle_dets:
            texts_by_frame.setdefault(d.frame_index, []).append(normalize(d.text))

        unique_frame_texts = [tuple(sorted(t)) for t in texts_by_frame.values()]
        if len(set(unique_frame_texts)) > 1:
            logger.info("Dynamic subtitles detected: %d unique text sets across frames", len(set(unique_frame_texts)))
            return True

        logger.info("Static subtitles or overlays only — using fast template mode")
        return False

    def _process_and_render_video(
        self,
        video_path: Path,
        output_video: Path,
        translations: list[tuple],
        ocr_result: TextDetectionResult,
        srt_path: Path | None = None,
    ) -> None:
        """Process video frames and pipe directly to FFmpeg — no PNGs written to disk."""
        import cv2
        import subprocess
        import shutil

        fps = ocr_result.video_fps
        total = ocr_result.total_frames

        # Parse SRT to get exact subtitle intervals
        srt_intervals = []
        if srt_path and srt_path.exists():
            try:
                import pysrt
                subs = pysrt.open(str(srt_path), encoding="utf-8")
                for sub in subs:
                    start_ms = sub.start.ordinal
                    end_ms = sub.end.ordinal
                    start_frame = int(start_ms * fps / 1000.0)
                    end_frame = int(end_ms * fps / 1000.0)
                    srt_intervals.append((start_frame, end_frame))
                logger.info("Loaded %d subtitle intervals from SRT for precise inpainting timing", len(srt_intervals))
            except Exception as e:
                logger.error("Failed to parse SRT intervals: %s", e)

        # Build keyframe -> frame range mapping
        keyframes = getattr(ocr_result, "keyframes", [])
        if not keyframes:
            keyframes = sorted(set(d.frame_index for d in ocr_result.detections))
        if not keyframes:
            keyframes = [0]

        keyframe_ranges: dict[int, tuple[int, int]] = {}
        for idx, kf in enumerate(keyframes):
            start = 0 if idx == 0 else (keyframes[idx-1] + kf) // 2
            end = total if idx == len(keyframes) - 1 else (kf + keyframes[idx+1]) // 2
            keyframe_ranges[kf] = (start, end)

        # Build map from detection to its translation using object id
        trans_map = {id(t[0]): t[1] for t in translations}

        det_by_frame: dict[int, list[DetectedText]] = {}
        trans_by_frame: dict[int, list[tuple]] = {}

        def normalize_text(text: str) -> str:
            import unicodedata
            mapped = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
            normalized = unicodedata.normalize('NFKD', mapped)
            cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
            return cleaned.strip().lower()

        # 1. Run Scene Detection to group/deduplicate overlays by scenes
        scenes = []
        try:
            from scenedetect import detect, ContentDetector
            scene_list = detect(str(video_path), ContentDetector(threshold=30.0))
            for scene in scene_list:
                s_f = scene[0].get_frames()
                e_f = scene[1].get_frames()
                scenes.append((s_f, e_f))
        except Exception as e:
            logger.warning("Scene detection failed for overlay mapping: %s", e)

        if not scenes:
            scenes = [(0, total)]
        else:
            # Ensure the scenes cover the whole video
            scenes[0] = (0, scenes[0][1])
            scenes[-1] = (scenes[-1][0], total)

        logger.info("Scene boundaries for overlay range mapping: %s", scenes)

        # 2. Process and deduplicate overlays per scene
        for s_idx, (s_f, e_f) in enumerate(scenes):
            scene_dets = []
            for det in ocr_result.detections:
                if getattr(det, "category", "overlay") == "subtitle":
                    continue
                if s_f <= det.frame_index < e_f:
                    scene_dets.append(det)

            # Deduplicate overlays to prevent double-rendering in the same scene
            unique_overlays = []
            for d in scene_dets:
                duplicate = False
                for existing in unique_overlays:
                    text_sim = normalize_text(d.text) == normalize_text(existing.text)
                    # IoU bounding box overlap check
                    x1 = max(d.x, existing.x)
                    y1 = max(d.y, existing.y)
                    x2 = min(d.x + d.width, existing.x + existing.width)
                    y2 = min(d.y + d.height, existing.y + existing.height)
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    union = (d.width * d.height) + (existing.width * existing.height) - inter
                    iou = inter / union if union > 0 else 0.0
                    
                    if text_sim or iou > 0.5:
                        duplicate = True
                        break
                if not duplicate:
                    unique_overlays.append(d)

            # Map unique overlays to every frame in this scene
            for det in unique_overlays:
                translated_text = trans_map.get(id(det), det.text)
                is_changed = normalize_text(translated_text) != normalize_text(det.text)
                # Keep unchanged overlays exactly as-is in original video
                if not is_changed:
                    continue

                for f in range(s_f, e_f):
                    det_by_frame.setdefault(f, []).append(det)
                    trans_by_frame.setdefault(f, []).append((det, translated_text))

        # 3. Handle subtitles (only if SRT timing is NOT active)
        use_srt_timing = False
        if srt_path and srt_path.exists():
            use_srt_timing = True

        if not use_srt_timing:
            for det in ocr_result.detections:
                if getattr(det, "category", "overlay") == "subtitle":
                    translated_text = trans_map.get(id(det), det.text)
                    kf = det.frame_index
                    s, e = keyframe_ranges.get(kf, (kf, kf))
                    for f in range(s, e):
                        det_by_frame.setdefault(f, []).append(det)
                        trans_by_frame.setdefault(f, []).append((det, translated_text))

        logger.info(
            "Piping %d frames directly to FFmpeg (only %d frame ranges need processing)...",
            total, len(keyframe_ranges)
        )

        # Open the video to get dimensions
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Build global subtitle box and background color
        subtitle_box = None
        sub_bg_color = None
        subtitle_dets = [d for d in ocr_result.detections if getattr(d, "category", "overlay") == "subtitle"]
        if subtitle_dets:
            sub_x1 = min(d.x for d in subtitle_dets)
            sub_y1 = min(d.y for d in subtitle_dets)
            sub_x2 = max(d.x + d.width for d in subtitle_dets)
            sub_y2 = max(d.y + d.height for d in subtitle_dets)
            
            # Margin padding
            margin = 8
            sub_x1 = max(0, sub_x1 - margin)
            sub_y1 = max(0, sub_y1 - margin)
            sub_x2 = min(fw, sub_x2 + margin)
            sub_y2 = min(fh, sub_y2 + margin)
            
            subtitle_box = (sub_x1, sub_y1, sub_x2 - sub_x1, sub_y2 - sub_y1)
            
            # Sample background color
            first_sub_det = subtitle_dets[0]
            cap.set(cv2.CAP_PROP_POS_FRAMES, first_sub_det.frame_index)
            ret_temp, frame_temp = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # seek back to start
            if ret_temp:
                rgb_bg = self._renderer.estimate_bg_color(frame_temp, first_sub_det)
                sub_bg_color = (rgb_bg[2], rgb_bg[1], rgb_bg[0]) # BGR

        # Parse SRT to get exact subtitle intervals and text
        srt_subtitles = []
        use_srt_timing = False
        if srt_path and srt_path.exists():
            try:
                import pysrt
                subs = pysrt.open(str(srt_path), encoding="utf-8")
                for sub in subs:
                    start_ms = sub.start.ordinal
                    end_ms = sub.end.ordinal
                    start_frame = int(start_ms * fps / 1000.0)
                    end_frame = int(end_ms * fps / 1000.0)
                    srt_subtitles.append((start_frame, end_frame, sub.text))
                if srt_subtitles:
                    use_srt_timing = True
                    logger.info("Loaded %d subtitle intervals from SRT for precise frame-timing", len(srt_subtitles))
            except Exception as e:
                logger.error("Failed to parse SRT: %s", e)

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{fw}x{fh}",
            "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", settings.output_video_codec,
            "-crf", str(settings.output_video_crf),
            "-pix_fmt", "yuv420p",
            str(output_video),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                orig = frame.copy()

                # 1. Handle normal overlays (excluding subtitles if using SRT timing)
                frame_dets = det_by_frame.get(frame_idx, [])
                if use_srt_timing:
                    overlay_dets = [d for d in frame_dets if getattr(d, "category", "overlay") != "subtitle"]
                else:
                    overlay_dets = frame_dets
                
                if overlay_dets:
                    frame = self._remover.remove_text(frame, overlay_dets)

                # 2. Subtitles: Always cover the subtitle region with the background color in every frame
                if use_srt_timing and subtitle_box and sub_bg_color:
                    x, y, w, h = subtitle_box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), sub_bg_color, -1)

                # 3. Assemble render list
                render_list = []
                frame_trans = trans_by_frame.get(frame_idx, [])
                if use_srt_timing:
                    for det, trans in frame_trans:
                        if getattr(det, "category", "overlay") != "subtitle":
                            render_list.append((det, trans))
                else:
                    render_list.extend(frame_trans)

                # 4. English subtitles if active
                if use_srt_timing and subtitle_box:
                    active_sub_text = None
                    for s_f, e_f, text in srt_subtitles:
                        if s_f <= frame_idx <= e_f:
                            active_sub_text = text
                            break

                    if active_sub_text:
                        # Create a dummy DetectedText for rendering the subtitle
                        dummy_det = DetectedText(
                            text="",
                            confidence=1.0,
                            x=subtitle_box[0],
                            y=subtitle_box[1],
                            width=subtitle_box[2],
                            height=subtitle_box[3],
                            timestamp=0.0,
                            frame_index=frame_idx,
                            category="subtitle"
                        )
                        render_list.append((dummy_det, active_sub_text))

                # 5. Render everything in a single pass
                if render_list:
                    frame = self._renderer.render_all_texts(frame, render_list, orig)

                proc.stdin.write(frame.tobytes())

                if frame_idx % 500 == 0:
                    logger.info("Piped frame %d/%d", frame_idx, total)
                frame_idx += 1
        finally:
            cap.release()
            proc.stdin.close()
            proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg pipe encoding failed with code {proc.returncode}")
        logger.info("Frame processing complete: %d frames piped → %s", frame_idx, output_video)

    def _save_detections(self, result: TextDetectionResult, path: Path) -> None:
        data = {
            "keyframe_count": result.keyframe_count,
            "total_frames": result.total_frames,
            "video_fps": result.video_fps,
            "video_duration": result.video_duration,
            "keyframes": getattr(result, "keyframes", []),
            "detections": [d.to_dict() for d in result.detections],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d detections → %s", len(result.detections), path.name)

    def _save_translations(self, translations: list[tuple], path: Path) -> None:
        data = [{"original": d.text, "translated": t, "category": getattr(d, "category", "overlay")} for d, t in translations]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved %d translations → %s", len(translations), path.name)

    def _load_detections(self, path: Path) -> TextDetectionResult:
        """Load a previously saved detections.json back into a TextDetectionResult."""
        data = json.loads(path.read_text(encoding="utf-8"))
        result = TextDetectionResult(
            keyframe_count=data.get("keyframe_count", 0),
            total_frames=data.get("total_frames", 0),
            video_fps=data.get("video_fps", 30.0),
            video_duration=data.get("video_duration", 0.0),
            keyframes=data.get("keyframes", []),
        )
        for d in data.get("detections", []):
            result.detections.append(DetectedText(
                text=d["text"], confidence=d["confidence"],
                x=d["x"], y=d["y"], width=d["width"], height=d["height"],
                timestamp=d["timestamp"], frame_index=d["frame_index"],
                bbox_points=d.get("bbox_points", []),
                translation=d.get("translation", ""),
                category=d.get("category", "overlay"),
            ))
        logger.info("Loaded %d detections from cache", len(result.detections))
        return result

    def _load_translations(self, path: Path, detections: list[DetectedText]) -> list[tuple]:
        """Load cached translations.json and pair with detections by text match."""
        data = json.loads(path.read_text(encoding="utf-8"))
        trans_map = {item["original"]: item["translated"] for item in data}
        result = [(det, trans_map.get(det.text, det.text)) for det in detections]
        logger.info("Loaded %d translations from cache", len(result))
        return result

    def _process_and_render_video_fast(
        self,
        video_path: Path,
        output_video: Path,
        translations: list[tuple],
        ocr_result: TextDetectionResult,
        srt_path: Path | None = None,
    ) -> None:
        """Fast template-mode: FFmpeg overlay for static titles + drawtext for subtitles."""
        import cv2
        import subprocess
        import shutil

        logger.info("Running in Fast Template Mode")

        # 1. Open video to get dimensions and detect subtitle region
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Separate bottom speech subtitles (y >= 76% frame height) from upper/middle titles & banners
        bottom_subs = [d for d in ocr_result.detections if getattr(d, "category", "overlay") == "subtitle" and d.y >= int(fh * 0.76)]
        has_subtitles = False
        subtitle_box = None

        if bottom_subs and srt_path and srt_path.exists():
            has_subtitles = True
            sub_x1 = max(10, min(d.x for d in bottom_subs) - 20)
            sub_y1 = max(int(fh * 0.76), min(d.y for d in bottom_subs) - 15)
            sub_x2 = min(fw - 10, max(d.x + d.width for d in bottom_subs) + 20)
            sub_y2 = min(int(fh * 0.90), max(d.y + d.height for d in bottom_subs) + 15)
            subtitle_box = (sub_x1, sub_y1, sub_x2 - sub_x1, sub_y2 - sub_y1)
            logger.info("Dynamic bottom subtitles detected at box: %s", subtitle_box)

        # 2. Build static overlay render list (skip unchanged text and bottom speech subtitles)
        import unicodedata
        def _norm(text: str) -> str:
            mapped = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
            normalized = unicodedata.normalize('NFKD', mapped)
            return "".join(c for c in normalized if not unicodedata.combining(c)).strip().lower()

        trans_map = {id(t[0]): t[1] for t in translations}
        static_render_list = []
        for det in ocr_result.detections:
            # Bottom speech subtitles are covered by subtitle_box and rendered dynamically via FFmpeg drawtext
            if has_subtitles and det.y >= int(fh * 0.76) and getattr(det, "category", "overlay") == "subtitle":
                continue
            translated_text = trans_map.get(id(det), det.text)
            if _norm(translated_text) == _norm(det.text):
                continue
            static_render_list.append((det, translated_text))

        # Deduplicate by normalized text ONLY (ignore frame/position)
        seen_texts = set()
        deduped = []
        for det, trans in static_render_list:
            key = _norm(trans)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            deduped.append((det, trans))
        if len(deduped) < len(static_render_list):
            logger.info("Dedup removed %d duplicate overlays", len(static_render_list) - len(deduped))
        static_render_list = deduped

        # 3. Check for end-screen (subscribe/abone ol) — stop processing after it starts
        end_frame = None
        def _norm2(text: str) -> str:
            mapped = text.replace('İ', 'i').replace('I', 'i').replace('ı', 'i')
            normalized = unicodedata.normalize('NFKD', mapped)
            return "".join(c for c in normalized if not unicodedata.combining(c)).strip().lower()

        # Only trigger if "abone ol" is in the last 10 sampled keyframes
        all_sampled_frames = sorted(set(det.frame_index for det in ocr_result.detections))
        last_10 = set(all_sampled_frames[-10:]) if len(all_sampled_frames) >= 10 else set(all_sampled_frames)
        for det in ocr_result.detections:
            text_lower = _norm2(det.text)
            if text_lower in ("abone ol", "subscribe", "abone olun") and det.frame_index in last_10:
                if end_frame is None or det.frame_index < end_frame:
                    end_frame = det.frame_index
        if end_frame is not None:
            # Add 2-second buffer before the detected frame (Gemini detects late)
            fps = getattr(ocr_result, "video_fps", 30.0)
            buffer_frames = int(fps * 2)
            end_frame = max(0, end_frame - buffer_frames)
            logger.info("End-screen detected at frame %d (with %d-frame buffer) — disabling translations from there",
                        end_frame, buffer_frames)
            # Filter overlay render list to frames before end_frame
            static_render_list = [(det, trans) for det, trans in static_render_list
                                  if det.frame_index < end_frame]

        # 3. Create template PNG with static overlays
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame of video: {video_path}")

        rendered_frame = first_frame.copy()
        if static_render_list:
            rendered_frame = self._renderer.render_all_texts(first_frame, static_render_list, first_frame)

        from PIL import Image
        template_img = Image.new("RGBA", (fw, fh), (0, 0, 0, 0))
        for det, trans in static_render_list:
            font_size = self._renderer.estimate_font_size(trans, det.width, det.height)
            margin = font_size // 2  # match render_all_texts margin
            x1 = max(0, det.x - margin)
            y1 = max(0, det.y - margin)
            x2 = min(fw, det.x + det.width + margin)
            y2 = min(fh, det.y + det.height + margin)
            if x2 <= x1 or y2 <= y1:
                continue
            cropped_bgr = rendered_frame[y1:y2, x1:x2]
            cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            cropped_pil = Image.fromarray(cropped_rgb).convert("RGBA")
            template_img.paste(cropped_pil, (x1, y1))

        # Helper function to sample perimeter background color
        def _estimate_perimeter_bg(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int]:
            bx, by, bw, bh = box
            y1, y2 = max(0, by), min(frame.shape[0], by + bh)
            x1, x2 = max(0, bx), min(frame.shape[1], bx + bw)
            reg = frame[y1:y2, x1:x2]
            if reg.size == 0:
                return (255, 255, 255)
            border = np.vstack([reg[0, :], reg[-1, :], reg[:, 0], reg[:, -1]])
            med = np.median(border, axis=0).astype(int)
            return (int(med[2]), int(med[1]), int(med[0]))  # BGR to RGB

        # Cover subtitle region with perimeter-sampled background color
        if has_subtitles and subtitle_box:
            from PIL import ImageDraw
            rgb_bg = _estimate_perimeter_bg(first_frame, subtitle_box)
            draw = ImageDraw.Draw(template_img)
            draw.rectangle([
                subtitle_box[0], subtitle_box[1],
                subtitle_box[0] + subtitle_box[2], subtitle_box[1] + subtitle_box[3]
            ], fill=(rgb_bg[0], rgb_bg[1], rgb_bg[2], 255))

        # Save template
        temp_dir = settings.temp_path / video_path.stem
        temp_dir.mkdir(parents=True, exist_ok=True)
        template_path = temp_dir / "template.png"
        template_img.save(template_path)
        logger.info("Saved template → %s", template_path)

        # 4. Build FFmpeg command with overlay + drawtext for subtitles
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        # Base overlay filter — optionally stop at end_frame
        if end_frame is not None:
            overlay_filter = f"[0:v][1:v] overlay=0:0:enable='lt(n\\,{end_frame})'"
        else:
            overlay_filter = "[0:v][1:v] overlay=0:0"
        filters = [overlay_filter]

        # Add drawtext for each subtitle entry
        if has_subtitles and srt_path and srt_path.exists():
            try:
                import pysrt
                subs = pysrt.open(str(srt_path), encoding="utf-8")
                sub_x, sub_y = subtitle_box[0], subtitle_box[1]
                sub_w, sub_h = subtitle_box[2], subtitle_box[3]
                font_candidates = [
                    "/system/fonts/Roboto-Regular.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                ]
                font_path = next((p for p in font_candidates if os.path.exists(p)), "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
                font_size = max(14, int(22 * (fh / 1280.0)))

                # Sample subtitle bg color to pick text color
                rgb_bg = _estimate_perimeter_bg(first_frame, subtitle_box)
                lum = 0.299 * rgb_bg[0] + 0.587 * rgb_bg[1] + 0.114 * rgb_bg[2]
                bg_is_light = lum > 128
                logger.info("Subtitle bg color: %s (lum=%.0f, %s)", rgb_bg, lum, "light" if bg_is_light else "dark")

                text_color = "black" if bg_is_light else "white"
                border_color = "white" if bg_is_light else "black"

                import textwrap
                # Use PIL to wrap subtitles by pixel width for safety
                from PIL import ImageFont, ImageDraw, Image as PILImage

                def _wrap_by_px(text: str, font_path: str, font_size: int, max_width: int) -> str:
                    font = ImageFont.truetype(font_path, font_size)
                    lines = []
                    for paragraph in text.split('\n'):
                        words = paragraph.split()
                        if not words:
                            lines.append('')
                            continue
                        cur = words[0]
                        for w in words[1:]:
                            test = cur + ' ' + w
                            bbox = font.getbbox(test)
                            tw = bbox[2] - bbox[0]
                            if tw <= max_width:
                                cur = test
                            else:
                                lines.append(cur)
                                cur = w
                        lines.append(cur)
                    return '\n'.join(lines)

                max_text_width = int(fw * 0.85)  # 85% of frame width

                for sub in subs:
                    start_s = sub.start.ordinal / 1000.0
                    end_s = sub.end.ordinal / 1000.0
                    text = _wrap_by_px(sub.text.replace("'", "\u2019"), font_path, font_size, max_text_width)
                    text = self._renderer.escape_ffmpeg_text(text)
                    enable_expr = f"between(t\\,{start_s:.3f}\\,{end_s:.3f})"
                    if end_frame is not None:
                        enable_expr += f"*lt(n\\,{end_frame})"
                    dt = (
                        f"drawtext=fontfile='{font_path}'"
                        f":text='{text}'"
                        f":fontsize={font_size}"
                        f":fontcolor={text_color}"
                        f":borderw=3"
                        f":bordercolor={border_color}"
                        f":x=(w-text_w)/2"
                        f":y={sub_y + 25}"
                        f":enable='{enable_expr}'"
                    )
                    filters.append(dt)
                logger.info("Built %d drawtext filters (fontsize=%d, %s text on %s bg)",
                           len(subs), font_size, text_color, "light" if bg_is_light else "dark")
            except Exception as e:
                logger.error("Failed to parse SRT for drawtext: %s", e)

        filter_str = ",".join(filters)

        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path),
            "-i", str(template_path),
            "-filter_complex", filter_str,
            "-c:v", settings.output_video_codec,
            "-preset", "superfast",
            "-crf", str(settings.output_video_crf),
            "-pix_fmt", "yuv420p",
            str(output_video),
        ]
        logger.info("Running FFmpeg (fast template)...")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            logger.error("FFmpeg failed: %s", proc.stderr[-500:])
            raise RuntimeError(f"FFmpeg failed: {proc.stderr[-500:]}")
        logger.info("Fast template render complete → %s", output_video)
