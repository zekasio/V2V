import os, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve() / "v2v_core"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.resolve() / "v2v_core" / ".env")

from src.config import settings
from src.logger import get_logger
from src.ocr_engine import OCREngine
from src.translator import Translator
from src.text_renderer import TextRenderer
from src.ocr_engine import DetectedText

logger = get_logger("test_run")

def find_video():
    root = Path(__file__).parent.resolve()
    mp4 = list(root.glob("*.mp4"))
    if mp4:
        return mp4[0]
    return None

async def main():
    video = find_video()
    if not video:
        print("[-] No mp4 found in V2V/")
        return

    print(f"[+] Video: {video.name}")
    out_dir = Path(__file__).parent.resolve() / "test_output"
    out_dir.mkdir(exist_ok=True)

    settings.fast_template_mode = True  # 3 frames only

    # ── OCR ──
    print("[+] Running OCR (3 keyframes)...")
    ocr = OCREngine()
    result = await asyncio.get_running_loop().run_in_executor(
        None, ocr.detect_text_in_video, video,
    )

    print(f"    Detected {len(result.detections)} texts")
    for d in result.detections:
        print(f"    [{d.category:8s}] ({d.x:4d},{d.y:4d},{d.width:4d}x{d.height:4d}) {d.text}")

    # ── Translation ──
    print("[+] Translating...")
    translator = Translator()
    translations = await translator.translate_detections(result.detections)

    print(f"    Translated {len(translations)} texts")
    for det, trans in translations:
        changed = "✓" if trans != det.text else "–"
        print(f"    [{changed}] {det.text[:50]:50s} → {trans[:50]}")

    # ── Save detections + translations as JSON for inspection ──
    data = []
    for det, trans in translations:
        data.append({
            "category": getattr(det, "category", "overlay"),
            "original": det.text,
            "translated": trans,
            "x": det.x, "y": det.y, "w": det.width, "h": det.height,
        })
    json_path = out_dir / f"{video.stem}_test.json"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[+] JSON → {json_path}")

    # ── Render preview on first frame ──
    print("[+] Rendering preview on first frame...")
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[-] Cannot read first frame")
        return

    fh, fw = frame.shape[:2]
    renderer = TextRenderer()

    subtitle_dets = [d for d in result.detections if getattr(d, "category", "overlay") == "subtitle"]
    has_sub = False
    sub_box = None
    if subtitle_dets:
        texts_by_frame = {}
        for d in subtitle_dets:
            texts_by_frame.setdefault(d.frame_index, []).append(d.text)
        unique = [tuple(sorted(t)) for t in texts_by_frame.values()]
        if len(set(unique)) > 1:
            has_sub = True
            sx1 = min(d.x for d in subtitle_dets)
            sy1 = min(d.y for d in subtitle_dets)
            sx2 = max(d.x + d.width for d in subtitle_dets)
            sy2 = max(d.y + d.height for d in subtitle_dets)
            m = 15
            sx1 = max(0, sx1 - m)
            sy1 = max(0, sy1 - m)
            sx2 = min(fw, sx2 + m)
            sy2 = min(fh, sy2 + m)
            sub_box = (sx1, sy1, sx2 - sx1, sy2 - sy1)

    # Render all overlays except dynamic subtitles
    preview = frame.copy()
    overlay_list = []
    trans_map = {id(t[0]): t[1] for t in translations}
    subtitle_list = []

    for det in result.detections:
        cat = getattr(det, "category", "overlay")
        trans = trans_map.get(id(det), det.text)
        if cat == "subtitle" and has_sub:
            subtitle_list.append((det, trans))
        else:
            overlay_list.append((det, trans))

    if overlay_list:
        preview = renderer.render_all_texts(preview, overlay_list, frame)

    # Cover subtitle region
    if has_sub and sub_box:
        import numpy as np
        rgb_bg = renderer.estimate_bg_color(frame, DetectedText(
            text="", confidence=1.0,
            x=sub_box[0], y=sub_box[1],
            width=sub_box[2], height=sub_box[3],
            timestamp=0.0, frame_index=0
        ))
        bg_bgr = (rgb_bg[2], rgb_bg[1], rgb_bg[0])
        cv2.rectangle(preview, (sub_box[0], sub_box[1]),
                      (sub_box[0] + sub_box[2], sub_box[1] + sub_box[3]), bg_bgr, -1)

    # Render subtitles on top with proper color
    if subtitle_list:
        # Check bg color for text color choice
        lum = 0.299 * rgb_bg[0] + 0.587 * rgb_bg[1] + 0.114 * rgb_bg[2]
        bg_is_light = lum > 128
        # We don't need to render subtitles here since drawtext will do it
        # But for the preview, render them
        pass

    # Save preview
    preview_path = out_dir / f"{video.stem}_preview.png"
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
    Image.fromarray(preview_bgr).save(preview_path)
    print(f"[+] Preview → {preview_path}")
    print("[+] Done!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
