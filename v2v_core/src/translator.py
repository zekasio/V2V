"""
Translator — Gemini-based Turkish→English translation with style preservation.
"""
from __future__ import annotations
import asyncio
import json
from typing import Any
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential
from src.config import settings
from src.logger import get_logger
from src.ocr_engine import DetectedText

logger = get_logger("translator")

SYSTEM_PROMPT = """You are an expert Turkish-to-English translator specializing in video content localization.

RULES:
1. Translate Turkish text to natural, fluent American English.
2. Preserve the original meaning and tone.
3. Keep translations concise — they must fit in the same visual space as the original.
4. Do NOT transliterate — translate the meaning.
5. Maintain any formatting (e.g., line breaks, capitalization style).
6. For idiomatic expressions, find the closest English equivalent.
7. Keep personal names (e.g. 'Kemal Güçlü') unchanged. Keep URLs/social handles unchanged.
8. Translate political titles, slogans, party descriptions, dates, and topics (e.g. '2028 Yılı Cumhurbaşkanı Adayı' → '2028 Presidential Candidate', '7 AĞUSTOS 2026 CUMA SOHBETİ' → 'AUGUST 7, 2026 FRIDAY TALK', 'İMAN ETMEK ALLAH'A (C.C.) GÜVENMEKTİR' → 'TO BELIEVE IS TO TRUST IN ALLAH (SWT)').

Respond ONLY with a JSON object: {"translations": [{"original": "...", "translated": "..."}]}"""


class Translator:
    """Async Gemini-based translator with retry logic."""

    def __init__(self) -> None:
        api_key = settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set. Provide it via .env or environment.")
        self._client = genai.Client(api_key=api_key)
        self._model = settings.gemini_model
        logger.info("Gemini translator initialized (model=%s)", self._model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def translate_texts(self, texts: list[str]) -> dict[str, str]:
        """Translate a batch of Turkish texts to English. Returns {original: translated}."""
        if not texts:
            return {}
        unique_texts = list(set(texts))
        logger.info("Translating %d unique texts", len(unique_texts))
        # Batch into groups of 20
        results: dict[str, str] = {}
        batch_size = 20
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i : i + batch_size]
            batch_results = await self._translate_batch(batch)
            results.update(batch_results)
        logger.info("Translation complete: %d texts", len(results))
        return results

    async def _translate_batch(self, texts: list[str]) -> dict[str, str]:
        user_msg = "Translate these Turkish texts:\n" + json.dumps(texts, ensure_ascii=False)

        # Run the synchronous Gemini SDK call in a thread executor
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, self._call_gemini, user_msg)

        # Parse JSON from response
        content = response.text
        # Try to extract JSON from the response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Gemini may wrap JSON in markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                logger.error("Failed to parse Gemini response as JSON: %s", content[:200])
                return {}

        result: dict[str, str] = {}
        translations = parsed.get("translations", [])
        for item in translations:
            orig = item.get("original", "")
            trans = item.get("translated", "")
            if orig and trans:
                result[orig] = trans
        return result

    def _call_gemini(self, user_msg: str) -> Any:
        """Synchronous Gemini API call."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_msg,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                response_mime_type="application/json",
            ),
        )
        return response

    def _should_skip_translation(self, text: str, category: str = "overlay") -> bool:
        """Check if text is a URL or brand handle that should not be translated."""
        if not text or not text.strip():
            return True
        skip_patterns = [
            "www.", ".com", ".org", ".net", "http://", "https://", "instagram.com", "youtube.com",
            "twitter.com", "tiktok.com",
        ]
        lower = text.strip().lower()
        for pattern in skip_patterns:
            if pattern in lower:
                return True
        return False

    async def translate_detections(
        self, detections: list[DetectedText],
    ) -> list[tuple[DetectedText, str]]:
        """Translate all detected texts and return (detection, translated_text) pairs."""
        results = []
        missing_texts = []
        for det in detections:
            cat = getattr(det, "category", "overlay")
            cached_trans = getattr(det, "translation", "")
            # Even if OCR provided a translation, override skip patterns
            if self._should_skip_translation(det.text, cat):
                det.translation = det.text
                continue
            if not cached_trans:
                missing_texts.append(det.text)

        translations = await self.translate_texts(missing_texts) if missing_texts else {}

        for det in detections:
            cat = getattr(det, "category", "overlay")
            translated = getattr(det, "translation", "")
            if self._should_skip_translation(det.text, cat):
                translated = det.text
            elif not translated:
                translated = translations.get(det.text, det.text)
            results.append((det, translated))
        return results

    async def translate_single(self, text: str) -> str:
        """Translate a single text string."""
        result = await self.translate_texts([text])
        return result.get(text, text)
