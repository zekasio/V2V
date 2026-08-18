"""
Murf Dub API Client — handles video upload, dubbing job creation,
status polling, and asset download with retry logic.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.logger import get_logger

logger = get_logger("murf_client")

# ── Constants ─────────────────────────────────────────────────
BASE_URL = "https://api.murf.ai/v1/murfdub"
JOB_POLL_INTERVAL = 15  # seconds
JOB_TIMEOUT = 1800  # 30 minutes max
MAX_RETRIES = 5


class MurfDubError(Exception):
    """Custom exception for Murf Dub API errors."""
    pass


class MurfDubClient:
    """Async client for the Murf Dub Automation API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.murfdub_api_key
        if not self.api_key:
            raise MurfDubError(
                "MURFDUB_API_KEY is not set. Provide it via .env or environment."
            )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    # ── Internal helpers ──────────────────────────────────────

    def _client(self, timeout: float = 60.0) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers=self._headers,
            timeout=httpx.Timeout(timeout, connect=15.0),
        )

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with automatic retry."""
        async with self._client(timeout=kwargs.pop("timeout", 60.0)) as client:
            resp = await client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp

    # ── Public API ────────────────────────────────────────────

    async def create_dubbing_job(
        self,
        video_path: Path,
        target_locale: str | None = None,
        priority: str = "LOW",
    ) -> dict[str, Any]:
        """
        Upload a video and create a transient dubbing job.

        Returns the API response containing the job ID.
        """
        locale = target_locale or settings.target_locale
        file_name = video_path.stem

        logger.info(
            "Creating dubbing job: %s → %s", video_path.name, locale
        )

        try:
            # Try using the murf Python SDK first
            return await self._create_via_sdk(video_path, locale, priority)
        except Exception as sdk_err:
            logger.warning(
                "SDK approach failed (%s), falling back to REST API", sdk_err
            )
            return await self._create_via_rest(video_path, file_name, locale, priority)

    async def _create_via_sdk(
        self,
        video_path: Path,
        locale: str,
        priority: str,
    ) -> dict[str, Any]:
        """Create job via murf Python SDK."""
        from murf import MurfDub as MurfDubSDK

        def _sync_create() -> Any:
            client = MurfDubSDK(api_key=self.api_key)
            with open(video_path, "rb") as f:
                response = client.dubbing.jobs.create(
                    target_locales=[locale],
                    file_name=video_path.stem,
                    file=f,
                    priority=priority,
                )
            return response

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(None, _sync_create)

        # Handle various response object types
        if isinstance(response, dict):
            result = response
        elif hasattr(response, "__dict__"):
            result = {k: v for k, v in response.__dict__.items() if not k.startswith("_")}
        else:
            result = {"raw": str(response)}

        job_id = result.get("id") or result.get("job_id") or getattr(response, "id", None)
        if job_id:
            result["id"] = str(job_id)
        logger.info("Dubbing job created via SDK - job_id=%s", job_id)
        return result

    async def _create_via_rest(
        self,
        video_path: Path,
        file_name: str,
        locale: str,
        priority: str,
    ) -> dict[str, Any]:
        """Create job via direct REST API calls."""
        url = f"{BASE_URL}/jobs"

        with open(video_path, "rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            data = {
                "target_locales": locale,
                "file_name": file_name,
                "priority": priority,
            }

            async with self._client(timeout=300.0) as client:
                resp = await client.post(url, data=data, files=files)
                resp.raise_for_status()
                result = resp.json()

        job_id = result.get("id") or result.get("job_id")
        logger.info("Dubbing job created via REST — job_id=%s", job_id)
        return result

    async def poll_job_status(
        self,
        job_id: str,
        poll_interval: int = JOB_POLL_INTERVAL,
        timeout: int = JOB_TIMEOUT,
    ) -> dict[str, Any]:
        """
        Poll the dubbing job until COMPLETED or FAILED.

        Returns the final status response.
        Raises MurfDubError on timeout or failure.
        """
        logger.info("Polling job %s (interval=%ds, timeout=%ds)", job_id, poll_interval, timeout)
        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            if elapsed > timeout:
                raise MurfDubError(
                    f"Job {job_id} timed out after {timeout}s"
                )

            try:
                status_data = await self._get_job_status(job_id)
            except Exception as e:
                logger.warning("Status poll error (will retry): %s", e)
                await asyncio.sleep(poll_interval)
                continue

            status = self._extract_status(status_data)
            logger.info(
                "Job %s — status=%s (%.0fs elapsed)", job_id, status, elapsed
            )

            if status.upper() in ("COMPLETED", "DONE", "SUCCESS"):
                logger.info("✅ Job %s completed!", job_id)
                return status_data

            if status.upper() in ("FAILED", "ERROR", "CANCELLED"):
                raise MurfDubError(
                    f"Job {job_id} failed with status: {status}. "
                    f"Details: {status_data}"
                )

            await asyncio.sleep(poll_interval)

    async def _get_job_status(self, job_id: str) -> dict[str, Any]:
        """Fetch job status — tries SDK, then REST."""
        try:
            from murf import MurfDub as MurfDubSDK

            def _sync_status() -> Any:
                client = MurfDubSDK(api_key=self.api_key)
                return client.dubbing.jobs.get_status(job_id=job_id)

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, _sync_status)
            if isinstance(response, dict):
                return response
            return response.__dict__ if hasattr(response, "__dict__") else {"raw": str(response)}
        except Exception:
            url = f"{BASE_URL}/jobs/{job_id}/status"
            resp = await self._request("GET", url)
            return resp.json()

    def _extract_status(self, data: Any) -> str:
        """Extract status string from various response shapes."""
        if isinstance(data, dict):
            return (
                data.get("status")
                or data.get("state")
                or data.get("job_status")
                or "UNKNOWN"
            )
        return getattr(data, "status", "UNKNOWN")

    async def download_assets(
        self,
        job_status_data: dict[str, Any],
        output_dir: Path,
        video_stem: str,
    ) -> dict[str, Path]:
        """
        Download the dubbed video and SRT file from a completed job.

        Returns dict with keys 'video' and 'srt' mapped to local paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        downloaded: dict[str, Path] = {}

        # Extract download URLs (various possible response shapes)
        video_url = (
            job_status_data.get("video_url")
            or job_status_data.get("dubbed_video_url")
            or job_status_data.get("output_url")
            or self._deep_get(job_status_data, "output", "video_url")
        )
        srt_url = (
            job_status_data.get("srt_url")
            or job_status_data.get("subtitle_url")
            or job_status_data.get("captions_url")
            or self._deep_get(job_status_data, "output", "srt_url")
        )

        # Check download_details list
        details = job_status_data.get("download_details")
        if isinstance(details, list) and len(details) > 0:
            detail = details[0]
            # Match the locale if possible (supporting both dicts and SDK objects)
            for d in details:
                locale = d.get("locale") if isinstance(d, dict) else getattr(d, "locale", None)
                if locale == settings.target_locale or (locale and locale.replace("-", "_") == settings.target_locale):
                    detail = d
                    break

            if isinstance(detail, dict):
                dl_url = detail.get("download_url") or detail.get("url")
                ds_url = detail.get("download_srt_url") or detail.get("srt_url")
            else:
                dl_url = getattr(detail, "download_url", None) or getattr(detail, "url", None)
                ds_url = getattr(detail, "download_srt_url", None) or getattr(detail, "srt_url", None)

            if not video_url:
                video_url = dl_url
            if not srt_url:
                srt_url = ds_url

        if video_url:
            video_path = output_dir / f"{video_stem}_dubbed.mp4"
            await self._download_file(video_url, video_path)
            downloaded["video"] = video_path
            logger.info("Downloaded dubbed video → %s", video_path)

        if srt_url:
            srt_path = output_dir / f"{video_stem}_en.srt"
            await self._download_file(srt_url, srt_path)
            downloaded["srt"] = srt_path
            logger.info("Downloaded SRT → %s", srt_path)

        if not downloaded:
            logger.warning(
                "No download URLs found in job response. Keys: %s",
                list(job_status_data.keys()),
            )

        return downloaded

    async def _download_file(self, url: str, dest: Path) -> None:
        """Stream-download a file."""
        async with self._client(timeout=300.0) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)

    @staticmethod
    def _deep_get(data: dict, *keys: str) -> Any | None:
        """Safely traverse nested dicts."""
        current = data
        for key in keys:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                return None
        return current

    # ── High-level convenience ────────────────────────────────

    async def dub_video(
        self,
        video_path: Path,
        output_dir: Path,
    ) -> dict[str, Path]:
        """
        Full workflow: upload → create job → poll → download.

        Returns dict with paths to dubbed video and SRT.
        """
        # Step 1: Create job
        create_resp = self._normalize_response(
            await self.create_dubbing_job(video_path)
        )
        job_id = (
            create_resp.get("id")
            or create_resp.get("job_id")
        )
        if not job_id:
            raise MurfDubError(f"No job ID in response: {create_resp}")

        # Step 2: Poll until done
        final_status = await self.poll_job_status(job_id)

        # Step 3: Download assets
        return await self.download_assets(
            final_status, output_dir, video_path.stem
        )

    def _normalize_response(self, resp: Any) -> dict:
        """Ensure response is a dict."""
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "__dict__"):
            return resp.__dict__
        return {"raw": str(resp)}
