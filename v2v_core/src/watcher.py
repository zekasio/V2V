"""
Folder Watcher — monitors input/ for new video files and triggers processing.
"""
from __future__ import annotations
import asyncio
import time
from pathlib import Path
from typing import Callable, Awaitable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
from src.config import settings
from src.logger import get_logger

logger = get_logger("watcher")

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv"}


class VideoFileHandler(FileSystemEventHandler):
    """Watchdog handler that queues new video files for processing."""

    def __init__(self, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._queue = queue
        self._loop = loop
        self._seen: set[str] = set()

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        # Skip temp/partial files
        if path.name.startswith(".") or path.name.startswith("~"):
            return
        key = str(path.resolve())
        if key in self._seen:
            return
        self._seen.add(key)
        logger.info("🎬 New video detected: %s", path.name)
        asyncio.run_coroutine_threadsafe(self._queue.put(path), self._loop)


class FolderWatcher:
    """Watch input directory for new video files."""

    def __init__(
        self,
        process_callback: Callable[[Path], Awaitable[None]],
        input_dir: Path | None = None,
    ) -> None:
        self._input_dir = input_dir or settings.input_path
        self._process_callback = process_callback
        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._observer: Observer | None = None

    async def start(self) -> None:
        """Start watching and processing."""
        self._input_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_running_loop()

        handler = VideoFileHandler(self._queue, loop)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._input_dir), recursive=False)
        self._observer.start()

        logger.info("👁 Watching %s for new videos…", self._input_dir)

        # Process any existing files first
        for f in sorted(self._input_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS:
                await self._queue.put(f)

        try:
            while True:
                path = await self._queue.get()
                # Wait briefly to ensure file is fully written
                await asyncio.sleep(2)
                await self._wait_for_stable(path)
                try:
                    await self._process_callback(path)
                except Exception as e:
                    logger.error("❌ Failed to process %s: %s", path.name, e, exc_info=True)
        except asyncio.CancelledError:
            logger.info("Watcher stopped")
        finally:
            if self._observer:
                self._observer.stop()
                self._observer.join()

    async def _wait_for_stable(self, path: Path, timeout: float = 60.0) -> None:
        """Wait until file size stabilizes (fully written)."""
        start = time.monotonic()
        prev_size = -1
        while time.monotonic() - start < timeout:
            if not path.exists():
                return
            size = path.stat().st_size
            if size == prev_size and size > 0:
                return
            prev_size = size
            await asyncio.sleep(1)
        logger.warning("File %s may not be fully written (timeout)", path.name)

    def stop(self) -> None:
        if self._observer:
            self._observer.stop()
