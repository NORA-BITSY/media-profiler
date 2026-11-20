from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)

VideoProcessor = Callable[[Path], Awaitable[None]]


class VideoFileHandler(FileSystemEventHandler):
    def __init__(self, processor: VideoProcessor, debounce_seconds: float = 2.0):
        self.processor = processor
        self.debounce_seconds = debounce_seconds
        self._pending: dict[Path, float] = {}
        self._task: asyncio.Task | None = None

    def on_created(self, event: FileCreatedEvent) -> None:
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv"}:
            self._pending[path] = time.time()
            logger.debug("file_detected", path=str(path))

    async def _process_pending(self) -> None:
        while True:
            await asyncio.sleep(self.debounce_seconds)
            now = time.time()
            ready = [
                path for path, ts in self._pending.items()
                if now - ts >= self.debounce_seconds
            ]
            for path in ready:
                self._pending.pop(path, None)
                if path.exists():
                    logger.info("processing_file", path=str(path))
                    try:
                        await self.processor(path)
                    except Exception as e:
                        logger.error("processing_failed", path=str(path), error=str(e))


class FileWatcher:
    def __init__(
        self,
        input_dir: Path | None = None,
        processor: VideoProcessor | None = None,
        recursive: bool = True,
        debounce_seconds: float = 2.0,
    ):
        settings = get_settings()
        self.input_dir = input_dir or settings.paths.videos_dir
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.recursive = recursive
        self.debounce_seconds = debounce_seconds
        self.observer = Observer()
        self.handler = VideoFileHandler(
            processor or self._default_processor,
            debounce_seconds=debounce_seconds,
        )
        self._running = False

    async def _default_processor(self, path: Path) -> None:
        logger.info("default_processor", path=str(path))

    def start(self) -> None:
        if self._running:
            return
        self.observer.schedule(self.handler, str(self.input_dir), recursive=self.recursive)
        self.observer.start()
        self._running = True
        logger.info("watcher_started", path=str(self.input_dir))

    def stop(self) -> None:
        if not self._running:
            return
        self.observer.stop()
        self.observer.join(timeout=5)
        self._running = False
        logger.info("watcher_stopped")

    async def run_forever(self) -> None:
        self.start()
        try:
            while self._running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            self.stop()