from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yt_dlp
from tqdm import tqdm

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger
logger = get_logger(__name__)


@dataclass
class VideoManifest:
    video_id: str
    source: str
    local_path: str
    title: str | None = None
    duration: float | None = None
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    download_ts: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        if not self.download_ts:
            self.download_ts = datetime.now(timezone.utc).isoformat()
        if not self.video_id:
            self.video_id = str(uuid.uuid4())[:8]


class ProgressHook:
    def __init__(self, video_id: str):
        self.video_id = video_id
        self.pbar: tqdm | None = None
        self.last_downloaded = 0

    def __call__(self, d: dict[str, Any]) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if self.pbar is None and total > 0:
                self.pbar = tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f"Downloading {self.video_id}",
                    leave=False,
                )
            if self.pbar:
                self.pbar.update(downloaded - self.last_downloaded)
                self.last_downloaded = downloaded
        elif d["status"] == "finished" and self.pbar:
            self.pbar.close()
            self.pbar = None
            logger.info("download_complete", video_id=self.video_id)


class VideoDownloader:
    def __init__(self, output_dir: Path | None = None):
        settings = get_settings()
        self.output_dir = output_dir or settings.paths.videos_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ytdlp_format = settings.ingestion.ytdlp_format
        self.download_timeout = settings.ingestion.download_timeout

    def _build_ydl_opts(self, source: str, video_id: str) -> dict[str, Any]:
        output_template = str(self.output_dir / source / "%(id)s.%(ext)s")
        return {
            "format": self.ytdlp_format,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "progress_hooks": [ProgressHook(video_id)],
            "socket_timeout": self.download_timeout,
            "retries": 3,
            "fragment_retries": 3,
            "ignoreerrors": False,
            "noprogress": True,
        }

    def download(self, url: str, source: str | None = None) -> VideoManifest:
        if source is None:
            source = self._detect_source(url)

        with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        video_id = info.get("id", str(uuid.uuid4())[:8])

        ydl_opts = self._build_ydl_opts(source, video_id)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        local_files = list((self.output_dir / source).glob(f"{video_id}.*"))
        if not local_files:
            raise FileNotFoundError(f"Downloaded file not found for {video_id}")

        local_path = local_files[0]

        manifest = VideoManifest(
            video_id=video_id,
            source=source,
            local_path=str(local_path),
            title=info.get("title"),
            duration=info.get("duration"),
            fps=info.get("fps"),
            width=info.get("width"),
            height=info.get("height"),
            metadata={
                "uploader": info.get("uploader"),
                "uploader_id": info.get("uploader_id"),
                "upload_date": info.get("upload_date"),
                "view_count": info.get("view_count"),
                "like_count": info.get("like_count"),
                "description": info.get("description")[:500] if info.get("description") else None,
                "tags": info.get("tags"),
                "categories": info.get("categories"),
                "webpage_url": info.get("webpage_url"),
            },
        )

        manifest_path = local_path.with_suffix(".manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(asdict(manifest), f, indent=2)

        logger.info("video_downloaded", video_id=video_id, source=source, path=str(local_path))
        return manifest

    def download_batch(self, urls: list[str], source: str | None = None) -> list[VideoManifest]:
        manifests = []
        for url in urls:
            try:
                manifest = self.download(url.strip(), source)
                manifests.append(manifest)
            except (yt_dlp.DownloadError, yt_dlp.ExtractorError, FileNotFoundError) as e:
                logger.error("download_failed", url=url, error=str(e))
                continue
        return manifests

    def _detect_source(self, url: str) -> str:
        url_lower = url.lower()
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        elif "tiktok.com" in url_lower:
            return "tiktok"
        elif "twitter.com" in url_lower or "x.com" in url_lower:
            return "twitter"
        elif "instagram.com" in url_lower:
            return "instagram"
        return "generic"