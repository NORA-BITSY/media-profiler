from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticFeatures:
    video_id: str
    description: str
    keyframe_count: int
    embedding: np.ndarray
    keyframe_descriptions: list[str]


class SemanticExtractor:
    def __init__(
        self,
        vlm_model: str = "moondream2",
        vlm_host: str = "http://localhost:11434",
        vlm_timeout: int = 30,
        max_keyframes: int = 20,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.vlm_model = vlm_model
        self.vlm_host = vlm_host.rstrip("/")
        self.vlm_timeout = vlm_timeout
        self.max_keyframes = max_keyframes
        self.embedding_model_name = embedding_model
        self._embedding_model: SentenceTransformer | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.vlm_timeout)
        return self._client

    def _encode_image(self, frame: np.ndarray) -> str:
        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode("utf-8")

    async def _query_vlm(self, image_b64: str, prompt: str) -> str:
        payload = {
            "model": self.vlm_model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        try:
            response = await self.client.post(f"{self.vlm_host}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("response", "").strip()
        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            logger.error("vlm_error", error=str(e))
            return ""

    async def extract_from_video(self, video_path: Path, video_id: str, keyframe_rate: float = 0.5) -> SemanticFeatures | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("cannot_open_video", path=str(video_path))
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / keyframe_rate)) if keyframe_rate > 0 else max(1, total_frames // self.max_keyframes)
        frame_interval = min(frame_interval, max(1, total_frames // self.max_keyframes))

        prompt = (
            "Describe this video frame in one sentence. Focus on: "
            "- Setting/environment\n"
            "- Number of people visible\n"
            "- Primary activity/action\n"
            "- Notable visual attributes (clothing colors, objects)"
        )

        keyframe_descriptions = []
        frame_idx = 0
        processed = 0

        while processed < self.max_keyframes:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            image_b64 = self._encode_image(frame)
            description = await self._query_vlm(image_b64, prompt)

            if description:
                keyframe_descriptions.append(description)
                processed += 1

            frame_idx += 1

        cap.release()

        if not keyframe_descriptions:
            logger.warning("no_keyframes_processed", video_id=video_id)
            return None

        full_description = " ".join(keyframe_descriptions)
        embedding = self.embedding_model.encode(full_description, convert_to_numpy=True).astype(np.float32)

        return SemanticFeatures(
            video_id=video_id,
            description=full_description,
            keyframe_count=len(keyframe_descriptions),
            embedding=embedding,
            keyframe_descriptions=keyframe_descriptions,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()