from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import insightface
import numpy as np
from sklearn.cluster import DBSCAN

from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FaceEmbedding:
    embedding_id: str
    video_id: str
    cluster_id: int
    embedding: np.ndarray
    confidence: float
    bbox: tuple[int, int, int, int]  # x, y, w, h
    frame_idx: int
    timestamp: float


@dataclass
class IdentityCluster:
    cluster_id: int
    embeddings: list[FaceEmbedding]
    centroid: np.ndarray
    count: int
    avg_confidence: float


class IdentityExtractor:
    def __init__(self, model_name: str = "buffalo_l", det_size: tuple[int, int] = (640, 640)):
        self.model_name = model_name
        self.det_size = det_size
        self.app = insightface.app.FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=det_size)

    def extract_from_video(
        self,
        video_path: Path,
        video_id: str,
        sample_rate: float = 1.0,
        cluster_eps: float = 0.4,
        min_cluster_size: int = 2,
    ) -> list[IdentityCluster]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("cannot_open_video", path=str(video_path))
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / sample_rate)) if sample_rate > 0 else 1

        all_embeddings: list[FaceEmbedding] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / fps if fps > 0 else 0.0
            faces = self.app.get(frame)

            for face in faces:
                if face.embedding is not None and face.det_score > 0.5:
                    bbox = face.bbox.astype(int)
                    embedding = FaceEmbedding(
                        embedding_id=str(uuid.uuid4())[:16],
                        video_id=video_id,
                        cluster_id=-1,
                        embedding=face.embedding.astype(np.float32),
                        confidence=float(face.det_score),
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])),
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                    )
                    all_embeddings.append(embedding)

            frame_idx += 1

        cap.release()

        if not all_embeddings:
            logger.warning("no_faces_detected", video_id=video_id, path=str(video_path))
            return []

        embeddings_array = np.stack([e.embedding for e in all_embeddings])

        clustering = DBSCAN(eps=cluster_eps, min_samples=min_cluster_size, metric="cosine")
        labels = clustering.fit_predict(embeddings_array)

        clusters: dict[int, list[FaceEmbedding]] = {}
        for i, label in enumerate(labels):
            if label >= 0:
                all_embeddings[i].cluster_id = int(label)
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(all_embeddings[i])

        identity_clusters = []
        for cluster_id, embeddings in clusters.items():
            centroid = np.mean([e.embedding for e in embeddings], axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            avg_conf = float(np.mean([e.confidence for e in embeddings]))
            identity_clusters.append(IdentityCluster(
                cluster_id=cluster_id,
                embeddings=embeddings,
                centroid=centroid.astype(np.float32),
                count=len(embeddings),
                avg_confidence=avg_conf,
            ))

        noise_count = sum(1 for l in labels if l == -1)
        logger.info(
            "identity_extraction_complete",
            video_id=video_id,
            total_faces=len(all_embeddings),
            clusters=len(identity_clusters),
            noise=noise_count,
        )

        return identity_clusters

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()