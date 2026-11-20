from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)


class VectorDatabase:
    def __init__(self, persist_dir: Path | None = None):
        settings = get_settings()
        self.persist_dir = persist_dir or settings.paths.db_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self.face_collection = self.client.get_or_create_collection(
            name="face_embeddings",
            metadata={"hnsw:space": "cosine"},
        )
        self.semantic_collection = self.client.get_or_create_collection(
            name="semantic_embeddings",
            metadata={"hnsw:space": "cosine"},
        )

    def add_face_embedding(
        self,
        embedding: np.ndarray,
        video_id: str,
        cluster_id: int,
        confidence: float,
        frame_idx: int,
        timestamp: float,
        embedding_id: str | None = None,
    ) -> str:
        if embedding_id is None:
            embedding_id = str(uuid.uuid4())[:16]

        self.face_collection.add(
            ids=[embedding_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "video_id": video_id,
                "cluster_id": cluster_id,
                "confidence": confidence,
                "frame_idx": frame_idx,
                "timestamp": timestamp,
            }],
        )
        return embedding_id

    def add_semantic_embedding(
        self,
        embedding: np.ndarray,
        video_id: str,
        description: str,
        keyframe_count: int,
        embedding_id: str | None = None,
    ) -> str:
        if embedding_id is None:
            embedding_id = str(uuid.uuid4())[:16]

        self.semantic_collection.add(
            ids=[embedding_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "video_id": video_id,
                "description": description[:1000],
                "keyframe_count": keyframe_count,
            }],
        )
        return embedding_id

    def add_face_embeddings_batch(
        self,
        embeddings: list[np.ndarray],
        video_ids: list[str],
        cluster_ids: list[int],
        confidences: list[float],
        frame_indices: list[int],
        timestamps: list[float],
        embedding_ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple face embeddings in a single batch operation."""
        if embedding_ids is None:
            embedding_ids = [str(uuid.uuid4())[:16] for _ in embeddings]

        assert len(embeddings) == len(video_ids) == len(cluster_ids) == len(confidences) == len(frame_indices) == len(timestamps) == len(embedding_ids)

        self.face_collection.add(
            ids=embedding_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[{
                "video_id": video_ids[i],
                "cluster_id": cluster_ids[i],
                "confidence": confidences[i],
                "frame_idx": frame_indices[i],
                "timestamp": timestamps[i],
            } for i in range(len(embeddings))],
        )
        return embedding_ids

    def add_semantic_embeddings_batch(
        self,
        embeddings: list[np.ndarray],
        video_ids: list[str],
        descriptions: list[str],
        keyframe_counts: list[int],
        embedding_ids: list[str] | None = None,
    ) -> list[str]:
        """Add multiple semantic embeddings in a single batch operation."""
        if embedding_ids is None:
            embedding_ids = [str(uuid.uuid4())[:16] for _ in embeddings]

        assert len(embeddings) == len(video_ids) == len(descriptions) == len(keyframe_counts) == len(embedding_ids)

        self.semantic_collection.add(
            ids=embedding_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[{
                "video_id": video_ids[i],
                "description": descriptions[i][:1000],
                "keyframe_count": keyframe_counts[i],
            } for i in range(len(embeddings))],
        )
        return embedding_ids

    def search_face_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.6,
    ) -> list[dict[str, Any]]:
        results = self.face_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        matches = []
        if results["ids"] and results["ids"][0]:
            for i, emb_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                if similarity >= threshold:
                    metadata = results["metadatas"][0][i]
                    matches.append({
                        "embedding_id": emb_id,
                        "similarity": similarity,
                        "video_id": metadata["video_id"],
                        "cluster_id": metadata["cluster_id"],
                        "confidence": metadata["confidence"],
                        "frame_idx": metadata["frame_idx"],
                        "timestamp": metadata["timestamp"],
                    })
        return matches

    def search_semantic_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        threshold: float = 0.5,
    ) -> list[dict[str, Any]]:
        results = self.semantic_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        matches = []
        if results["ids"] and results["ids"][0]:
            for i, emb_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                similarity = 1.0 - distance
                if similarity >= threshold:
                    metadata = results["metadatas"][0][i]
                    matches.append({
                        "embedding_id": emb_id,
                        "similarity": similarity,
                        "video_id": metadata["video_id"],
                        "description": metadata["description"],
                        "keyframe_count": metadata["keyframe_count"],
                    })
        return matches

    def get_face_embedding(self, embedding_id: str) -> dict[str, Any] | None:
        results = self.face_collection.get(ids=[embedding_id], include=["embeddings", "metadatas"])
        if results["ids"]:
            return {
                "embedding_id": results["ids"][0],
                "embedding": np.array(results["embeddings"][0]),
                "metadata": results["metadatas"][0],
            }
        return None

    def get_semantic_embedding(self, embedding_id: str) -> dict[str, Any] | None:
        results = self.semantic_collection.get(ids=[embedding_id], include=["embeddings", "metadatas"])
        if results["ids"]:
            return {
                "embedding_id": results["ids"][0],
                "embedding": np.array(results["embeddings"][0]),
                "metadata": results["metadatas"][0],
            }
        return None

    def delete_by_video_id(self, video_id: str) -> None:
        face_results = self.face_collection.get(where={"video_id": video_id})
        if face_results["ids"]:
            self.face_collection.delete(ids=face_results["ids"])

        semantic_results = self.semantic_collection.get(where={"video_id": video_id})
        if semantic_results["ids"]:
            self.semantic_collection.delete(ids=semantic_results["ids"])

    def get_stats(self) -> dict[str, int]:
        return {
            "face_embeddings": self.face_collection.count(),
            "semantic_embeddings": self.semantic_collection.count(),
        }


_vector_db_instance: VectorDatabase | None = None


def get_vector_db() -> VectorDatabase:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDatabase()
    return _vector_db_instance