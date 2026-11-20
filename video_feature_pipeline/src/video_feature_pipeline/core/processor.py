from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import cv2

from video_feature_pipeline.core.config import get_settings
from video_feature_pipeline.core.database import get_database
from video_feature_pipeline.core.logging import get_logger
from video_feature_pipeline.extraction.identity import IdentityCluster, IdentityExtractor
from video_feature_pipeline.extraction.semantic import SemanticExtractor, SemanticFeatures
from video_feature_pipeline.extraction.spatial import SpatialExtractor, SpatialFeatures
from video_feature_pipeline.storage.graph_db import get_graph_db
from video_feature_pipeline.storage.vector_db import get_vector_db

logger = get_logger(__name__)


class VideoProcessor:
    def __init__(self):
        settings = get_settings()
        self.db = get_database()
        self.vector_db = get_vector_db()
        self.graph_db = get_graph_db()

        self.spatial_extractor = SpatialExtractor(complexity=settings.extraction.mediapipe_complexity)
        self.identity_extractor = IdentityExtractor(
            model_name=settings.extraction.insightface_model,
            det_size=tuple(settings.extraction.insightface_det_size),
        )
        self.semantic_extractor = SemanticExtractor(
            vlm_model=settings.extraction.vlm_model,
            vlm_host=settings.extraction.vlm_host,
            vlm_timeout=settings.extraction.vlm_timeout,
            max_keyframes=settings.extraction.max_keyframes,
        )

    def _get_video_metadata(self, video_path: Path) -> dict[str, Any]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {}

        metadata = {
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0,
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return metadata

    def _save_spatial_to_db(self, video_id: str, features: SpatialFeatures) -> None:
        self.db.upsert_spatial_features(video_id, {
            "shoulder_width_mean": features.shoulder_width_mean,
            "shoulder_width_std": features.shoulder_width_std,
            "torso_height_mean": features.torso_height_mean,
            "torso_height_std": features.torso_height_std,
            "limb_lengths_json": json.dumps(features.limb_lengths),
            "face_landmark_distances_json": json.dumps(features.face_landmark_distances),
            "physical_signature_json": json.dumps(features.physical_signature),
        })

    def _save_identities_to_db(self, video_id: str, clusters: list[IdentityCluster]) -> list[str]:
        embedding_ids = []
        face_records = []  # (embedding_id, frame_idx, timestamp)
        
        for cluster in clusters:
            for embedding in cluster.embeddings:
                emb_id = self.vector_db.add_face_embedding(
                    embedding=embedding.embedding,
                    video_id=video_id,
                    cluster_id=cluster.cluster_id,
                    confidence=embedding.confidence,
                    frame_idx=embedding.frame_idx,
                    timestamp=embedding.timestamp,
                    embedding_id=embedding.embedding_id,
                )
                embedding_ids.append(emb_id)
                face_records.append((embedding.embedding_id, embedding.frame_idx, embedding.timestamp))

                self.db.upsert_identity({
                    "identity_id": embedding.embedding_id,
                    "video_id": video_id,
                    "cluster_id": cluster.cluster_id,
                    "embedding_id": emb_id,
                    "confidence": embedding.confidence,
                })

        # Detect co-occurrences: faces in same frame ± window
        window = 5  # frames
        for i, (id_a, frame_a, ts_a) in enumerate(face_records):
            for id_b, frame_b, ts_b in face_records[i+1:]:
                if abs(frame_a - frame_b) <= window:
                    self.db.add_co_occurrence(id_a, id_b, video_id, min(frame_a, frame_b), (ts_a + ts_b) / 2)
                    # Sync to Neo4j
                    self.graph_db.sync_co_occurrence(id_a, id_b, video_id, min(frame_a, frame_b), (ts_a + ts_b) / 2)

        # Sync identities to Neo4j
        for cluster in clusters:
            for embedding in cluster.embeddings:
                self.graph_db.sync_identity(embedding.embedding_id, video_id, cluster.cluster_id)

        return embedding_ids

    def _save_semantic_to_db(self, video_id: str, features: SemanticFeatures) -> str:
        emb_id = self.vector_db.add_semantic_embedding(
            embedding=features.embedding,
            video_id=video_id,
            description=features.description,
            keyframe_count=features.keyframe_count,
        )
        return emb_id

    def _trigger_harvest(self, clusters: list[IdentityCluster], video_id: str) -> None:
        """Trigger harvest for high-confidence new identities."""
        settings = get_settings()
        min_conf = settings.harvest.min_confidence

        video = self.db.get_video(video_id)
        source_handle = video.get("source") if video else None

        for cluster in clusters:
            if cluster.avg_confidence >= min_conf:
                for embedding in cluster.embeddings:
                    existing = self.db.get_identity_by_embedding_id(embedding.embedding_id)
                    if not existing:
                        logger.info("harvest_triggered",
                                   video_id=video_id,
                                   identity_id=embedding.embedding_id,
                                   confidence=cluster.avg_confidence,
                                   source_handle=source_handle)

                        if source_handle and settings.harvest.socq_api_key:
                            import asyncio
                            asyncio.run(self._harvest_profile(source_handle, embedding.embedding_id))

    async def _harvest_profile(self, source_handle: str, identity_id: str) -> None:
        """Harvest videos from a source profile."""
        from video_feature_pipeline.ingestion.socq_client import get_socq_client
        from video_feature_pipeline.ingestion.downloader import VideoDownloader

        socq = get_socq_client()
        downloader = VideoDownloader()

        try:
            videos = await socq.get_profile_videos(
                profile_id=source_handle,
                platform="generic",  # Would need mapping
                limit=settings.harvest.max_concurrent * 10,
            )

            for video_info in videos[:settings.harvest.max_concurrent]:
                url = video_info.get("url")
                if url:
                    try:
                        downloader.download(url, source=source_handle)
                        logger.info("harvest_downloaded", url=url, identity=identity_id)
                    except Exception as e:
                        logger.error("harvest_download_failed", url=url, error=str(e))

        except Exception as e:
            logger.error("harvest_failed", source_handle=source_handle, error=str(e))
        finally:
            await socq.close()

    def process_video(self, video_path: Path, video_id: str | None = None, source: str = "local") -> dict[str, Any]:
        if video_id is None:
            video_id = video_path.stem

        logger.info("processing_started", video_id=video_id, path=str(video_path))

        metadata = self._get_video_metadata(video_path)
        self.db.upsert_video({
            "video_id": video_id,
            "source": source,
            "local_path": str(video_path),
            "duration": metadata.get("duration"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "status": "processing",
        })

        settings = get_settings()

        # Determine frame rate: adaptive or static
        frame_rate = settings.extraction.frame_sample_rate
        if settings.extraction.adaptive_frame_rate:
            frame_rate = settings.extraction.base_frame_rate

        spatial_features = self.spatial_extractor.extract_from_video(
            video_path,
            sample_rate=frame_rate,
        )
        if spatial_features:
            self._save_spatial_to_db(video_id, spatial_features)
            logger.info("spatial_extraction_done", video_id=video_id)

        # First pass identity extraction at base rate
        identity_clusters = self.identity_extractor.extract_from_video(
            video_path,
            video_id=video_id,
            sample_rate=frame_rate,
        )
        
        # Adaptive second pass: re-analyze high-confidence segments at higher rate
        if (settings.extraction.adaptive_frame_rate and identity_clusters 
            and settings.extraction.high_confidence_frame_rate > frame_rate):
            high_conf_clusters = [
                c for c in identity_clusters 
                if c.avg_confidence >= settings.extraction.identity_confidence_threshold
            ]
            if high_conf_clusters:
                logger.info("adaptive_rate_triggered", 
                           video_id=video_id,
                           high_conf_clusters=len(high_conf_clusters),
                           rate=settings.extraction.high_confidence_frame_rate)
                # Re-extract at higher rate for detailed analysis
                detailed_clusters = self.identity_extractor.extract_from_video(
                    video_path,
                    video_id=video_id,
                    sample_rate=settings.extraction.high_confidence_frame_rate,
                )
                if detailed_clusters:
                    identity_clusters = detailed_clusters

        if identity_clusters:
            self._save_identities_to_db(video_id, identity_clusters)
            logger.info("identity_extraction_done", video_id=video_id, clusters=len(identity_clusters))

            # Harvest trigger: check for high-confidence new identities
            if settings.harvest.enabled and settings.harvest.auto_scrape_on_new_identity:
                self._trigger_harvest(identity_clusters, video_id)

        semantic_features = asyncio.run(self.semantic_extractor.extract_from_video(
            video_path,
            video_id=video_id,
            keyframe_rate=settings.extraction.keyframe_rate,
        ))
        if semantic_features:
            self._save_semantic_to_db(video_id, semantic_features)
            logger.info("semantic_extraction_done", video_id=video_id)

        self.db.update_video_status(video_id, "done")
        logger.info("processing_complete", video_id=video_id)

        return {
            "video_id": video_id,
            "status": "done",
            "spatial": spatial_features is not None,
            "identities": len(identity_clusters) if identity_clusters else 0,
            "semantic": semantic_features is not None,
        }

    async def process_video_async(self, video_path: Path, video_id: str | None = None, source: str = "local") -> dict[str, Any]:
        if video_id is None:
            video_id = video_path.stem

        logger.info("processing_started", video_id=video_id, path=str(video_path))

        metadata = self._get_video_metadata(video_path)
        self.db.upsert_video({
            "video_id": video_id,
            "source": source,
            "local_path": str(video_path),
            "duration": metadata.get("duration"),
            "fps": metadata.get("fps"),
            "width": metadata.get("width"),
            "height": metadata.get("height"),
            "status": "processing",
        })

        try:
            settings = get_settings()

            # Determine frame rate: adaptive or static
            frame_rate = settings.extraction.frame_sample_rate
            if settings.extraction.adaptive_frame_rate:
                frame_rate = settings.extraction.base_frame_rate

            spatial_features = self.spatial_extractor.extract_from_video(
                video_path,
                sample_rate=frame_rate,
            )
            if spatial_features:
                self._save_spatial_to_db(video_id, spatial_features)

            # First pass identity extraction at base rate
            identity_clusters = self.identity_extractor.extract_from_video(
                video_path,
                video_id=video_id,
                sample_rate=frame_rate,
            )
            
            # Adaptive second pass: re-analyze high-confidence segments at higher rate
            if (settings.extraction.adaptive_frame_rate and identity_clusters 
                and settings.extraction.high_confidence_frame_rate > frame_rate):
                high_conf_clusters = [
                    c for c in identity_clusters 
                    if c.avg_confidence >= settings.extraction.identity_confidence_threshold
                ]
                if high_conf_clusters:
                    logger.info("adaptive_rate_triggered", 
                               video_id=video_id,
                               high_conf_clusters=len(high_conf_clusters),
                               rate=settings.extraction.high_confidence_frame_rate)
                    # Re-extract at higher rate for detailed analysis
                    detailed_clusters = self.identity_extractor.extract_from_video(
                        video_path,
                        video_id=video_id,
                        sample_rate=settings.extraction.high_confidence_frame_rate,
                    )
                    if detailed_clusters:
                        identity_clusters = detailed_clusters

            if identity_clusters:
                self._save_identities_to_db(video_id, identity_clusters)

            semantic_features = await self.semantic_extractor.extract_from_video(
                video_path,
                video_id=video_id,
                keyframe_rate=settings.extraction.keyframe_rate,
            )
            if semantic_features:
                self._save_semantic_to_db(video_id, semantic_features)

            self.db.update_video_status(video_id, "done")
            logger.info("processing_complete", video_id=video_id)

            return {
                "video_id": video_id,
                "status": "done",
                "spatial": spatial_features is not None,
                "identities": len(identity_clusters) if identity_clusters else 0,
                "semantic": semantic_features is not None,
            }

        except Exception as e:
            self.db.update_video_status(video_id, "failed")
            logger.error("processing_failed", video_id=video_id, error=str(e))
            raise

    def close(self) -> None:
        self.spatial_extractor.close()
        self.identity_extractor.close()
        asyncio.run(self.semantic_extractor.close())
        self.graph_db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()