from __future__ import annotations

from video_feature_pipeline.extraction.identity import FaceEmbedding, IdentityCluster, IdentityExtractor
from video_feature_pipeline.extraction.semantic import SemanticExtractor, SemanticFeatures
from video_feature_pipeline.extraction.spatial import SpatialExtractor, SpatialFeatures

__all__ = [
    "FaceEmbedding",
    "IdentityCluster",
    "IdentityExtractor",
    "SemanticExtractor",
    "SemanticFeatures",
    "SpatialExtractor",
    "SpatialFeatures",
]