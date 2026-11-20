"""Performer profile aggregation from multi-image data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np

from .models import (
    BodyMeasurement, BodyPose, FaceEmbedding, FacialBiometrics,
    MeasurementType, PerformerProfile
)
from .database import PhotoIndexDB


@dataclass
class AggregationConfig:
    """Configuration for performer profile aggregation."""
    min_images_for_profile: int = 3
    embedding_aggregation: str = "mean"  # mean, median, weighted_mean
    outlier_std_threshold: float = 2.0
    require_face_match: bool = True
    face_match_threshold: float = 0.6


class PerformerAggregator:
    """Aggregate multi-image data into comprehensive performer profiles."""

    def __init__(self, db: PhotoIndexDB, config: AggregationConfig = None):
        self.db = db
        self.config = config or AggregationConfig()

    def build_performer_profile(
        self,
        name: str,
        asset_ids: List[str],
        aliases: List[str] = None,
        reference_height_cm: float = None
    ) -> PerformerProfile:
        """Build a performer profile from a list of asset IDs."""
        # Collect all data from assets
        face_embeddings = []
        body_poses = []
        all_measurements: Dict[str, List[BodyMeasurement]] = defaultdict(list)
        facial_measurements: Dict[str, List[float]] = defaultdict(list)
        face_qualities = []
        timestamps = []
        source_tags = set()
        for asset_id in asset_ids:
            # Get face embeddings
            faces = self.db.get_faces_for_asset(asset_id)  # Need to add this method
            for face in faces:
                embedding = face.get("embedding_json")
                if embedding:
                    import json
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    face_embeddings.append(np.array(embedding))
                    face_qualities.append(face.get("det_score", 0.5))
            
            # Get body poses
            poses = self.db.get_body_poses(asset_id)
            for pose_data in poses:
                pose = self._pose_from_db(pose_data)
                body_poses.append(pose)
                
                # Compute measurements from this pose
                if pose.landmarks_2d:
                    measurements = compute_measurements_from_pose(pose, reference_height_cm)
                    for m in measurements:
                        all_measurements[m.measurement_type].append(m)
            
            # Get existing measurements
            measurements = self.db.get_body_measurements(asset_id=asset_id)
            for m_data in measurements:
                m = BodyMeasurement(
                    measurement_type=MeasurementType(m_data["measurement_type"]),
                    value=m_data["value"],
                    unit=m_data["unit"],
                    confidence=m_data["confidence"],
                    method=m_data["method"],
                    reference_scale=m_data["reference_scale"],
                    landmarks_used=m_data["landmarks_used_json"],
                    asset_id=UUID(asset_id),
                )
                all_measurements[m.measurement_type].append(m)

            # Get facial measurements if available
            # (would come from face biometrics)

            # Track timestamps
            asset = self.db.get_asset(asset_id)
            if asset and asset.indexed_at:
                timestamps.append(asset.indexed_at)

            # Source tags from asset metadata
            if asset:
                # Extract studio/site from path or metadata
                pass

        if not face_embeddings and self.config.require_face_match:
            raise ValueError(f"No face embeddings found for assets {asset_ids}")

        # Aggregate face embedding
        canonical_embedding = self._aggregate_embeddings(face_embeddings)
        embedding_quality = np.mean(face_qualities) if face_qualities else 0.0

        # Aggregate body measurements
        aggregated_measurements = {}
        measurement_confidence = {}
        for meas_type, measurements in all_measurements.items():
            if len(measurements) >= self.config.min_images_for_profile:
                values = [m.value for m in measurements]
                confidences = [m.confidence for m in measurements]
                
                # Remove outliers
                values_clean, confidences_clean = self._remove_outliers(values, confidences)
                
                if values_clean:
                    aggregated_measurements[meas_type] = {
                        "mean": float(np.mean(values_clean)),
                        "std": float(np.std(values_clean)),
                        "min": float(np.min(values_clean)),
                        "max": float(np.max(values_clean)),
                        "median": float(np.median(values_clean)),
                        "count": len(values_clean),
                    }
                    measurement_confidence[meas_type] = float(np.mean(confidences_clean))

        # Estimate height from measurements
        height_cm = None
        height_confidence = 0.0
        if "height" in aggregated_measurements:
            height_cm = aggregated_measurements["height"]["mean"]
            height_confidence = measurement_confidence.get("height", 0.0)
        elif "leg_length" in aggregated_measurements and "torso_length" in aggregated_measurements:
            # Estimate from components
            height_cm = (aggregated_measurements["leg_length"]["mean"] +
                        aggregated_measurements["torso_length"]["mean"] +
                        aggregated_measurements.get("head_height", {}).get("mean", 25))
            height_confidence = 0.5

        # Classify body type
        body_type = self._classify_body_type(aggregated_measurements)

        # Create performer profile
        profile = PerformerProfile(
            id=uuid4(),
            name=name,
            aliases=aliases or [],
            face_embedding=canonical_embedding.tolist() if canonical_embedding is not None else None,
            face_embedding_quality=embedding_quality,
            measurements=aggregated_measurements,
            measurement_confidence=measurement_confidence,
            body_type=body_type,
            height_cm=height_cm,
            height_confidence=height_confidence,
            asset_count=len(asset_ids),
            first_seen=min(timestamps) if timestamps else None,
            last_seen=max(timestamps) if timestamps else None,
            source_tags=list(source_tags),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        return profile

    def _aggregate_embeddings(self, embeddings: List[np.ndarray]) -> Optional[np.ndarray]:
        """Aggregate multiple face embeddings into a canonical one."""
        if not embeddings:
            return None
        
        embeddings = np.array(embeddings)
        
        if self.config.embedding_aggregation == "mean":
            agg = np.mean(embeddings, axis=0)
        elif self.config.embedding_aggregation == "median":
            agg = np.median(embeddings, axis=0)
        elif self.config.embedding_aggregation == "weighted_mean":
            # Weight by quality (would need quality scores)
            agg = np.mean(embeddings, axis=0)
        else:
            agg = np.mean(embeddings, axis=0)
        
        # Renormalize
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm
        return agg

    def _remove_outliers(
        self, values: List[float], confidences: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Remove outliers using z-score method."""
        if len(values) < 3:
            return values, confidences
        
        values_arr = np.array(values)
        mean = np.mean(values_arr)
        std = np.std(values_arr)
        
        if std == 0:
            return values, confidences
        
        z_scores = np.abs((values_arr - mean) / std)
        mask = z_scores < self.config.outlier_std_threshold
        
        return (values_arr[mask].tolist(), 
                np.array(confidences)[mask].tolist())

    def _classify_body_type(self, measurements: Dict) -> Optional[str]:
        """Classify body type from aggregated measurements."""
        # Simple heuristic based on ratios
        whr = measurements.get("waist_to_hip_ratio", {}).get("mean", 0.7)
        shoulder_hip = 1.0
        if "shoulder_width" in measurements and "hip_width" in measurements:
            shoulder_hip = measurements["shoulder_width"]["mean"] / \
                          measurements["hip_width"]["mean"]
        
        if whr < 0.75 and shoulder_hip > 1.05:
            return "mesomorph"  # Athletic, broad shoulders, narrow waist
        elif whr > 0.85 and shoulder_hip < 1.0:
            return "endomorph"  # Softer, wider waist/hips
        else:
            return "ectomorph"  # Lean, narrow

    def _pose_from_db(self, pose_data: Dict) -> BodyPose:
        """Convert DB row to BodyPose."""
        import json
        return BodyPose(
            landmarks_2d=json.loads(pose_data["landmarks_2d_json"]),
            landmarks_3d=json.loads(pose_data["landmarks_3d_json"]) if pose_data.get("landmarks_3d_json") else None,
            bbox=json.loads(pose_data["bbox_json"]) if pose_data.get("bbox_json") else None,
            det_score=pose_data.get("det_score"),
            model_name=pose_data.get("model_name", "mediapipe_pose"),
        )

    def build_facial_biometrics(
        self,
        performer_id: UUID,
        asset_ids: List[str]
    ) -> FacialBiometrics:
        """Build comprehensive facial biometrics from multiple face images."""
        identity_embeddings = []
        all_landmarks_68 = []
        all_landmarks_468 = []
        all_head_poses = []
        all_expression_vectors = []
        quality_scores = []
        pose_bins = defaultdict(int)
        expression_labels = defaultdict(int)

        for asset_id in asset_ids:
            faces = self.db.get_faces_for_asset(asset_id)
            for face in faces:
                embedding = face.get("embedding_json")
                if embedding:
                    import json
                    if isinstance(embedding, str):
                        embedding = json.loads(embedding)
                    identity_embeddings.append(np.array(embedding))
                    quality_scores.append(face.get("det_score", 0.5))
                
                landmarks_468 = face.get("landmarks_468_json")
                if landmarks_468:
                    import json
                    if isinstance(landmarks_468, str):
                        landmarks_468 = json.loads(landmarks_468)
                    landmarks = np.array(landmarks_468)
                    all_landmarks_468.append(landmarks[:, :2])
                    # Extract 68 key landmarks
                    indices_68 = self._get_68_from_468_indices()
                    all_landmarks_68.append(landmarks[indices_68, :2])
                
                head_pose = face.get("head_pose_json")
                if head_pose:
                    import json
                    if isinstance(head_pose, str):
                        head_pose = json.loads(head_pose)
                    all_head_poses.append([head_pose.get("yaw", 0), head_pose.get("pitch", 0), head_pose.get("roll", 0)])
                    # Bin yaw
                    yaw_bin = int(round(head_pose.get("yaw", 0) / 15) * 15)
                    pose_bins[str(yaw_bin)] += 1
                
                expr_vec = face.get("expression_vector_json")
                if expr_vec:
                    import json
                    if isinstance(expr_vec, str):
                        expr_vec = json.loads(expr_vec)
                    all_expression_vectors.append(expr_vec)

        if not identity_embeddings:
            raise ValueError("No face embeddings found for facial biometrics")

        # Aggregate identity embedding
        identity_embedding = self._aggregate_embeddings(identity_embeddings)
        identity_std = np.std(np.array(identity_embeddings), axis=0).tolist() if len(identity_embeddings) > 1 else None
        quality_mean = float(np.mean(quality_scores)) if quality_scores else 0.0

        # Aggregate landmarks
        canonical_68 = None
        canonical_68_cov = None
        if all_landmarks_68:
            canonical_68 = np.mean(all_landmarks_68, axis=0).tolist()
            # Compute covariance
            stacked = np.stack(all_landmarks_68)
            canonical_68_cov = np.cov(stacked.reshape(len(stacked), -1).T).tolist()

        canonical_468 = None
        if all_landmarks_468:
            canonical_468 = np.mean(all_landmarks_468, axis=0).tolist()

        # Aggregate expressions
        expr_mean = None
        expr_std = None
        if all_expression_vectors:
            expr_arr = np.array(all_expression_vectors)
            expr_mean = np.mean(expr_arr, axis=0).tolist()
            expr_std = np.std(expr_arr, axis=0).tolist()

        # Compute facial measurements from canonical landmarks
        facial_meas = {}
        if canonical_68:
            facial_meas = self._compute_facial_measurements_from_68(canonical_68)

        return FacialBiometrics(
            id=uuid4(),
            performer_id=performer_id,
            identity_embedding=identity_embedding.tolist() if identity_embedding is not None else [],
            identity_embedding_std=identity_std,
            identity_quality_mean=quality_mean,
            canonical_landmarks_68=canonical_68,
            canonical_landmarks_68_cov=canonical_68_cov,
            canonical_landmarks_468=canonical_468,
            facial_measurements=facial_meas,
            source_image_count=len(asset_ids),
            pose_coverage=dict(pose_bins),
            expression_coverage=dict(expression_labels),
            updated_at=datetime.utcnow(),
            model_versions={"insightface": "buffalo_l", "mediapipe": "face_mesh"},
        )

    def _get_68_from_468_indices(self) -> List[int]:
        """Map 468 MediaPipe landmarks to 68 standard landmarks."""
        # Simplified mapping - key indices
        return [
            162, 127, 234, 93, 132, 58, 172, 136, 150, 149,  # Jaw line
            176, 148, 152, 377, 400, 378, 379, 365, 397, 288,  # Jaw line cont.
            361, 323, 454, 356, 389,  # Right side
            70, 63, 105, 66, 107,  # Left eyebrow
            336, 296, 334, 293, 300,  # Right eyebrow
            107, 55, 65, 52, 53, 46,  # Left eye
            336, 285, 295, 282, 283, 276,  # Right eye
            168, 6, 197, 195, 5,  # Nose
            98, 327,  # Nose sides
            61, 84, 17, 314, 405, 321, 375, 291,  # Mouth outer
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,  # Mouth inner
        ][:68]

    def _compute_facial_measurements_from_68(self, landmarks_68: List) -> Dict:
        """Compute facial measurements from 68 landmarks."""
        lm = np.array(landmarks_68)
        meas = {}
        
        def dist(i, j):
            return float(np.linalg.norm(lm[i] - lm[j]))
        
        # Standard facial measurements
        meas["face_height"] = {"mean": dist(8, 19), "std": 0}  # forehead to chin
        meas["face_width"] = {"mean": dist(0, 16), "std": 0}   # cheek to cheek
        meas["interpupillary_dist"] = {"mean": dist(36, 45), "std": 0}
        meas["nose_length"] = {"mean": dist(27, 33), "std": 0}
        meas["nose_width"] = {"mean": dist(31, 35), "std": 0}
        meas["mouth_width"] = {"mean": dist(48, 54), "std": 0}
        
        return meas


def compute_measurements_from_pose(
    pose: BodyPose,
    reference_height_cm: Optional[float] = None
) -> List[BodyMeasurement]:
    """Compute body measurements from detected pose landmarks.
    
    Re-exported from body module for convenience.
    """
    from .body import compute_measurements_from_pose as _compute
    return _compute(pose, reference_height_cm)