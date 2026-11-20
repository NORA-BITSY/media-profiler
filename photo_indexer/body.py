"""Body pose detection and measurement extraction using MediaPipe Pose (tasks API)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmarkerResult
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from .models import BodyLandmarkSet, BodyPose, BodyMeasurement, MeasurementType
# MediaPipe Pose landmark indices (33 landmarks)
MP_POSE_LANDMARKS = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


# Measurement landmark pairs (indices) for 2D landmark-based measurements
MEASUREMENT_LANDMARKS = {
    MeasurementType.SHOULDER_WIDTH: (11, 12),  # left_shoulder, right_shoulder
    MeasurementType.HIP_WIDTH: (23, 24),  # left_hip, right_hip
    MeasurementType.TORSO_LENGTH: (11, 23),  # left_shoulder to left_hip (approx)
    MeasurementType.LEG_LENGTH: (23, 27),  # left_hip to left_ankle
    MeasurementType.ARM_LENGTH: (11, 15),  # left_shoulder to left_wrist
    MeasurementType.NECK_LENGTH: (0, 11),  # nose to left_shoulder (approx)
    MeasurementType.HEAD_HEIGHT: (0, 10),  # nose to mouth_right (rough approx)
    MeasurementType.INSEAM: (24, 28),  # right_hip to right_ankle
    MeasurementType.OUTSEAM: (12, 16),  # right_shoulder to right_wrist (arm span proxy)
}


@dataclass
class BodyDetectorConfig:
    """Configuration for body pose detection."""
    model_complexity: int = 1  # 0=light, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False
    static_image_mode: bool = True
    model_asset_path: str = "pose_landmarker_heavy.task"  # Will be downloaded if not found


class BodyDetector:
    """Wrapper around MediaPipe PoseLandmarker for body pose detection and landmark extraction."""

    def __init__(self, config: BodyDetectorConfig = None):
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe")
        self.config = config or BodyDetectorConfig()
        self._landmarker = None

    def _ensure_loaded(self):
        if self._landmarker is None:
            # Determine model path - let MediaPipe auto-download if not found
            model_path = self.config.model_asset_path
            import os
            if not os.path.exists(model_path):
                # MediaPipe will auto-download the model if we use the standard name
                model_path = "pose_landmarker_heavy.task"
            
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE if self.config.static_image_mode else mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=self.config.min_detection_confidence,
                min_pose_presence_confidence=self.config.min_tracking_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_segmentation_masks=self.config.enable_segmentation,
            )
            self._landmarker = PoseLandmarker.create_from_options(options)

    def detect_pose(self, image_path: Union[str, Path]) -> Optional[BodyPose]:
        """Detect body pose in an image and extract landmarks."""
        self._ensure_loaded()

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        result: PoseLandmarkerResult = self._landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Get first detected pose
        landmarks = result.pose_landmarks[0]
        
        # Extract 2D landmarks (normalized 0-1) with visibility
        landmarks_2d = []
        for lm in landmarks:
            landmarks_2d.append([lm.x, lm.y, lm.visibility])

        # Extract 3D world landmarks if available
        landmarks_3d = None
        if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
            world_landmarks = result.pose_world_landmarks[0]
            landmarks_3d = []
            for lm in world_landmarks:
                landmarks_3d.append([lm.x, lm.y, lm.z])

        # Compute bounding box from landmarks
        xs = [lm[0] for lm in landmarks_2d if lm[2] > 0.5]
        ys = [lm[1] for lm in landmarks_2d if lm[2] > 0.5]
        bbox = None
        if xs and ys:
            bbox = [min(xs), min(ys), max(xs), max(ys)]

        # Average visibility as detection score
        visibilities = [lm[2] for lm in landmarks_2d]
        det_score = float(np.mean(visibilities)) if visibilities else 0.0

        return BodyPose(
            id=uuid4(),
            asset_id=None,  # Set by caller
            landmark_set=BodyLandmarkSet.MEDIAPIPE_POSE,
            landmarks_2d=landmarks_2d,
            landmarks_3d=landmarks_3d,
            bbox=bbox,
            det_score=det_score,
            model_name="mediapipe_pose_landmarker",
            detected_at=None,
        )

    def detect_pose_batch(
        self, image_paths: List[Union[str, Path]]
    ) -> List[Tuple[Path, Optional[BodyPose]]]:
        """Detect poses in multiple images."""
        results = []
        for path in image_paths:
            pose = self.detect_pose(path)
            results.append((Path(path), pose))
        return results


def detect_pose_in_image(
    image_path: Union[str, Path],
    config: BodyDetectorConfig = None
) -> Optional[BodyPose]:
    """Convenience function to detect pose in a single image."""
    detector = BodyDetector(config)
    return detector.detect_pose(image_path)


def compute_measurements_from_pose(
    pose: BodyPose,
    reference_height_cm: Optional[float] = None
) -> List[BodyMeasurement]:
    """Compute body measurements from detected pose landmarks.
    
    Uses 2D landmark distances with optional absolute scaling from reference height.
    """
    if not pose.landmarks_2d or len(pose.landmarks_2d) < 33:
        return []

    landmarks = pose.landmarks_2d
    measurements = []

    # Helper to get landmark coordinates
    def get_lm(idx: int) -> Optional[Tuple[float, float]]:
        if idx < len(landmarks) and landmarks[idx][2] > 0.3:  # visibility threshold
            return (landmarks[idx][0], landmarks[idx][1])
        return None

    def euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Compute scale factor if reference height provided
    scale_factor = 1.0
    if reference_height_cm:
        # Estimate height from landmarks (head to ankle)
        head = get_lm(MP_POSE_LANDMARKS["nose"])
        left_ankle = get_lm(MP_POSE_LANDMARKS["left_ankle"])
        right_ankle = get_lm(MP_POSE_LANDMARKS["right_ankle"])
        if head and (left_ankle or right_ankle):
            ankle = left_ankle if left_ankle else right_ankle
            pixel_height = euclidean_dist(head, ankle)
            if pixel_height > 0:
                scale_factor = reference_height_cm / pixel_height

    # Compute each measurement
    for meas_type, (idx1, idx2) in MEASUREMENT_LANDMARKS.items():
        p1 = get_lm(idx1)
        p2 = get_lm(idx2)
        if p1 and p2:
            dist_normalized = euclidean_dist(p1, p2)
            value_cm = dist_normalized * scale_factor  # normalized coords * scale = cm
            
            measurements.append(BodyMeasurement(
                id=uuid4(),
                asset_id=pose.asset_id,
                measurement_type=meas_type,
                value=round(value_cm, 1),
                unit="cm",
                confidence=min(landmarks[idx1][2], landmarks[idx2][2]),
                method="landmark_2d",
                reference_scale=reference_height_cm,
                landmarks_used=[idx1, idx2],
                computed_at=None,
            ))

    # Compute ratios
    # Waist-to-hip ratio (need waist landmarks - approximate from hip and shoulder)
    left_hip = get_lm(MP_POSE_LANDMARKS["left_hip"])
    right_hip = get_lm(MP_POSE_LANDMARKS["right_hip"])
    left_shoulder = get_lm(MP_POSE_LANDMARKS["left_shoulder"])
    right_shoulder = get_lm(MP_POSE_LANDMARKS["right_shoulder"])
    
    if left_hip and right_hip:
        hip_width = euclidean_dist(left_hip, right_hip) * scale_factor
        # Waist approx at midpoint between hip and shoulder
        if left_shoulder and right_shoulder:
            shoulder_width = euclidean_dist(left_shoulder, right_shoulder) * scale_factor
            waist_est = (hip_width + shoulder_width) / 2 * 0.8  # rough estimate
            if hip_width > 0:
                measurements.append(BodyMeasurement(
                    id=uuid4(),
                    asset_id=pose.asset_id,
                    measurement_type=MeasurementType.WHR,
                    value=round(waist_est / hip_width, 3),
                    unit="ratio",
                    confidence=0.4,
                    method="landmark_2d_estimate",
                    landmarks_used=[23, 24, 11, 12],
                    computed_at=None,
                ))

    # Leg-to-body ratio
    if left_hip and get_lm(MP_POSE_LANDMARKS["left_ankle"]):
        leg_len = euclidean_dist(left_hip, get_lm(MP_POSE_LANDMARKS["left_ankle"])) * scale_factor
        if left_shoulder:
            torso_len = euclidean_dist(left_shoulder, left_hip) * scale_factor
            if torso_len > 0:
                measurements.append(BodyMeasurement(
                    id=uuid4(),
                    asset_id=pose.asset_id,
                    measurement_type=MeasurementType.LEG_TO_BODY,
                    value=round(leg_len / torso_len, 3),
                    unit="ratio",
                    confidence=0.5,
                    method="landmark_2d_estimate",
                    landmarks_used=[23, 27, 11],
                    computed_at=None,
                ))

    return measurements


# MediaPipe Face Mesh landmark indices for facial measurements (when using MediaPipe Face Landmarker)
MP_FACE_LANDMARKS = {
    # 468 landmarks - key indices for measurements
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "nose_bottom": 2,
    "nose_left": 98,
    "nose_right": 327,
    "mouth_left": 61,
    "mouth_right": 291,
    "mouth_top": 13,
    "mouth_bottom": 14,
    "chin": 152,
    "forehead": 10,
    "left_eyebrow_inner": 70,
    "right_eyebrow_inner": 300,
    "left_ear": 234,
    "right_ear": 454,
    "jaw_left": 172,
    "jaw_right": 397,
}


def compute_facial_measurements_from_landmarks(
    landmarks_468: List[List[float]],
    reference_interpupillary_mm: Optional[float] = 64.0  # average IPD in mm
) -> List[BodyMeasurement]:
    """Compute facial measurements from MediaPipe 468 face landmarks."""
    if not landmarks_468 or len(landmarks_468) < 468:
        return []

    lm = landmarks_468
    measurements = []

    def get_pt(idx: int) -> np.ndarray:
        return np.array([lm[idx][0], lm[idx][1]])

    def dist(idx1: int, idx2: int) -> float:
        return float(np.linalg.norm(get_pt(idx1) - get_pt(idx2)))

    # Compute scale from interpupillary distance if reference provided
    scale = 1.0
    if reference_interpupillary_mm:
        left_eye_center = (get_pt(33) + get_pt(133)) / 2
        right_eye_center = (get_pt(362) + get_pt(263)) / 2
        ipd_pixels = float(np.linalg.norm(left_eye_center - right_eye_center))
        if ipd_pixels > 0:
            scale = reference_interpupillary_mm / ipd_pixels

    # Facial measurements
    meas_defs = [
        (MeasurementType.FACE_HEIGHT, MP_FACE_LANDMARKS["forehead"], MP_FACE_LANDMARKS["chin"]),
        (MeasurementType.FACE_WIDTH, MP_FACE_LANDMARKS["left_ear"], MP_FACE_LANDMARKS["right_ear"]),
        (MeasurementType.INTERPUPILLARY_DIST, MP_FACE_LANDMARKS["left_eye_inner"], MP_FACE_LANDMARKS["right_eye_inner"]),
        (MeasurementType.NOSE_LENGTH, MP_FACE_LANDMARKS["nose_tip"], MP_FACE_LANDMARKS["nose_bottom"]),
        (MeasurementType.NOSE_WIDTH, MP_FACE_LANDMARKS["nose_left"], MP_FACE_LANDMARKS["nose_right"]),
        (MeasurementType.MOUTH_WIDTH, MP_FACE_LANDMARKS["mouth_left"], MP_FACE_LANDMARKS["mouth_right"]),
        (MeasurementType.EYE_WIDTH, MP_FACE_LANDMARKS["left_eye_outer"], MP_FACE_LANDMARKS["left_eye_inner"]),
        (MeasurementType.EYEBROW_TO_EYE, MP_FACE_LANDMARKS["left_eyebrow_inner"], MP_FACE_LANDMARKS["left_eye_inner"]),
        (MeasurementType.CHIN_TO_MOUTH, MP_FACE_LANDMARKS["chin"], MP_FACE_LANDMARKS["mouth_bottom"]),
        (MeasurementType.FOREHEAD_HEIGHT, MP_FACE_LANDMARKS["forehead"], MP_FACE_LANDMARKS["left_eyebrow_inner"]),
        (MeasurementType.JAW_WIDTH, MP_FACE_LANDMARKS["jaw_left"], MP_FACE_LANDMARKS["jaw_right"]),
    ]

    for meas_type, idx1, idx2 in meas_defs:
        value_mm = dist(idx1, idx2) * scale
        measurements.append(BodyMeasurement(
            id=uuid4(),
            measurement_type=meas_type,
            value=round(value_mm, 1),
            unit="mm",
            confidence=0.7,
            method="landmark_468",
            reference_scale=reference_interpupillary_mm,
            landmarks_used=[idx1, idx2],
            computed_at=None,
        ))

    # Face aspect ratio
    face_height = dist(MP_FACE_LANDMARKS["forehead"], MP_FACE_LANDMARKS["chin"]) * scale
    face_width = dist(MP_FACE_LANDMARKS["left_ear"], MP_FACE_LANDMARKS["right_ear"]) * scale
    if face_width > 0:
        measurements.append(BodyMeasurement(
            id=uuid4(),
            measurement_type=MeasurementType.FACE_ASPECT_RATIO,
            value=round(face_height / face_width, 3),
            unit="ratio",
            confidence=0.7,
            method="landmark_468",
            computed_at=None,
        ))
        measurements.append(BodyMeasurement(
            id=uuid4(),
            measurement_type=MeasurementType.FACIAL_INDEX,
            value=round(face_height / face_width * 100, 1),
            unit="index",
            confidence=0.7,
            method="landmark_468",
            computed_at=None,
        ))

    return measurements