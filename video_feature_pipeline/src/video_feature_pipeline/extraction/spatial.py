from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision.core.image import Image as MpImage
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HolisticLandmarker, HolisticLandmarkerOptions, RunningMode

from video_feature_pipeline.core.logging import get_logger

logger = get_logger(__name__)

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "holistic_landmarker.task"


def _ensure_model_downloaded() -> Path:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        logger.info("downloading_mediapipe_model", url=MODEL_URL)
        urlretrieve(MODEL_URL, MODEL_PATH)
        logger.info("model_downloaded", path=str(MODEL_PATH))
    return MODEL_PATH


@dataclass
class SpatialFeatures:
    shoulder_width_mean: float
    shoulder_width_std: float
    torso_height_mean: float
    torso_height_std: float
    limb_lengths: dict[str, float]
    face_landmark_distances: dict[str, float]
    physical_signature: dict[str, float]
    frame_count: int


class SpatialExtractor:
    def __init__(self, complexity: int = 1):
        self.complexity = complexity
        model_path = _ensure_model_downloaded()

        self.holistic = HolisticLandmarker.create_from_options(
            HolisticLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_hand_landmarks_confidence=0.5,
            )
        )

    def _landmark_to_array(self, landmark, frame_shape) -> np.ndarray:
        h, w = frame_shape[:2]
        return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    def _calculate_limb_lengths(self, pose_landmarks) -> dict[str, float]:
        def get(idx):
            lm = pose_landmarks[idx]
            return np.array([lm.x, lm.y, lm.z])

        lengths = {}

        left_shoulder = get(11)
        right_shoulder = get(12)
        left_elbow = get(13)
        right_elbow = get(14)
        left_wrist = get(15)
        right_wrist = get(16)
        left_hip = get(23)
        right_hip = get(24)
        left_knee = get(25)
        right_knee = get(26)
        left_ankle = get(27)
        right_ankle = get(28)

        lengths["shoulder_width"] = self._euclidean_distance(left_shoulder, right_shoulder)
        lengths["left_upper_arm"] = self._euclidean_distance(left_shoulder, left_elbow)
        lengths["right_upper_arm"] = self._euclidean_distance(right_shoulder, right_elbow)
        lengths["left_forearm"] = self._euclidean_distance(left_elbow, left_wrist)
        lengths["right_forearm"] = self._euclidean_distance(right_elbow, right_wrist)
        lengths["left_thigh"] = self._euclidean_distance(left_hip, left_knee)
        lengths["right_thigh"] = self._euclidean_distance(right_hip, right_knee)
        lengths["left_shin"] = self._euclidean_distance(left_knee, left_ankle)
        lengths["right_shin"] = self._euclidean_distance(right_knee, right_ankle)
        mid_shoulder = (left_shoulder + right_shoulder) / 2
        mid_hip = (left_hip + right_hip) / 2
        lengths["torso_height"] = self._euclidean_distance(mid_shoulder, mid_hip)

        # Camera-invariant ratios
        ratios = {}
        # Assign to local variables for ratio computation
        left_upper_arm_val = lengths["left_upper_arm"]
        right_upper_arm_val = lengths["right_upper_arm"]
        left_forearm_val = lengths["left_forearm"]
        right_forearm_val = lengths["right_forearm"]
        left_thigh_val = lengths["left_thigh"]
        right_thigh_val = lengths["right_thigh"]
        left_shin_val = lengths["left_shin"]
        right_shin_val = lengths["right_shin"]
        shoulder_width_val = lengths["shoulder_width"]
        torso_height_val = lengths["torso_height"]

        if left_upper_arm_val > 0:
            ratios["forearm_to_upper_arm_L"] = left_forearm_val / left_upper_arm_val
        if right_upper_arm_val > 0:
            ratios["forearm_to_upper_arm_R"] = right_forearm_val / right_upper_arm_val
        if left_thigh_val > 0:
            ratios["shin_to_thigh_L"] = left_shin_val / left_thigh_val
        if right_thigh_val > 0:
            ratios["shin_to_thigh_R"] = right_shin_val / right_thigh_val
        if shoulder_width_val > 0:
            ratios["torso_to_shoulder"] = torso_height_val / shoulder_width_val
        arm_span = left_upper_arm_val + left_forearm_val + right_upper_arm_val + right_forearm_val
        if torso_height_val > 0:
            ratios["arm_span_to_height"] = arm_span / torso_height_val
        if left_shin_val + left_thigh_val > 0:
            ratios["leg_length_to_height"] = (left_shin_val + left_thigh_val) / torso_height_val

        return lengths, ratios

    def _calculate_face_distances(self, face_landmarks, frame_shape) -> dict[str, float]:
        def get(idx):
            lm = face_landmarks[idx]
            h, w = frame_shape[:2]
            return np.array([lm.x * w, lm.y * h])

        distances = {}

        left_eye_inner = get(468)
        right_eye_inner = get(473)
        distances["inter_ocular"] = self._euclidean_distance(left_eye_inner, right_eye_inner)

        nose_tip = get(1)
        chin = get(152)
        distances["face_height"] = self._euclidean_distance(nose_tip, chin)

        left_ear = get(234)
        right_ear = get(454)
        distances["face_width"] = self._euclidean_distance(left_ear, right_ear)

        left_mouth = get(61)
        right_mouth = get(291)
        distances["mouth_width"] = self._euclidean_distance(left_mouth, right_mouth)

        return distances

    def extract_from_video(self, video_path: Path, sample_rate: float = 1.0) -> SpatialFeatures | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("cannot_open_video", path=str(video_path))
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / sample_rate)) if sample_rate > 0 else 1

        shoulder_widths = []
        torso_heights = []
        limb_lengths_accum: dict[str, list[float]] = {}
        face_distances_accum: dict[str, list[float]] = {}
        ratios_accum: dict[str, list[float]] = {}
        frame_count = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = MpImage(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.holistic.detect(mp_image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                limb_lengths, ratios = self._calculate_limb_lengths(landmarks)

                shoulder_widths.append(limb_lengths["shoulder_width"])
                torso_heights.append(limb_lengths["torso_height"])

                for key, value in limb_lengths.items():
                    if key not in limb_lengths_accum:
                        limb_lengths_accum[key] = []
                    limb_lengths_accum[key].append(value)

                for key, value in ratios.items():
                    if key not in ratios_accum:
                        ratios_accum[key] = []
                    ratios_accum[key].append(value)

            if results.face_landmarks:
                face_landmarks = results.face_landmarks
                face_distances = self._calculate_face_distances(face_landmarks, frame.shape)
                for key, value in face_distances.items():
                    if key not in face_distances_accum:
                        face_distances_accum[key] = []
                    face_distances_accum[key].append(value)

            frame_count += 1
            processed += 1

        cap.release()

        if not shoulder_widths:
            logger.warning("no_pose_detected", path=str(video_path))
            return None

        def stats(values: list[float]) -> tuple[float, float]:
            if not values:
                return 0.0, 0.0
            arr = np.array(values)
            return float(arr.mean()), float(arr.std())

        shoulder_mean, shoulder_std = stats(shoulder_widths)
        torso_mean, torso_std = stats(torso_heights)

        limb_means = {k: float(np.mean(v)) for k, v in limb_lengths_accum.items() if v}
        face_means = {k: float(np.mean(v)) for k, v in face_distances_accum.items() if v}
        ratio_means = {k: float(np.mean(v)) for k, v in ratios_accum.items() if v}

        return SpatialFeatures(
            shoulder_width_mean=shoulder_mean,
            shoulder_width_std=shoulder_std,
            torso_height_mean=torso_mean,
            torso_height_std=torso_std,
            limb_lengths=limb_means,
            face_landmark_distances=face_means,
            physical_signature=ratio_means,
            frame_count=processed,
        )

    def close(self) -> None:
        self.holistic.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()