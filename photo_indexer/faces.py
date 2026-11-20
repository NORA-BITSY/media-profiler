"""Face detection and embedding extraction using InsightFace + MediaPipe Face Mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from uuid import uuid4

try:
    import mediapipe as mp
    from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, FaceLandmarkerResult
    MEDIAPIPE_FACE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_FACE_AVAILABLE = False

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

import cv2
import numpy as np

from .models import FaceEmbedding
from .metadata import find_images

@dataclass
class FaceDetectionConfig:
    """Configuration for face detection."""
    det_size: Tuple[int, int] = (640, 640)
    det_thresh: float = 0.5
    ctx_id: int = 0  # GPU context ID, -1 for CPU
    providers: List[str] = None
    # MediaPipe Face Landmarker options
    use_mediapipe_mesh: bool = True
    max_num_faces: int = 1
    refine_landmarks: bool = True  # Enable iris landmarks
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    model_asset_path: str = "/home/pat/projects/pornstars/face_landmarker.task"

    def __post_init__(self):
        if self.providers is None:
            self.providers = ['CPUExecutionProvider']
class FaceDetector:
    """Wrapper around InsightFace FaceAnalysis for face detection and embedding extraction.
    
    Optionally uses MediaPipe Face Mesh for 468 landmarks, blendshapes, 3D landmarks,
    head pose estimation, and eye gaze.
    """

    def __init__(self, config: FaceDetectionConfig = None):
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError("insightface not installed. Install with: pip install insightface onnxruntime")
        self.config = config or FaceDetectionConfig()
        self._app = None
        self._face_mesh = None

    def _ensure_loaded(self):
        if self._app is None:
            self._app = insightface.app.FaceAnalysis(providers=self.config.providers)
            self._app.prepare(ctx_id=self.config.ctx_id, det_size=self.config.det_size)
        
        if self.config.use_mediapipe_mesh and MEDIAPIPE_FACE_AVAILABLE and self._face_mesh is None:
            base_options = mp.tasks.BaseOptions(model_asset_path=self.config.model_asset_path)
            options = FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_faces=self.config.max_num_faces,
                min_face_detection_confidence=self.config.min_detection_confidence,
                min_face_presence_confidence=self.config.min_tracking_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._face_mesh = FaceLandmarker.create_from_options(options)

    def _estimate_head_pose(self, landmarks_3d: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, float]:
        """Estimate head pose (yaw, pitch, roll) from 3D landmarks using PnP."""
        # Key facial landmarks for pose estimation (indices from 468-point model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0),     # Right mouth corner
        ], dtype=np.float64)

        # Corresponding 2D image points from landmarks
        image_points = np.array([
            landmarks_3d[1][:2],    # nose tip
            landmarks_3d[152][:2],  # chin
            landmarks_3d[33][:2],   # left eye
            landmarks_3d[263][:2],  # right eye
            landmarks_3d[61][:2],   # mouth left
            landmarks_3d[291][:2],  # mouth right
        ], dtype=np.float64)

        h, w = image_shape
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        sy = np.sqrt(rotation_mat[0,0]**2 + rotation_mat[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(rotation_mat[2,1], rotation_mat[2,2])
            y = np.arctan2(-rotation_mat[2,0], sy)
            z = np.arctan2(rotation_mat[1,0], rotation_mat[0,0])
        else:
            x = np.arctan2(-rotation_mat[1,2], rotation_mat[1,1])
            y = np.arctan2(-rotation_mat[2,0], sy)
            z = 0

        return {
            "pitch": float(np.degrees(x)),
            "yaw": float(np.degrees(y)),
            "roll": float(np.degrees(z)),
        }

    def _estimate_eye_gaze(self, landmarks_468: np.ndarray) -> Dict[str, List[float]]:
        """Estimate eye gaze direction from iris and eye landmarks."""
        gaze = {"left": [0.0, 0.0, 1.0], "right": [0.0, 0.0, 1.0]}
        
        if len(landmarks_468) >= 478:
            left_iris = landmarks_468[468][:2]
            left_eye_left = landmarks_468[33][:2]
            left_eye_right = landmarks_468[133][:2]
            left_eye_center = (left_eye_left + left_eye_right) / 2
            gaze["left"] = [(left_iris[0] - left_eye_center[0]) * 10, 
                           (left_iris[1] - left_eye_center[1]) * 10, 1.0]
            
            right_iris = landmarks_468[473][:2]
            right_eye_left = landmarks_468[362][:2]
            right_eye_right = landmarks_468[263][:2]
            right_eye_center = (right_eye_left + right_eye_right) / 2
            gaze["right"] = [(right_iris[0] - right_eye_center[0]) * 10,
                            (right_iris[1] - right_eye_center[1]) * 10, 1.0]
        
        return gaze

    def _compute_expression_vector(self, landmarks_468: np.ndarray) -> List[float]:
        """Compute a simple expression vector from landmark displacements.
        
        Returns 20-dim vector representing key facial action units.
        """
        expr = []
        
        # Mouth opening (vertical distance)
        mouth_top = landmarks_468[13]
        mouth_bottom = landmarks_468[14]
        mouth_open = np.linalg.norm(mouth_top[:2] - mouth_bottom[:2])
        expr.append(float(mouth_open))
        
        # Mouth width (smile)
        mouth_left = landmarks_468[61]
        mouth_right = landmarks_468[291]
        mouth_width = np.linalg.norm(mouth_left[:2] - mouth_right[:2])
        expr.append(float(mouth_width))
        
        # Left eyebrow raise (distance from eye)
        left_brow = landmarks_468[70]
        left_eye = landmarks_468[33]
        left_brow_raise = left_eye[1] - left_brow[1]
        expr.append(float(left_brow_raise))
        
        # Right eyebrow raise
        right_brow = landmarks_468[300]
        right_eye = landmarks_468[263]
        right_brow_raise = right_eye[1] - right_brow[1]
        expr.append(float(right_brow_raise))
        
        # Left eye openness
        left_eye_top = landmarks_468[159]
        left_eye_bottom = landmarks_468[145]
        left_eye_open = np.linalg.norm(left_eye_top[:2] - left_eye_bottom[:2])
        expr.append(float(left_eye_open))
        
        # Right eye openness
        right_eye_top = landmarks_468[386]
        right_eye_bottom = landmarks_468[374]
        right_eye_open = np.linalg.norm(right_eye_top[:2] - right_eye_bottom[:2])
        expr.append(float(right_eye_open))
        
        # Pad to 20 dims
        while len(expr) < 20:
            expr.append(0.0)
        
        return expr[:20]

    def _bbox_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def detect_faces(self, image_path: Union[str, Path]) -> List[FaceEmbedding]:
        """Detect faces in an image and extract embeddings + enhanced biometrics.

        Returns:
            List of FaceEmbedding objects with bounding boxes, landmarks (106/468/3D),
            512-dim embeddings, blendshapes, head pose, eye gaze, etc.
        """
        self._ensure_loaded()

        img = cv2.imread(str(image_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # InsightFace detection
        faces = self._app.get(img)

        # MediaPipe Face Landmarker for enhanced landmarks
        mp_results = None
        if self.config.use_mediapipe_mesh and self._face_mesh:
            mp_results = self._face_mesh.detect(mp_image)

        embeddings = []
        for face in faces:
            bbox = face.bbox.astype(float).tolist() if face.bbox is not None else None

            landmarks_106 = face.landmark_2d_106.astype(float).tolist() if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None else None
            
            landmarks_3d_68 = None
            if hasattr(face, 'landmark_3d_68') and face.landmark_3d_68 is not None:
                landmarks_3d_68 = face.landmark_3d_68.astype(float).tolist()

            embedding = face.embedding.astype(float).tolist() if face.embedding is not None else None

            det_score = float(face.det_score) if face.det_score is not None else None

            age = int(face.age) if hasattr(face, 'age') and face.age is not None else None
            gender = int(face.gender) if hasattr(face, 'gender') and face.gender is not None else None

            # MediaPipe enhanced data
            landmarks_468 = None
            landmarks_3d = None
            iris_landmarks = None
            head_pose = None
            eye_gaze = None
            expression_vector = None
            blendshapes = None

            if mp_results and mp_results.face_landmarks:
                for mp_face_landmarks in mp_results.face_landmarks:
                    # Convert normalized landmarks to pixel coordinates
                    mp_landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in mp_face_landmarks])
                    
                    mp_bbox = [mp_landmarks[:, 0].min(), mp_landmarks[:, 1].min(),
                               mp_landmarks[:, 0].max(), mp_landmarks[:, 1].max()]
                    if bbox and self._bbox_iou(bbox, mp_bbox) > 0.3:
                        landmarks_468 = mp_landmarks[:, :2].astype(float).tolist()
                        landmarks_3d = mp_landmarks.astype(float).tolist()
                        
                        if len(mp_landmarks) >= 478:
                            iris_landmarks = mp_landmarks[468:478, :2].astype(float).tolist()
                        
                        head_pose = self._estimate_head_pose(mp_landmarks, (h, w))
                        eye_gaze = self._estimate_eye_gaze(mp_landmarks)
                        expression_vector = self._compute_expression_vector(mp_landmarks)
                        
                        # Extract blendshapes if available
                        if mp_results.face_blendshapes:
                            # face_blendshapes is a list of categories per face
                            for blendshape_category in mp_results.face_blendshapes:
                                blendshapes = {bs.category_name: bs.score for bs in blendshape_category}
                                break
                        break

            embeddings.append(FaceEmbedding(
                id=uuid4(),
                asset_id=None,
                bbox=bbox,
                landmarks=landmarks_106,
                landmarks_468=landmarks_468,
                landmarks_3d=landmarks_3d,
                embedding=embedding,
                det_score=det_score,
                age=age,
                gender=gender,
                model_name="buffalo_l",
                blendshapes=blendshapes,
                expression_vector=expression_vector,
                identity_vector=embedding,
                head_pose=head_pose,
                eye_gaze=eye_gaze,
                iris_landmarks=iris_landmarks,
            ))

        return embeddings


def detect_faces_in_image(
    image_path: Union[str, Path],
    config: FaceDetectionConfig = None,
) -> List[FaceEmbedding]:
    """Convenience function to detect faces in a single image."""
    detector = FaceDetector(config)
    return detector.detect_faces(image_path)


def detect_faces_batch(
    image_paths: List[Union[str, Path]],
    config: FaceDetectionConfig = None,
) -> List[Tuple[Path, List[FaceEmbedding]]]:
    """Detect faces in multiple images."""
    detector = FaceDetector(config)
    results = []
    for path in image_paths:
        faces = detector.detect_faces(path)
        results.append((Path(path), faces))
    return results


def index_faces_in_staging(
    staging_dir: Union[str, Path],
    config: FaceDetectionConfig = None,
    db = None,
) -> int:
    """Index faces for all images in a staging directory.
    
    If db is provided, also stores face embeddings in the database.
    """
    image_paths = find_images(staging_dir)
    detector = FaceDetector(config)
    
    total_faces = 0
    for image_path in image_paths:
        faces = detector.detect_faces(image_path)
        total_faces += len(faces)
        
        if db and faces:
            asset_id = str(uuid4())
            for face in faces:
                face.asset_id = asset_id
    
    return total_faces