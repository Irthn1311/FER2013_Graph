"""MediaPipe-derived pixel priors for D16.

The output is deliberately numeric-array only per sample. Human-readable names
live in global JSON files written by the precompute script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.request import urlretrieve

import numpy as np


PART_NAMES = [
    "left_eye",
    "right_eye",
    "left_brow",
    "right_brow",
    "nose",
    "mouth",
    "left_mouth_corner",
    "right_mouth_corner",
    "left_cheek",
    "right_cheek",
    "chin",
    "face_contour",
    "outside_face",
]

MICRO_ANCHOR_NAMES = [
    "left_mouth_corner",
    "right_mouth_corner",
    "upper_lip",
    "lower_lip",
    "left_eye_inner",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye_outer",
    "left_brow_inner",
    "right_brow_inner",
    "nose_tip",
    "nose_bridge",
]

FALLBACK_TYPES = {"detected": 0, "center_face": 1, "uniform": 2}
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
DEFAULT_FACE_LANDMARKER_MODEL_PATH = Path("outputs") / "d16_mediapipe_pixel_priors" / "assets" / "face_landmarker.task"

PART_LANDMARK_GROUPS = {
    "left_eye": [33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163],
    "right_eye": [263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390],
    "left_brow": [46, 52, 53, 55, 65, 66, 70, 105, 107],
    "right_brow": [276, 282, 283, 285, 295, 296, 300, 334, 336],
    "nose": [1, 2, 4, 5, 6, 19, 94, 98, 168, 195, 197, 327],
    "mouth": [13, 14, 17, 37, 39, 40, 61, 78, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415],
    "left_mouth_corner": [61],
    "right_mouth_corner": [291],
    "left_cheek": [50, 101, 117, 118, 123, 187, 205],
    "right_cheek": [280, 330, 347, 348, 352, 411, 425],
    "chin": [136, 148, 149, 150, 152, 172, 176, 288, 361, 365, 377, 378, 379, 397, 400],
    "face_contour": [10, 21, 54, 58, 67, 93, 103, 109, 127, 132, 136, 148, 149, 150, 152, 162, 172, 176, 234, 251, 284, 288, 297, 323, 332, 338, 356, 361, 365, 377, 378, 379, 389, 397, 454],
}

MICRO_ANCHOR_LANDMARKS = {
    "left_mouth_corner": [61],
    "right_mouth_corner": [291],
    "upper_lip": [13],
    "lower_lip": [14],
    "left_eye_inner": [133],
    "left_eye_outer": [33],
    "right_eye_inner": [362],
    "right_eye_outer": [263],
    "left_brow_inner": [107],
    "right_brow_inner": [336],
    "nose_tip": [1],
    "nose_bridge": [168],
}


@dataclass
class D16PriorResult:
    arrays: Dict[str, np.ndarray]
    detected: bool
    fallback_type: str
    quality_score: float


def pixels_to_image48(pixels: str | Iterable[int] | np.ndarray) -> np.ndarray:
    if isinstance(pixels, str):
        arr = np.fromstring(pixels, sep=" ", dtype=np.float32)
    else:
        arr = np.asarray(list(pixels) if not isinstance(pixels, np.ndarray) else pixels, dtype=np.float32)
    if arr.size != 48 * 48:
        raise ValueError(f"Expected 2304 FER pixels, got {arr.size}")
    return arr.reshape(48, 48)


def image48_to_rgb_uint8(image48: np.ndarray, detection_size: int = 192) -> np.ndarray:
    image = np.asarray(image48, dtype=np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    if int(detection_size) != 48:
        yy = np.linspace(0, 47, int(detection_size)).round().astype(np.int64)
        xx = np.linspace(0, 47, int(detection_size)).round().astype(np.int64)
        image = image[np.ix_(yy, xx)]
    return np.repeat(image[..., None], 3, axis=2)


class MediaPipeFaceDetector:
    def __init__(
        self,
        detection_size: int = 192,
        min_detection_confidence: float = 0.5,
        model_path: str | Path = DEFAULT_FACE_LANDMARKER_MODEL_PATH,
    ) -> None:
        self.detection_size = int(detection_size)
        self.min_detection_confidence = float(min_detection_confidence)
        self.face_mesh = None
        self.landmarker = None
        self.mp = None
        self.available = False
        try:
            import mediapipe as mp  # type: ignore

            self.mp = mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=self.min_detection_confidence,
                )
            else:
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                model_path = Path(model_path)
                model_path.parent.mkdir(parents=True, exist_ok=True)
                if not model_path.exists():
                    print(f"[D16 MediaPipe] downloading FaceLandmarker model: {FACE_LANDMARKER_MODEL_URL}")
                    urlretrieve(FACE_LANDMARKER_MODEL_URL, model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=vision.RunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_detection_confidence,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
            self.available = True
        except Exception as exc:
            print(f"[D16 MediaPipe] unavailable, using fallback priors only: {exc}")

    def detect(self, image48: np.ndarray) -> Optional[np.ndarray]:
        if not self.available:
            return None
        rgb = image48_to_rgb_uint8(image48, self.detection_size)
        if self.face_mesh is not None:
            result = self.face_mesh.process(rgb)
            if not getattr(result, "multi_face_landmarks", None):
                return None
            landmarks = result.multi_face_landmarks[0].landmark
        else:
            if self.landmarker is None or self.mp is None:
                return None
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = self.landmarker.detect(mp_image)
            if not getattr(result, "face_landmarks", None):
                return None
            landmarks = result.face_landmarks[0]
        xy = np.asarray([[float(lm.x) * 47.0, float(lm.y) * 47.0] for lm in landmarks], dtype=np.float32)
        return np.clip(xy, 0.0, 47.0)

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.landmarker is not None:
            self.landmarker.close()


def _grid48() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.meshgrid(np.arange(48, dtype=np.float32), np.arange(48, dtype=np.float32), indexing="ij")
    return yy, xx


def gaussian_map(center_xy: np.ndarray, sigma: float) -> np.ndarray:
    yy, xx = _grid48()
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    out = np.exp(-dist2 / (2.0 * float(sigma) ** 2))
    return (out / max(float(out.max()), 1e-8)).astype(np.float32)


def distance_map(center_xy: np.ndarray) -> np.ndarray:
    yy, xx = _grid48()
    cx, cy = float(center_xy[0]), float(center_xy[1])
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / np.sqrt(47.0 ** 2 + 47.0 ** 2)
    return np.clip(dist, 0.0, 1.0).astype(np.float32)


def center_face_mask() -> np.ndarray:
    yy, xx = _grid48()
    cx, cy = 23.5, 24.0
    rx, ry = 17.5, 21.0
    value = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.exp(-np.maximum(value - 1.0, 0.0) * 4.0)
    mask[value <= 1.0] = 1.0
    return mask.astype(np.float32)


def _valid_center(landmark_xy: np.ndarray, indices: List[int]) -> tuple[bool, np.ndarray]:
    valid = [idx for idx in indices if 0 <= idx < len(landmark_xy)]
    if not valid:
        return False, np.zeros(2, dtype=np.float32)
    pts = landmark_xy[np.asarray(valid, dtype=np.int64)]
    return True, pts.mean(axis=0).astype(np.float32)


def _face_mask_from_landmarks(landmark_xy: np.ndarray) -> np.ndarray:
    if len(landmark_xy) == 0:
        return center_face_mask()
    x_min, y_min = landmark_xy.min(axis=0)
    x_max, y_max = landmark_xy.max(axis=0)
    cx = float((x_min + x_max) / 2.0)
    cy = float((y_min + y_max) / 2.0)
    rx = max(float((x_max - x_min) / 2.0) * 1.05, 10.0)
    ry = max(float((y_max - y_min) / 2.0) * 1.10, 12.0)
    yy, xx = _grid48()
    value = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.exp(-np.maximum(value - 1.0, 0.0) * 5.0)
    mask[value <= 1.0] = 1.0
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


def fallback_priors(image48: np.ndarray, label: int, sample_index: int, fallback_type: str = "center_face") -> D16PriorResult:
    face = center_face_mask()
    part_masks = np.zeros((len(PART_NAMES), 48, 48), dtype=np.float32)
    outside_idx = PART_NAMES.index("outside_face")
    part_masks[outside_idx] = 1.0 - face
    anchor_maps = np.zeros((len(MICRO_ANCHOR_NAMES), 48, 48), dtype=np.float32)
    dist_maps = np.ones((len(MICRO_ANCHOR_NAMES), 48, 48), dtype=np.float32)
    valid_part_mask = np.zeros((len(PART_NAMES),), dtype=np.float32)
    valid_part_mask[outside_idx] = 1.0
    arrays = {
        "image_48": np.asarray(image48, dtype=np.float32),
        "face_mask": face,
        "part_soft_masks": part_masks,
        "micro_anchor_maps": anchor_maps,
        "distance_maps": dist_maps,
        "landmark_xy_48": np.zeros((0, 2), dtype=np.float32),
        "detected": np.asarray(False, dtype=np.bool_),
        "fallback_type_id": np.asarray(FALLBACK_TYPES.get(fallback_type, FALLBACK_TYPES["center_face"]), dtype=np.int64),
        "landmark_missing_flag": np.asarray(1, dtype=np.int64),
        "valid_part_mask": valid_part_mask,
        "valid_anchor_mask": np.zeros((len(MICRO_ANCHOR_NAMES),), dtype=np.float32),
        "quality_score": np.asarray(0.0, dtype=np.float32),
        "label": np.asarray(int(label), dtype=np.int64),
        "sample_index": np.asarray(int(sample_index), dtype=np.int64),
    }
    return D16PriorResult(arrays=arrays, detected=False, fallback_type=fallback_type, quality_score=0.0)


def detected_priors(image48: np.ndarray, landmark_xy: np.ndarray, label: int, sample_index: int) -> D16PriorResult:
    face = _face_mask_from_landmarks(landmark_xy)
    part_masks = np.zeros((len(PART_NAMES), 48, 48), dtype=np.float32)
    valid_part_mask = np.zeros((len(PART_NAMES),), dtype=np.float32)
    for part_idx, name in enumerate(PART_NAMES):
        if name == "outside_face":
            part_masks[part_idx] = 1.0 - face
            valid_part_mask[part_idx] = 1.0
            continue
        ok, center = _valid_center(landmark_xy, PART_LANDMARK_GROUPS.get(name, []))
        if ok:
            sigma = 2.5 if name in ("left_mouth_corner", "right_mouth_corner") else 4.0
            if name in ("face_contour", "chin"):
                sigma = 5.0
            part_masks[part_idx] = gaussian_map(center, sigma=sigma) * face
            valid_part_mask[part_idx] = 1.0
    anchor_maps = np.zeros((len(MICRO_ANCHOR_NAMES), 48, 48), dtype=np.float32)
    dist_maps = np.ones((len(MICRO_ANCHOR_NAMES), 48, 48), dtype=np.float32)
    valid_anchor_mask = np.zeros((len(MICRO_ANCHOR_NAMES),), dtype=np.float32)
    for anchor_idx, name in enumerate(MICRO_ANCHOR_NAMES):
        ok, center = _valid_center(landmark_xy, MICRO_ANCHOR_LANDMARKS[name])
        if ok:
            anchor_maps[anchor_idx] = gaussian_map(center, sigma=2.0)
            dist_maps[anchor_idx] = distance_map(center)
            valid_anchor_mask[anchor_idx] = 1.0
    quality = float(np.mean((landmark_xy[:, 0] >= 0) & (landmark_xy[:, 0] <= 47) & (landmark_xy[:, 1] >= 0) & (landmark_xy[:, 1] <= 47)))
    arrays = {
        "image_48": np.asarray(image48, dtype=np.float32),
        "face_mask": face.astype(np.float32),
        "part_soft_masks": part_masks.astype(np.float32),
        "micro_anchor_maps": anchor_maps.astype(np.float32),
        "distance_maps": dist_maps.astype(np.float32),
        "landmark_xy_48": np.asarray(landmark_xy, dtype=np.float32),
        "detected": np.asarray(True, dtype=np.bool_),
        "fallback_type_id": np.asarray(FALLBACK_TYPES["detected"], dtype=np.int64),
        "landmark_missing_flag": np.asarray(0, dtype=np.int64),
        "valid_part_mask": valid_part_mask.astype(np.float32),
        "valid_anchor_mask": valid_anchor_mask.astype(np.float32),
        "quality_score": np.asarray(quality, dtype=np.float32),
        "label": np.asarray(int(label), dtype=np.int64),
        "sample_index": np.asarray(int(sample_index), dtype=np.int64),
    }
    return D16PriorResult(arrays=arrays, detected=True, fallback_type="detected", quality_score=quality)


def build_priors_for_sample(detector: MediaPipeFaceDetector, pixels: str, label: int, sample_index: int) -> D16PriorResult:
    image48 = pixels_to_image48(pixels)
    landmark_xy = detector.detect(image48)
    if landmark_xy is None:
        return fallback_priors(image48, label=label, sample_index=sample_index)
    return detected_priors(image48, landmark_xy=landmark_xy, label=label, sample_index=sample_index)


def schema_metadata() -> Dict[str, Any]:
    return {
        "schema_version": "d16_mediapipe_pixel_priors_v1",
        "image_shape": [48, 48],
        "part_count": len(PART_NAMES),
        "anchor_count": len(MICRO_ANCHOR_NAMES),
        "fallback_types": FALLBACK_TYPES,
        "npz_string_policy": "numeric_arrays_only_names_in_global_json",
    }


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
