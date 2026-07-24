"""Locked scientific constants for the TensorFlow OFIX7-mid port."""

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
SPLIT_COUNTS = {"train": 28709, "val": 3589, "test": 3589}
PRIOR_SCHEMA_ID = "d16_mediapipe_pixel_priors_v1"
NODE_FEATURE_NAMES = [
    "intensity", "gx", "gy", "x_norm", "y_norm", "face_mask",
    *[f"part_soft_{index}" for index in range(13)],
    *[f"distance_map_{index}" for index in range(12)],
    "landmark_missing_flag", "grad_mag", "local_mean_3x3",
    "local_std_3x3", "laplacian_abs", "center_surround",
]
EDGE_FEATURE_NAMES = [
    "dx", "dy", "spatial_dist", "abs_intensity_diff",
    "abs_grad_mag_diff", "abs_laplacian_diff",
    "part_similarity", "same_dominant_part",
]
EXPECTED_PARAMETER_COUNT = 1_061_192

