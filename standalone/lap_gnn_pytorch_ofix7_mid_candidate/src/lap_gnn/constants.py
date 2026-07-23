"""Locked scientific constants for the OFIX7-mid candidate."""

PACKAGE_NAME = "lap_gnn_pytorch_ofix7_mid_candidate"
PACKAGE_STATUS = "baseline_candidate"
PARENT_COMMIT = "241a8872027cd284fe679533a0be95cb48e7d253"
CHECKPOINT_POLICY_LOCK_SHA256 = "dfce606a69343b1a8de821ec3fc547d5700b94ff6c9a15f6b26d32e02601fc5f"
BASELINE_LOCK_SHA256 = "d54c9162de7e4f6bda2ee37dbe735939f7542195d9d2fe6dbd5e61cd85351dc3"
PRIOR_SCHEMA_ID = "d16_mediapipe_pixel_priors_v1"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
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
ANCHOR_GROUPS = ["mouth", "eye", "brow", "nose_cheek", "global"]
SPLIT_COUNTS = {"train": 28709, "val": 3589, "test": 3589}
PARAMETER_COUNT = 1_061_192
SIGNATURES = {
    "dataset": "695746df2e352c65370b4bb684abc2a99117e8dc4c297a0a145fe423ad6400a6",
    "split": "SPLIT_EXACT",
    "prior_schema": "8356d1e9c53fa91950da45e19a5170a458e1baff43fca938882770bcbc7c276c",
    "model": "50433ebc12de59a11a4e4168bb31c2c40b519566622e233c18aaeb215915cf3e",
    "graph": "363a9916bc4b5c76eda3dc95b6ae7df20bcc37a01a19c7b018468d181fb83561",
    "selector": "c97cca66c9b2256d255d15ecc6890dd0a1bd7398f11b085c821fffbc299a7643",
    "feature": "b97a6c8fe605022395bbe578e1d7a961fe8598ef519493798df83d8c79dfe043",
    "node_type": "0b203fc51f1db94b528493de08246339bcbb58ea06f31b13734223e474f01db7",
}
