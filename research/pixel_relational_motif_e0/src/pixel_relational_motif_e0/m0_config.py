"""Preregistered constants for PGM M0 (Issue #82)."""

from __future__ import annotations

import hashlib
import json


EXPERIMENT_ID = "PGM_M0_INCREMENTAL_GRAPH_STRUCTURED_PROCESSING"
ISSUE_NUMBER = 82
E0R_SCIENTIFIC_SHA = "671e3c2f69607778f08a923026e76744561e13b2"
E0R_HEAD_SHA = "49e65e1032f1af13fd615a2e6f7a64f20760684b"
OCCURRENCES_SHA256 = "30f1a5b642af2ecdfc29ae73960fb01f90c391964db845bd4b33cf3c017f7fa9"
R1_RESULTS_SHA256 = "b6675ce694f5a607cfea07abb3ed1065753f42ee848f7595d1cb8e682361a0ed"
R2_RESULTS_SHA256 = "cd23942b0934e005fda0701d4699275738d1134ebbca6c992fe957c5bdb56722"
E0R_SUMMARY_SHA256 = "0f223cb5e422810f138db387f4afc703575ae651ebf042efdc0898cb0b033821"
DISTANCE_MEDIAN = 0.5057389074044559
TRAIN_IDS_SHA256 = "5020e75f45ec0b8ea45fcb867c566367a232bb7245838a121f22b602c51e3be7"
PUBLIC_IDS_SHA256 = "3b82a8c2ab1c92c8469f3741bb2fa8cdf4003a4ba7bb2cf5ce1c2608565c6df7"

K = 128
GEOMETRY_BINS = 8
O_DIM = K
G_DIM = K * K * GEOMETRY_BINS
X_DIM = O_DIM + G_DIM
SEEDS = (42, 43, 44, 45, 46)
N_CLASSES = 7
HIDDEN_DIM = 32
EPOCHS = 50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42

L_CONFIG = {
    "family": "L",
    "scaler": "StandardScaler(with_mean=False)",
    "penalty": "l2",
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 2000,
    "tol": 1e-6,
}
M_CONFIG = {
    "family": "M",
    "dimensions": [X_DIM, HIDDEN_DIM, HIDDEN_DIM, N_CLASSES],
    "activations": ["relu", "relu"],
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "loss": "CrossEntropyLoss",
    "batch_size": 256,
    "epochs": EPOCHS,
    "dtype": "float32",
    "amp": False,
    "scheduler": None,
    "early_stopping": False,
    "class_weighting": False,
    "augmentation": False,
    "expected_parameter_count": 4199719,
}
G_CONFIG = {
    "family": "G",
    "components": K,
    "relations": GEOMETRY_BINS,
    "embedding_dim": HIDDEN_DIM,
    "message_layers": 2,
    "aggregation": "mean_incoming_nonself",
    "pooling": "mean_nodes",
    "optimizer": "AdamW",
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "loss": "CrossEntropyLoss",
    "batch_size": 64,
    "epochs": EPOCHS,
    "dtype": "float32",
    "amp": False,
    "scheduler": None,
    "early_stopping": False,
    "class_weighting": False,
    "augmentation": False,
    "relation_initializer": "per_matrix_pytorch_linear_default",
    "self_transform_bias": False,
    "expected_parameter_count": 22759,
}


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


L_CONFIG_SHA256 = config_sha256(L_CONFIG)
M_CONFIG_SHA256 = config_sha256(M_CONFIG)
G_CONFIG_SHA256 = config_sha256(G_CONFIG)
