from lap_gnn_tf.config import (
    EXECUTION_CONTRACT_SHA256 as CONFIG_CONTRACT_SHA,
    load_config,
    validate_locked_config,
)
from lap_gnn_tf.training.execution import (
    EXECUTION_CONTRACT_SHA256 as RUNTIME_CONTRACT_SHA,
)

from _execution_contract_evidence import (
    CONTRACT_SHA,
    ROOT,
    contract_sha256,
)


def test_execution_contract_lock():
    config = load_config(
        ROOT / "configs" / "fer2013_ofix7_mid_tensorflow_seed42.yaml"
    )
    validate_locked_config(config)
    assert contract_sha256() == CONTRACT_SHA
    assert CONFIG_CONTRACT_SHA == CONTRACT_SHA
    assert RUNTIME_CONTRACT_SHA == CONTRACT_SHA
    assert config["locked"]["execution_contract_sha256"] == CONTRACT_SHA
