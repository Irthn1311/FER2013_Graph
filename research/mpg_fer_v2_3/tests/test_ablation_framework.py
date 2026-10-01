from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from mpg_fer_table_vi.model import (
    ABLATION_REGISTRY,
    TABLE_VI_ORDER,
    AblationMPGFER,
    AblationMode,
    registry_document,
)
from mpg_fer_table_vi.protocol import (
    AblationConfig,
    TABLE_INFERENCE,
    aggregate_table_vi,
    config_from_final_recipe,
    evaluate_canonical_public_fp32,
    final_recipe_template,
    validate_ablation_data_paths,
    validate_final_recipe_lock,
    validate_kaggle_mounted_input_firewall,
    write_json,
)
from mpg_fer_v2_3.checkpoint import sha256_file
from mpg_fer_v2_3.ema import ModelEMA
from mpg_fer_v2_3.model import MPGFER
from mpg_fer_table_vi.train import (
    RecipeEpochScheduler,
    build_recipe_optimizer,
    build_recipe_scheduler,
    compute_ablation_training_loss,
)


ROOT = Path(__file__).resolve().parents[1]


def _input(batch: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(101)
    return torch.rand(batch, 1, 48, 48, generator=generator)


def _capture_nodes(model: nn.Module, images: torch.Tensor):
    captured = {}

    def pixel_hook(_module, _inputs, output):
        captured["pixel"] = output

    def motif_hook(_module, _inputs, output):
        captured["motif"] = output[0]

    pixel_handle = model.pixel_gnn[-1].register_forward_hook(pixel_hook)
    composer = getattr(model, "motif_composer", None)
    motif_handle = composer.register_forward_hook(motif_hook) if composer else None
    try:
        logits, outputs = model(images)
    finally:
        pixel_handle.remove()
        if motif_handle is not None:
            motif_handle.remove()
    return logits, outputs, captured


def test_registry_has_exactly_seven_variants_plus_full_in_semantic_order() -> None:
    assert list(TABLE_VI_ORDER) == [
        AblationMode.NO_PIXEL_GNN,
        AblationMode.FIXED_POOL,
        AblationMode.NO_MOTIF_GNN,
        AblationMode.SINGLE_SCALE_12,
        AblationMode.NO_GEOM_BIAS,
        AblationMode.DENSE_MOTIF,
        AblationMode.NO_PIXEL_FUSION,
        AblationMode.FULL,
    ]
    assert len(ABLATION_REGISTRY) == 8
    required = {
        "internal_id",
        "paper_name",
        "scientific_question",
        "exact_intervention",
        "active_modules",
        "applicable_losses",
        "applicable_diagnostics",
        "expected_tensor_invariants",
    }
    for record in registry_document()["variants"]:
        assert set(record) == required
    fixed = ABLATION_REGISTRY[AblationMode.FIXED_POOL]
    assert fixed.applicable_losses["prototype_mi"] is False
    assert fixed.applicable_losses["prototype_diversity"] is False
    assert fixed.applicable_diagnostics["motif_prototype_diagnostics"] is False
    no_motif = ABLATION_REGISTRY[AblationMode.NO_MOTIF_GNN]
    assert no_motif.applicable_diagnostics["motif_routing_diagnostics"] is False


def test_full_is_exactly_equal_to_frozen_v23_with_same_state_and_input() -> None:
    torch.manual_seed(42)
    baseline = MPGFER().eval()
    full = AblationMPGFER(mode=AblationMode.FULL).eval()
    full.load_state_dict(baseline.state_dict(), strict=True)
    images = _input(1)
    with torch.no_grad():
        expected_logits, expected = baseline(images, return_routing_supports=True)
        actual_logits, actual = full(images, return_routing_supports=True)
    assert torch.equal(expected_logits, actual_logits)
    for key in (
        "final_logits",
        "pixel_logits",
        "motif_logits",
        "h_pixel_readout",
        "h_motif_readout",
        "fusion_representation",
        "supcon_embeddings",
        "motif_assignments",
        "motif_geometry",
    ):
        assert torch.equal(expected[key], actual[key]), key
    assert baseline.state_dict().keys() == full.state_dict().keys()


def test_shared_module_initialization_is_aligned_across_all_modes() -> None:
    shared_prefixes = (
        "pixel_proj.",
        "pixel_gnn.",
        "pixel_attn_pool.",
        "pixel_readout_proj.",
        "aux_pixel_head.",
        "motif_gnn.",
        "motif_attn_pool.",
        "motif_readout_proj.",
        "aux_motif_head.",
        "supcon_head.",
        "classifier.",
    )
    torch.manual_seed(42)
    reference = AblationMPGFER(mode=AblationMode.FULL).state_dict()
    for mode in AblationMode:
        torch.manual_seed(42)
        candidate = AblationMPGFER(mode=mode).state_dict()
        for name, value in reference.items():
            if name.startswith(shared_prefixes):
                assert torch.equal(value, candidate[name]), (mode.value, name)


@pytest.mark.parametrize("mode", list(AblationMode))
def test_all_modes_obey_shape_contract(mode: AblationMode) -> None:
    model = AblationMPGFER(mode=mode).eval()
    images = _input(2)
    if mode is AblationMode.FULL:
        logits, outputs, captured = _capture_nodes(model, images)
        pixel_nodes = captured["pixel"]
        motif_nodes = captured["motif"]
    else:
        logits, outputs = model(images)
        pixel_nodes = outputs["h_pixel_nodes"]
        motif_nodes = outputs["h_motif_nodes_initial"]
    assert logits.shape == (2, 7)
    assert pixel_nodes.shape == (2, 2304, 96)
    assert motif_nodes.shape == (2, 49, 192)
    assert outputs["h_pixel_readout"].shape == (2, 128)
    assert outputs["h_motif_readout"].shape == (2, 384)
    assert outputs["fusion_representation"].shape == (2, 512)


@pytest.mark.parametrize("mode", list(AblationMode))
def test_all_modes_forward_backward_optimizer_clip_and_ema(mode: AblationMode) -> None:
    torch.manual_seed(42)
    model = AblationMPGFER(mode=mode).train()
    ema = ModelEMA(model, decay=0.9)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    labels = torch.tensor([0, 0, 1, 1])
    logits, outputs = model(_input(4))
    loss, components = compute_ablation_training_loss(
        logits,
        outputs,
        labels,
        nn.CrossEntropyLoss(),
        AblationConfig(ablation_mode=mode.value),
    )
    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)
    for value in components.values():
        assert value is None or torch.isfinite(value).all()
    loss.backward()
    required = {
        "pixel_projection": model.pixel_proj[0].weight,
        "pixel_readout": model.pixel_readout_proj[0].weight,
        "motif_readout": model.motif_readout_proj[0].weight,
        "pixel_aux": model.aux_pixel_head.weight,
        "motif_aux": model.aux_motif_head.weight,
        "classifier": model.classifier[-1].weight,
        "supcon_head": model.supcon_head[0].weight,
    }
    if mode is not AblationMode.FIXED_POOL:
        required["prototypes"] = model.motif_composer.prototypes
    else:
        required["fixed_projection"] = model.fixed_pool_composer.projection[0].weight
    for name, parameter in required.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.any(parameter.grad != 0), name

    def assert_activity(name, parameter, active):
        if active:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
            assert torch.any(parameter.grad != 0), name
        else:
            assert parameter.grad is None, name

    pixel_gnn_active = mode is not AblationMode.NO_PIXEL_GNN
    for name in ("q_proj", "k_proj", "v_proj"):
        assert_activity(
            f"pixel_gnn.{name}",
            getattr(model.pixel_gnn[0], name).weight,
            pixel_gnn_active,
        )
    assert_activity(
        "pixel_gnn.ffn", model.pixel_gnn[0].ffn[0].weight, pixel_gnn_active
    )

    if mode is not AblationMode.FIXED_POOL:
        composer = model.motif_composer
        for name, parameter in {
            "assignment_query": composer.assignment_query.weight,
            "prototype_key": composer.prototype_key.weight,
            "prototypes": composer.prototypes,
            "occurrence_projection": composer.occurrence_proj[0].weight,
        }.items():
            assert_activity(name, parameter, True)
        for scale in ("8", "12", "16"):
            assert_activity(
                f"saliency_{scale}",
                composer.scale_saliency[scale].weight,
                mode is not AblationMode.SINGLE_SCALE_12 or scale == "12",
            )
        assert_activity(
            "scale_gate",
            composer.scale_gate.weight,
            mode is not AblationMode.SINGLE_SCALE_12,
        )

    motif_gnn_active = mode is not AblationMode.NO_MOTIF_GNN
    motif_layer = model.motif_gnn[0]
    for name in ("q_proj", "k_proj", "v_proj"):
        assert_activity(
            f"motif_gnn.{name}", getattr(motif_layer, name).weight, motif_gnn_active
        )
    assert_activity("motif_gnn.ffn", motif_layer.ffn[0].weight, motif_gnn_active)
    assert_activity(
        "motif_gnn.geom_proj",
        motif_layer.geom_proj.weight,
        motif_gnn_active and mode is not AblationMode.NO_GEOM_BIAS,
    )
    clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(clipped)
    optimizer.step()
    ema.update(model)
    assert ema.num_updates == 1


def test_no_pixel_gnn_executes_zero_pixel_layers() -> None:
    model = AblationMPGFER(mode=AblationMode.NO_PIXEL_GNN).eval()
    calls = [0]
    handles = [
        layer.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        for layer in model.pixel_gnn
    ]
    try:
        _, outputs = model(_input(1))
    finally:
        for handle in handles:
            handle.remove()
    assert calls[0] == 0
    assert outputs["h_pixel_nodes"].shape == (1, 2304, 96)


def test_fixed_pool_is_exact_12x12_mean_then_projection() -> None:
    model = AblationMPGFER(mode=AblationMode.FIXED_POOL).eval()
    assert not hasattr(model, "motif_composer")
    with torch.no_grad():
        _, outputs = model(_input(1))
        direct_mean = outputs["h_pixel_nodes"][
            :, model.fixed_pool_composer.support_idx_12, :
        ].mean(dim=2)
        direct_occurrences = model.fixed_pool_composer.projection(direct_mean)
    assert direct_mean.shape == (1, 49, 96)
    assert torch.equal(outputs["fixed_support_mean"], direct_mean)
    assert torch.equal(outputs["h_motif_nodes_initial"], direct_occurrences)
    assert outputs["motif_assignments"] is None
    assert outputs["motif_prototype_diagnostics_applicable"] is False
    assert "loss_mi" not in outputs and "loss_diversity" not in outputs


def test_no_motif_gnn_executes_zero_motif_blocks_and_reports_na() -> None:
    model = AblationMPGFER(mode=AblationMode.NO_MOTIF_GNN).eval()
    calls = [0]
    handles = [
        layer.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
        for layer in model.motif_gnn
    ]
    try:
        _, outputs = model(_input(1), return_routing_supports=True)
    finally:
        for handle in handles:
            handle.remove()
    assert calls[0] == 0
    assert outputs["motif_routing_diagnostics_applicable"] is False
    assert not any(key.startswith("motif_l1_") for key in outputs)


def test_single_scale_uses_only_12_support_and_no_scale_gate() -> None:
    model = AblationMPGFER(mode=AblationMode.SINGLE_SCALE_12).eval()
    calls = {"8": 0, "12": 0, "16": 0, "gate": 0}
    handles = []
    for scale in ("8", "12", "16"):
        handles.append(
            model.motif_composer.scale_saliency[scale].register_forward_hook(
                lambda _m, _i, _o, scale=scale: calls.__setitem__(
                    scale, calls[scale] + 1
                )
            )
        )
    handles.append(
        model.motif_composer.scale_gate.register_forward_hook(
            lambda *_: calls.__setitem__("gate", calls["gate"] + 1)
        )
    )
    try:
        _, outputs = model(_input(1))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == {"8": 0, "12": 1, "16": 0, "gate": 0}
    assert outputs["scale_gate_applicable"] is False
    assert torch.equal(outputs["scale_weights"][0, 0], torch.tensor([0.0, 1.0, 0.0]))


def test_no_geometry_bias_never_calls_geometry_projection() -> None:
    model = AblationMPGFER(mode=AblationMode.NO_GEOM_BIAS).eval()
    calls = [0]
    handles = [
        layer.geom_proj.register_forward_hook(
            lambda *_: calls.__setitem__(0, calls[0] + 1)
        )
        for layer in model.motif_gnn
    ]
    try:
        model(_input(1))
    finally:
        for handle in handles:
            handle.remove()
    assert calls[0] == 0


def test_dense_motif_selects_every_non_self_node_in_every_head_and_layer() -> None:
    model = AblationMPGFER(mode=AblationMode.DENSE_MOTIF).eval()
    with torch.no_grad():
        _, outputs = model(_input(1), return_routing_supports=True)
    for layer in range(1, 6):
        selected = outputs[f"motif_l{layer}_selected_indices"]
        assert outputs[f"motif_l{layer}_selected_k"].item() == 48
        assert selected.shape == (1, 6, 49, 48)
        queries = torch.arange(49).view(1, 1, 49, 1)
        assert not selected.to(torch.long).eq(queries).any()


def test_no_pixel_fusion_zeros_only_pixel_slice() -> None:
    torch.manual_seed(42)
    full = AblationMPGFER(mode=AblationMode.FULL).eval()
    no_pixel = AblationMPGFER(mode=AblationMode.NO_PIXEL_FUSION).eval()
    no_pixel.load_state_dict(full.state_dict(), strict=True)
    images = _input(1)
    with torch.no_grad():
        _, expected = full(images)
        _, actual = no_pixel(images)
    assert torch.count_nonzero(actual["fusion_representation"][:, :128]) == 0
    assert torch.equal(
        actual["fusion_representation"][:, 128:],
        expected["fusion_representation"][:, 128:],
    )
    assert torch.equal(actual["h_pixel_readout"], expected["h_pixel_readout"])
    assert torch.equal(actual["h_motif_readout"], expected["h_motif_readout"])
    assert torch.equal(actual["pixel_logits"], expected["pixel_logits"])
    assert torch.equal(
        actual["supcon_source_representation"], expected["fusion_representation"]
    )
    assert torch.equal(actual["supcon_embeddings"], expected["supcon_embeddings"])


def test_official_checkpoint_strict_loads_when_supplied() -> None:
    raw = os.environ.get("MPG_FER_V23_OFFICIAL_CHECKPOINT")
    if not raw:
        pytest.skip("official checkpoint supplied only to release preflight")
    checkpoint = Path(raw)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload)
    result = AblationMPGFER(mode=AblationMode.FULL).load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys


def test_private_firewall_rejects_test_private_and_ambiguous_roles(tmp_path) -> None:
    train = tmp_path / "train.csv"
    public = tmp_path / "val.csv"
    private = tmp_path / "test.csv"
    for path in (train, public, private):
        path.write_text("emotion,pixels\n", encoding="utf-8")
    assert validate_ablation_data_paths(train, public) == (train, public)
    with pytest.raises(RuntimeError, match="PRIVATE_FIREWALL"):
        validate_ablation_data_paths(train, private)
    private_dir = tmp_path / "PrivateTest"
    private_dir.mkdir()
    disguised = private_dir / "val.csv"
    disguised.write_text("emotion,pixels\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="PRIVATE_FIREWALL"):
        validate_ablation_data_paths(train, disguised)
    with pytest.raises(ValueError, match="requires basename"):
        validate_ablation_data_paths(public, train)


def test_kaggle_mounted_input_firewall_accepts_train_public_by_name_only(
    tmp_path,
) -> None:
    mounted = tmp_path / "input"
    dataset = mounted / "fer13-split"
    dataset.mkdir(parents=True)
    (dataset / "train.csv").write_text("not opened", encoding="utf-8")
    (dataset / "val.csv").write_text("not opened", encoding="utf-8")
    with patch.object(Path, "open", side_effect=AssertionError("file opened")):
        result = validate_kaggle_mounted_input_firewall(mounted)
    assert result["forbidden_private_markers_found"] is False
    assert result["entries_inspected_by_name_only"] == 3


@pytest.mark.parametrize(
    "forbidden_relative",
    (
        Path("fer13-split/test.csv"),
        Path("PrivateTest/labels.csv"),
        Path("private_test/labels.csv"),
        Path("private-test/labels.csv"),
    ),
)
def test_kaggle_mounted_input_firewall_rejects_forbidden_coexisting_path_without_open(
    tmp_path, forbidden_relative
) -> None:
    mounted = tmp_path / "input"
    dataset = mounted / "fer13-split"
    dataset.mkdir(parents=True)
    (dataset / "train.csv").write_text("not opened", encoding="utf-8")
    (dataset / "val.csv").write_text("not opened", encoding="utf-8")
    forbidden = mounted / forbidden_relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("must never be opened", encoding="utf-8")
    with patch.object(Path, "open", side_effect=AssertionError("private file opened")):
        with pytest.raises(RuntimeError, match="PRIVATE_FIREWALL"):
            validate_kaggle_mounted_input_firewall(mounted)


class _TinyClassifier(nn.Module):
    def forward(self, images):
        feature = images.flatten(1).mean(dim=1)
        logits = torch.stack(
            [
                feature,
                -feature,
                feature * 0,
                feature * 0,
                feature * 0,
                feature * 0,
                feature * 0,
            ],
            dim=1,
        )
        return logits, {}


def test_canonical_fp32_public_evaluator_is_repeatable_and_single_view() -> None:
    images = torch.tensor([0.0, 1.0, 0.5, 0.25]).view(4, 1, 1, 1).expand(-1, 1, 48, 48)
    labels = torch.tensor([1, 0, 0, 0])
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    model = _TinyClassifier()
    first = evaluate_canonical_public_fp32(
        model, loader, "cpu", dataset_role="PublicTest", source_path="val.csv"
    )
    second = evaluate_canonical_public_fp32(
        model, loader, "cpu", dataset_role="PublicTest", source_path="val.csv"
    )
    assert first == second
    assert first["inference"] == TABLE_INFERENCE
    assert first["autocast_enabled"] is False
    assert first["tf32_enabled"] is False
    assert first["single_view"] is True
    with pytest.raises(RuntimeError, match="PRIVATE_FIREWALL"):
        evaluate_canonical_public_fp32(
            model, loader, "cpu", dataset_role="PrivateTest", source_path="test.csv"
        )


def test_training_refuses_missing_or_unbound_final_recipe(tmp_path) -> None:
    config = AblationConfig()
    design = tmp_path / "ABLATION_DESIGN_LOCK.json"
    design.write_text(
        json.dumps(
            {
                "final_recipe_lock_sha256": None,
                "scientific_training_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="FINAL_RECIPE_LOCK.json absent"):
        validate_final_recipe_lock(tmp_path / "FINAL_RECIPE_LOCK.json", design, config)
    recipe = tmp_path / "FINAL_RECIPE_LOCK.json"
    recipe.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="absent or mismatched"):
        validate_final_recipe_lock(recipe, design, config)


def _authorized_recipe(tmp_path, *, optimizer="SGD", scheduler="constant"):
    recipe_path = tmp_path / "FINAL_RECIPE_LOCK.json"
    recipe = final_recipe_template()
    recipe["optimizer"] = {
        "family": optimizer,
        "kwargs": {"momentum": 0.9} if optimizer == "SGD" else {},
    }
    recipe["scheduler"] = {"family": scheduler, "kwargs": {}}
    write_json(recipe_path, recipe)
    design_path = tmp_path / "ABLATION_DESIGN_LOCK.json"
    write_json(
        design_path,
        {
            "final_recipe_lock_sha256": sha256_file(recipe_path),
            "scientific_training_authorized": True,
        },
    )
    config = config_from_final_recipe(
        recipe_path,
        mode=AblationMode.FULL,
        run_id="lock-test",
        output_dir=tmp_path / "out",
    )
    return recipe_path, design_path, config


def test_final_recipe_drives_optimizer_scheduler_and_binds_complete_config(
    tmp_path,
) -> None:
    recipe, design, config = _authorized_recipe(tmp_path)
    assert config.optimizer_family == "SGD"
    assert config.optimizer_kwargs == {"momentum": 0.9}
    assert config.scheduler_family == "constant"
    model = nn.Linear(2, 2)
    optimizer = build_recipe_optimizer(model, config)
    scheduler = build_recipe_scheduler(optimizer, config)
    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.param_groups[0]["momentum"] == 0.9
    assert isinstance(scheduler, RecipeEpochScheduler)
    assert scheduler.step(3) == config.learning_rate
    assert validate_final_recipe_lock(recipe, design, config)["sha256"] == sha256_file(
        recipe
    )

    # Runtime-safe values may change without altering scientific identity.
    config.num_workers = 7
    config.output_dir = str(tmp_path / "other-output")
    validate_final_recipe_lock(recipe, design, config)

    for field, value in (
        ("label_smoothing", 0.123),
        ("lambda_supcon", 0.123),
        ("pixel_dropout", 0.123),
        ("early_stop_patience", 99),
    ):
        tampered = config_from_final_recipe(
            recipe,
            mode=AblationMode.FULL,
            run_id="lock-test",
            output_dir=tmp_path / "out",
        )
        setattr(tampered, field, value)
        with pytest.raises(RuntimeError, match="non-runtime config differs"):
            validate_final_recipe_lock(recipe, design, tampered)


def test_final_recipe_rejects_incomplete_extra_or_unsupported_family(tmp_path) -> None:
    for mutation in ("missing", "extra", "unsupported"):
        recipe = final_recipe_template()
        if mutation == "missing":
            recipe["training_recipe"].pop("label_smoothing")
        elif mutation == "extra":
            recipe["training_recipe"]["unregistered"] = 1
        else:
            recipe["optimizer"]["family"] = "UnregisteredOptimizer"
        path = tmp_path / f"{mutation}.json"
        write_json(path, recipe)
        with pytest.raises(RuntimeError, match="SCIENTIFIC_TRAINING_REFUSED"):
            config_from_final_recipe(
                path,
                mode=AblationMode.FULL,
                run_id="x",
                output_dir=tmp_path,
            )


def test_table_aggregator_uses_fixed_order_and_never_performance_order(
    tmp_path,
) -> None:
    runs = []
    for index, mode in enumerate(reversed(TABLE_VI_ORDER)):
        root = tmp_path / mode.value
        root.mkdir()
        write_json(
            root / "ablation_manifest.json",
            {
                "ablation_mode": mode.value,
                "seed": 42,
                "private_test_permitted": False,
            },
        )
        write_json(
            root / "canonical_public_metrics.json",
            {
                "dataset_role": "PublicTest",
                "inference": TABLE_INFERENCE,
                "metrics": {"accuracy": index / 10, "macro_f1": index / 20},
            },
        )
        runs.append(root)
    document = aggregate_table_vi(runs, tmp_path / "table")
    assert document["row_order"] == [mode.value for mode in TABLE_VI_ORDER]
    assert [row["Configuration"] for row in document["rows"]] == [
        ABLATION_REGISTRY[mode].paper_name for mode in TABLE_VI_ORDER
    ]
    assert "SD" not in (tmp_path / "table" / "table_vi.csv").read_text(encoding="utf-8")


def test_committed_registry_matches_executable_registry() -> None:
    committed = json.loads(
        (ROOT / "ablation_registry.json").read_text(encoding="utf-8")
    )
    assert committed == registry_document()


def test_json_writer_rejects_non_finite_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="NaN and Infinity"):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
