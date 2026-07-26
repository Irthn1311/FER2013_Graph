"""Development-only bounded PyTorch readout trace for the repair task."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lap_gnn.config import load_config
from lap_gnn.model.d16_model import D16Model
from lap_gnn.validation import load_golden_batch, load_portable_model_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch-package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = args.pytorch_package_root.resolve()
    config = load_config(root / "configs/fer2013_ofix7_mid_seed42.yaml")
    model = D16Model.from_config(config, input_dim=37).cpu()
    model.load_state_dict(load_portable_model_state(root), strict=True)
    model.eval()
    batch = load_golden_batch(root)
    with torch.no_grad():
        output = model(batch)
    to_numpy = lambda tensor: tensor.detach().numpy()

    arrays = {
        "pre_readout_node_representation": to_numpy(output["node_embeddings"]),
        "classifier_input": to_numpy(output["z_image"]),
        "logits": to_numpy(output["logits"]),
        "major_tokens": to_numpy(output["micro_major_motif_tokens"]),
        "major_transformed_tokens": to_numpy(output["micro_major_motif_transformed_tokens"]),
        "micro_tokens": to_numpy(output["micro_motif_tokens"]),
        "micro_transformed_tokens": to_numpy(output["micro_motif_transformed_tokens"]),
        "micro_support_gate": to_numpy(output["micro_support_gate"]),
    }
    for name, tensor in output["part_embeddings"].items():
        arrays[f"part_pool_{name}"] = to_numpy(tensor)

    readout = model.readout
    major_transformed = output["micro_major_motif_transformed_tokens"]
    micro_transformed = output["micro_motif_transformed_tokens"]
    z_major = major_transformed.mean(dim=1)
    z_micro = micro_transformed.mean(dim=1)
    micro_project_norm = readout.micro_project[0](z_micro)
    micro_projected = readout.micro_project[1](micro_project_norm)
    gate_concat = torch.cat([z_major, z_micro], dim=1)
    gate_norm = readout.gate[0](gate_concat)
    gate_linear = readout.gate[1](gate_norm)
    gate_gelu = torch.nn.functional.gelu(gate_linear)
    gate_logits = readout.gate[3](gate_gelu)
    gate = torch.sigmoid(gate_logits)
    z_support = z_major + gate * micro_projected
    residual = torch.stack(
        [output["part_embeddings"][name] for name in readout.part_order], dim=1,
    ).flatten(start_dim=1)
    fused = torch.cat([z_support, residual], dim=1)
    projection_norm = readout.projection[0](fused)
    projection_linear = readout.projection[1](projection_norm)
    projection_gelu = torch.nn.functional.gelu(projection_linear)
    projection_output = readout.projection[4](projection_gelu)
    arrays.update({
        "z_major": to_numpy(z_major),
        "z_micro": to_numpy(z_micro),
        "micro_project_norm": to_numpy(micro_project_norm),
        "micro_projected": to_numpy(micro_projected),
        "gate_concat": to_numpy(gate_concat),
        "gate_norm": to_numpy(gate_norm),
        "gate_linear": to_numpy(gate_linear),
        "gate_gelu": to_numpy(gate_gelu),
        "gate_logits": to_numpy(gate_logits),
        "z_support": to_numpy(z_support),
        "residual_concat": to_numpy(residual),
        "fused_readout": to_numpy(fused),
        "projection_norm": to_numpy(projection_norm),
        "projection_linear": to_numpy(projection_linear),
        "projection_gelu": to_numpy(projection_gelu),
        "projection_output": to_numpy(projection_output),
    })

    tokens = torch.cat(
        [output["micro_major_motif_tokens"], output["micro_motif_tokens"]], dim=1,
    ) + readout.token_type_embedding
    cls = readout.cls_token.expand(batch.num_graphs, -1, -1)
    tokens_in = torch.cat([cls, tokens], dim=1)
    layer = readout.encoder.layers[0]
    qkv = torch.nn.functional.linear(
        tokens_in, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias,
    )
    q, k, v = qkv.chunk(3, dim=-1)
    head_dim = q.shape[-1] // layer.self_attn.num_heads

    def split_heads(value):
        return value.reshape(
            value.shape[0], value.shape[1], layer.self_attn.num_heads, head_dim,
        ).transpose(1, 2)

    q_heads = split_heads(q)
    k_heads = split_heads(k)
    v_heads = split_heads(v)
    q_scaled = q_heads * (head_dim ** -0.5)
    scores = torch.matmul(q_scaled, k_heads.transpose(-2, -1))
    attention = torch.softmax(scores, dim=-1)
    weighted = torch.matmul(attention, v_heads)
    weighted_merge = weighted.transpose(1, 2).reshape(
        tokens_in.shape[0], tokens_in.shape[1], -1,
    )
    attention_output = torch.nn.functional.linear(
        weighted_merge, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias,
    )
    norm1 = layer.norm1(tokens_in + attention_output)
    ff1 = layer.linear1(norm1)
    ff_gelu = torch.nn.functional.gelu(ff1)
    ff2 = layer.linear2(ff_gelu)
    norm2 = layer.norm2(norm1 + ff2)
    arrays.update({
        "transformer_tokens_in": to_numpy(tokens_in),
        "attention_q": to_numpy(q),
        "attention_k": to_numpy(k),
        "attention_v": to_numpy(v),
        "attention_scores": to_numpy(scores),
        "attention_softmax": to_numpy(attention),
        "attention_weighted_sum": to_numpy(weighted),
        "attention_output": to_numpy(attention_output),
        "transformer_norm1": to_numpy(norm1),
        "transformer_ff1": to_numpy(ff1),
        "transformer_ff_gelu": to_numpy(ff_gelu),
        "transformer_ff2": to_numpy(ff2),
        "transformer_norm2": to_numpy(norm2),
    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    print(f"saved={args.output} tensors={len(arrays)}")


if __name__ == "__main__":
    main()
