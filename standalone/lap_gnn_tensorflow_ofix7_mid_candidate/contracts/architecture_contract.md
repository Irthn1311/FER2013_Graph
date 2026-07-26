# Architecture Contract

This is the framework-neutral contract for the locked OFIX7-mid candidate.

- Input: variable-size graphs from 48x48 grayscale FER2013 images.
- Node features: 37 ordered float32 values from `feature_schema.json`.
- Edge features: 8 ordered float32 values from `edge_schema.json`.
- Graph nodes: selected face/context pixels followed by five semantic anchors.
- Encoder: Linear(37,96), LayerNorm(96), GELU, Dropout(0.2).
- Graph stack: three gated-edge-MLP layers, mean aggregation, residual and LayerNorm enabled, edge hidden size 32, layer dropout 0.25.
- Context injection: after the final graph layer; five pooled part-group tokens; one 4-head transformer layer; initial context scale 0.5.
- Readout: micro-motif-support with major counts 3/3/3/1/2 and micro counts 2/2/2/1/1 for mouth/eye/brow/nose_cheek/global.
- Readout-only CLS and motif tokens do not participate in graph message passing.
- Classifier output: seven logits in FER class order.
- Exact parameter count: 1,061,192.

Tensor order, reductions, normalization, residual placement and initializers are
defined by the mechanically extracted PyTorch source and golden fixtures.
