# D13A Kaggle Full Train

D13A is a pure GNN hierarchical pixel-to-region reduction baseline. It trains from the FER-2013 pixel graph repo, reduces 2304 pixel nodes to K=144 region nodes, and classifies FER-2013.

Run order:

1. Run the local 5 epoch debug configs first.
2. If both local debug checks pass, run EdgeAware Lite first on Kaggle.
3. Run GINE after EdgeAware as the encoder control.
4. After each Kaggle run, keep the output zip or create a Kaggle Dataset from the output directory.

Commands:

```bash
bash kaggle/d13/run_d13a_kaggle.sh edgeaware
bash kaggle/d13/run_d13a_kaggle.sh gine
```

First-run constraints:

- Do not enable AMP, DDP, or torch.compile.
- Do not add a CNN teacher.
- Do not enable SupCon.
- Do not open D13B, motif slots, or motif claims from this run.
- Treat the check script as a run-health gate, not a final model-quality verdict.

