# MPG-FER v2.3

Issue [#97](https://github.com/Irthn1311/FER2013_Graph/issues/97) preregisters this direct continuation from reviewed v2.2 commit `0a258fd43cc4d8f45afa54ea1328f068c52cbee0`.

The only scientific delta is fixed residual-branch scaling in the existing Motif Graph blocks: both attention and FFN residual updates use scale `0.5` in Layers 1-2 and `1.0` in Layers 3-5. The intervention adds no parameters. The v2.2 dynamic Top-K schedule remains `[8,16,16,16,24]`.

Canonical v2.3 source SHA-256: `1e63aadd13d53024c1b279dd4cc9bbc943048a6751899d8ecbabea3b12082f87`.

## Local validation

```powershell
cd D:\SGU\CNTT\DIP\FER_2013_GRAPH\fer_d5\research\mpg_fer_v2_3
$env:PYTHONPATH = (Resolve-Path 'src').Path
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe -m pytest -q
C:\Users\ADMIN\anaconda3\envs\fer-graph\python.exe tools\bounded_gpu_audit.py --batch-size 8
```

The official contract remains Kaggle Tesla T4, physical batch 16, gradient accumulation 2, EMA PublicTest flip-TTA checkpoint selection, and one PrivateTest evaluation only after checkpoint freeze. See `V23_IMPLEMENTATION_REPORT.md` for staging commands, measured validation, output paths, and scientific boundaries.
