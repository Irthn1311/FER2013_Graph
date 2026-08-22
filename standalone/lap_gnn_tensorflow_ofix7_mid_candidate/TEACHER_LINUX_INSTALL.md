# Prepare The Teacher Google Drive Bundle

The teacher must not install dependencies or edit paths manually. Prepare the
bundle once on this development machine, upload the complete resulting folder
to Google Drive, and tell the teacher to run only `bash run.sh`.

## Build the bundle locally

From the TensorFlow package directory:

```powershell
python -B tools/build_teacher_drive_bundle.py
```

Default sources are already bound to this repository:

- CSV: `data/train.csv`, `data/val.csv`, `data/test.csv`;
- priors: `outputs/d16_mediapipe_pixel_priors_best_retry_rescue`;
- cache: `outputs/tensorflow_clean_graph_cache/ofix7_mid_seed42_records`;
- package: `standalone/lap_gnn_tensorflow_ofix7_mid_candidate`.

The output is:

```text
outputs/teacher_drive_bundle/ofix7_mid_seed42_teacher_bundle
```

Hardlinks are used by default, so creating the local bundle does not duplicate
the roughly 27 GiB input payload on the same Windows volume. Uploading or
copying the folder materializes its real size. The redundant 18 GB cache ZIP is
excluded because the extracted cache shards are already included.

To include an offline Miniforge installer, pass:

```powershell
python -B tools/build_teacher_drive_bundle.py `
  --miniforge-installer C:\path\to\Miniforge3-26.3.2-2-Linux-x86_64.sh
```

The builder and `run.sh` both require the registered Linux x86_64 SHA-256
`42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94`.
When the installer is not bundled, `run.sh` downloads that exact release only
if the teacher machine has no existing Conda installation.

Do not alter files after `BUNDLE_COMPLETE.json` is created. Upload the whole
`ofix7_mid_seed42_teacher_bundle` directory to Google Drive.
