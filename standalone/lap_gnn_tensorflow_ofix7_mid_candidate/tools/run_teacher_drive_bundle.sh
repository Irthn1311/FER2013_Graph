#!/usr/bin/env bash
set -Eeuo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$BUNDLE_ROOT/package"
INPUT_ROOT="$BUNDLE_ROOT/inputs"
FER_CSV="$INPUT_ROOT/fer13-split/train.csv"
PRIOR_ROOT="$INPUT_ROOT/priors"
GRAPH_CACHE_DIR="$INPUT_ROOT/graph_cache"
RESULTS_ROOT="$BUNDLE_ROOT/results"
OUTPUT_ROOT="$RESULTS_ROOT/ofix7_mid_seed42"
RUNTIME_ROOT="$BUNDLE_ROOT/runtime"
ENV_PREFIX="$RUNTIME_ROOT/lap-gnn-tf-driver470"
LOG_ROOT="$BUNDLE_ROOT/logs"
SESSION_ID="$(date +%Y%m%d_%H%M%S)"
SESSION_LOG_DIR="$LOG_ROOT/bootstrap_$SESSION_ID"
BOOTSTRAP_LOG="$SESSION_LOG_DIR/bootstrap.log"
CURRENT_STAGE="startup"
MINIFORGE_VERSION="26.3.2-2"
MINIFORGE_NAME="Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"
MINIFORGE_SHA256="42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}"

mkdir -p "$SESSION_LOG_DIR" "$RESULTS_ROOT" "$RUNTIME_ROOT"
exec > >(tee -a "$BOOTSTRAP_LOG") 2>&1

on_error() {
  local status=$?
  set +e
  {
    echo "[FAIL] stage=$CURRENT_STAGE"
    echo "[FAIL] line=${BASH_LINENO[0]:-unknown}"
    echo "[FAIL] command=${BASH_COMMAND:-unknown}"
    echo "[FAIL] exit_code=$status"
    echo "[FAIL] bootstrap_log=$BOOTSTRAP_LOG"
  } | tee "$SESSION_LOG_DIR/bootstrap_failure.txt"
  exit "$status"
}
trap on_error ERR

stage() {
  CURRENT_STAGE="$1"
  echo
  echo "================================================================================"
  echo "[START] $CURRENT_STAGE"
  echo "================================================================================"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    return 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Missing required directory: $1" >&2
    return 1
  fi
}

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "SHA-256 mismatch: $path" >&2
    echo "expected=$expected" >&2
    echo "actual=$actual" >&2
    return 1
  fi
  echo "sha256_pass=$actual file=$path"
}

stage "validate_bundle_layout"
require_file "$BUNDLE_ROOT/BUNDLE_COMPLETE.json"
require_file "$PACKAGE_DIR/CHECKSUMS.sha256"
require_file "$PACKAGE_DIR/environment-linux-driver470.yml"
require_file "$FER_CSV"
require_file "$GRAPH_CACHE_DIR/CACHE_COMPLETE.json"
require_dir "$PRIOR_ROOT/train"
require_dir "$PRIOR_ROOT/val"
require_dir "$PRIOR_ROOT/test"
echo "bundle_root=$BUNDLE_ROOT"
echo "bootstrap_log=$BOOTSTRAP_LOG"

stage "record_host_state"
uname -a || true
lscpu || true
free -h || true
nvidia-smi || true
nvidia-smi topo -m || true

stage "locate_or_install_conda"
if command -v conda >/dev/null 2>&1; then
  CONDA_EXE="$(command -v conda)"
elif [[ -x "$RUNTIME_ROOT/miniforge3/bin/conda" ]]; then
  CONDA_EXE="$RUNTIME_ROOT/miniforge3/bin/conda"
else
  INSTALLER="$RUNTIME_ROOT/$MINIFORGE_NAME"
  if [[ -f "$BUNDLE_ROOT/installers/Miniforge3-Linux-x86_64.sh" ]]; then
    cp "$BUNDLE_ROOT/installers/Miniforge3-Linux-x86_64.sh" "$INSTALLER"
  elif command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 \
      "$MINIFORGE_URL" \
      -o "$INSTALLER"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$INSTALLER" "$MINIFORGE_URL"
  else
    echo "Conda is absent and neither curl nor wget is available." >&2
    exit 1
  fi
  verify_sha256 "$INSTALLER" "$MINIFORGE_SHA256"
  bash "$INSTALLER" -b -p "$RUNTIME_ROOT/miniforge3"
  CONDA_EXE="$RUNTIME_ROOT/miniforge3/bin/conda"
fi
echo "conda_executable=$CONDA_EXE"

stage "create_or_reuse_tensorflow_environment"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  "$CONDA_EXE" env create \
    --yes \
    --prefix "$ENV_PREFIX" \
    --file "$PACKAGE_DIR/environment-linux-driver470.yml"
else
  echo "Reusing environment: $ENV_PREFIX"
fi

stage "install_and_verify_package"
export PYTHONNOUSERSITE=1
"$CONDA_EXE" run --no-capture-output --prefix "$ENV_PREFIX" \
  python -m pip install "$PACKAGE_DIR" --no-deps --disable-pip-version-check
"$CONDA_EXE" run --no-capture-output --prefix "$ENV_PREFIX" \
  python -B "$PACKAGE_DIR/tools/verify_checksums.py"

stage "prepare_fresh_output"
if [[ -f "$OUTPUT_ROOT/TRAINING_COMPLETE.json" ]]; then
  echo "Training is already complete: $OUTPUT_ROOT"
  ALREADY_COMPLETE=1
else
  ALREADY_COMPLETE=0
  if [[ -d "$OUTPUT_ROOT" ]] && [[ -n "$(find "$OUTPUT_ROOT" -mindepth 1 -print -quit)" ]]; then
    FAILED_ROOT="$RESULTS_ROOT/incomplete_$SESSION_ID"
    mv "$OUTPUT_ROOT" "$FAILED_ROOT"
    echo "Previous incomplete output moved to: $FAILED_ROOT"
  fi
fi

if [[ "$ALREADY_COMPLETE" -eq 0 ]]; then
  stage "preflight_and_train_seed42"
  "$CONDA_EXE" run --no-capture-output --prefix "$ENV_PREFIX" \
    python -B "$PACKAGE_DIR/tools/run_teacher_linux_seed42.py" \
      --fer-csv "$FER_CSV" \
      --prior-root "$PRIOR_ROOT" \
      --graph-cache-dir "$GRAPH_CACHE_DIR" \
      --output-root "$OUTPUT_ROOT" \
      --log-dir "$LOG_ROOT/launcher" \
      --gpu-index 0
fi

stage "verify_completion"
require_file "$OUTPUT_ROOT/TRAINING_COMPLETE.json"
require_file "$OUTPUT_ROOT/artifact_manifest.json"

stage "archive_results"
ARCHIVE="$BUNDLE_ROOT/ofix7_mid_seed42_results.tar.gz"
rm -f "$ARCHIVE"
tar -czf "$ARCHIVE" -C "$BUNDLE_ROOT" results logs

stage "complete"
echo "[PASS] TensorFlow OFIX7-mid seed42 completed."
echo "[PASS] output=$OUTPUT_ROOT"
echo "[PASS] archive=$ARCHIVE"
echo "[PASS] bootstrap_log=$BOOTSTRAP_LOG"
