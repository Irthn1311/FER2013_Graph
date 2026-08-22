'''Fail-closed single-GPU launcher for the teacher Linux host.'''

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback


MINIMUM_CUDA11_DRIVER_MAJOR = 450
EXPECTED_TENSORFLOW_VERSION = '2.13.1'
EXPECTED_KERAS_VERSION = '2.13.1'
DEFAULT_GRAPH_WORKERS = 8
DEFAULT_INTRA_OP_THREADS = 4
DEFAULT_INTER_OP_THREADS = 2
DEFAULT_PREFETCH = 8


class RunLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a', encoding='utf-8', buffering=1)

    def close(self) -> None:
        self.stream.close()

    def write(self, message: str) -> None:
        line = '[{}] {}'.format(datetime.now(timezone.utc).isoformat(), message)
        print(line, flush=True)
        self.stream.write(line + '\n')

    def raw(self, message: str) -> None:
        print(message, flush=True)
        self.stream.write(message + '\n')


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description='Run locked TensorFlow OFIX7-mid seed42 on one CUDA 11 GPU.'
    )
    parser.add_argument('--fer-csv', required=True, type=Path)
    parser.add_argument('--prior-root', required=True, type=Path)
    parser.add_argument('--graph-cache-dir', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument(
        '--config',
        type=Path,
        default=root / 'configs/fer2013_ofix7_mid_tensorflow_optimized_seed42.yaml',
    )
    parser.add_argument('--gpu-index', type=int, default=0)
    parser.add_argument('--numa-node', default='auto')
    parser.add_argument('--graph-workers', type=int, default=DEFAULT_GRAPH_WORKERS)
    parser.add_argument('--intra-op-threads', type=int, default=DEFAULT_INTRA_OP_THREADS)
    parser.add_argument('--inter-op-threads', type=int, default=DEFAULT_INTER_OP_THREADS)
    parser.add_argument('--tf-data-prefetch', type=int, default=DEFAULT_PREFETCH)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--skip-bounded-validation', action='store_true')
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + '\n', encoding='utf-8')


def command_text(command: list[str]) -> str:
    return shlex.join(str(item) for item in command)


def capture_safe(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        return {
            'command': command_text(command),
            'returncode': result.returncode,
            'output': result.stdout.strip(),
        }
    except Exception as exc:
        return {
            'command': command_text(command),
            'error_type': type(exc).__name__,
            'error': str(exc),
        }


def collect_diagnostics(args: argparse.Namespace) -> dict[str, object]:
    commands = [
        ['uname', '-a'],
        ['lscpu'],
        ['free', '-h'],
        ['df', '-h'],
        ['nvidia-smi'],
        ['nvidia-smi', 'topo', '-m'],
    ]
    return {
        'captured_at_utc': utc_now(),
        'platform': platform.platform(),
        'python': sys.version,
        'python_executable': sys.executable,
        'working_directory': str(Path.cwd()),
        'arguments': {key: str(value) for key, value in vars(args).items()},
        'environment': {
            key: os.environ.get(key)
            for key in (
                'PATH', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH',
                'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX', 'VIRTUAL_ENV',
            )
        },
        'commands': [capture_safe(command) for command in commands],
    }


@contextmanager
def phase(name: str, state: dict[str, object], log: RunLog):
    state['phase'] = name
    started = time.monotonic()
    log.write('[START] {}'.format(name))
    try:
        yield
    except Exception:
        log.write('[FAIL]  {} ({:.1f}s)'.format(name, time.monotonic() - started))
        raise
    else:
        log.write('[PASS]  {} ({:.1f}s)'.format(name, time.monotonic() - started))


def run_streamed(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    state: dict[str, object],
    log: RunLog,
) -> None:
    state['last_command'] = command_text(command)
    log.write('$ ' + str(state['last_command']))
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        log.raw(line.rstrip('\n'))
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(
            'Command failed with exit code {}: {}'.format(
                returncode, state.get('last_command')
            )
        )


def capture(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def driver_major(version: str) -> int:
    match = re.match(r'\s*(\d+)', version)
    if not match:
        raise ValueError('Cannot parse NVIDIA driver version: {!r}'.format(version))
    return int(match.group(1))


def driver_is_supported(version: str) -> bool:
    return driver_major(version) >= MINIMUM_CUDA11_DRIVER_MAJOR


def gpu_inventory() -> list[dict[str, str]]:
    if not shutil.which('nvidia-smi'):
        raise RuntimeError('nvidia-smi is required but was not found')
    output = capture([
        'nvidia-smi',
        '--query-gpu=index,name,pci.bus_id,driver_version,memory.total',
        '--format=csv,noheader,nounits',
    ])
    rows = []
    for line in output.splitlines():
        values = [value.strip() for value in line.split(',')]
        if len(values) != 5:
            raise RuntimeError('Unexpected nvidia-smi row: {!r}'.format(line))
        rows.append(dict(zip(
            ['index', 'name', 'pci_bus_id', 'driver_version', 'memory_mib'],
            values,
        )))
    return rows


def sysfs_pci_id(pci_bus_id: str) -> str:
    match = re.search(
        r'([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])$',
        pci_bus_id,
    )
    if not match:
        raise ValueError('Unexpected PCI bus ID: {!r}'.format(pci_bus_id))
    return match.group(1).lower()


def detect_numa_node(pci_bus_id: str) -> int | None:
    path = Path('/sys/bus/pci/devices') / sysfs_pci_id(pci_bus_id) / 'numa_node'
    if not path.is_file():
        return None
    node = int(path.read_text(encoding='utf-8').strip())
    return node if node >= 0 else None


def validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError('{} must be positive, got {}'.format(name, value))


def validate_inputs(args: argparse.Namespace, package_root: Path) -> dict:
    required_files = [
        args.fer_csv,
        args.config,
        package_root / 'CHECKSUMS.sha256',
        package_root / 'tools/verify_checksums.py',
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Required files missing: {}'.format(missing))
    if not args.prior_root.is_dir():
        raise FileNotFoundError(args.prior_root)
    marker = args.graph_cache_dir / 'CACHE_COMPLETE.json'
    if not marker.is_file():
        raise FileNotFoundError('Graph cache completion marker missing: {}'.format(marker))
    cache = json.loads(marker.read_text(encoding='utf-8'))
    expected = {
        'schema_version': 'tf_clean_graph_cache_v2_records',
        'node_dim': 37,
        'edge_dim': 8,
    }
    mismatches = {
        key: {'actual': cache.get(key), 'expected': value}
        for key, value in expected.items()
        if cache.get(key) != value
    }
    if mismatches:
        raise RuntimeError('Graph cache contract mismatch: {}'.format(mismatches))
    if args.output_root.exists():
        if not args.output_root.is_dir():
            raise FileExistsError('Output root is not a directory: {}'.format(args.output_root))
        if any(args.output_root.iterdir()):
            raise FileExistsError('Fresh output must be absent or empty: {}'.format(args.output_root))
    for name in (
        'graph_workers', 'intra_op_threads', 'inter_op_threads', 'tf_data_prefetch'
    ):
        validate_positive(name, int(getattr(args, name)))
    return cache


def training_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        '-B',
        '-m',
        'lap_gnn_tf.cli.train',
        '--config', str(args.config.resolve()),
        '--fer-csv', str(args.fer_csv.resolve()),
        '--prior-root', str(args.prior_root.resolve()),
        '--output-root', str(args.output_root.resolve()),
        '--clean-graph-cache-dir', str(args.graph_cache_dir.resolve()),
        '--device', 'gpu',
        '--graph-workers', str(args.graph_workers),
        '--intra-op-threads', str(args.intra_op_threads),
        '--inter-op-threads', str(args.inter_op_threads),
        '--tf-data-prefetch', str(args.tf_data_prefetch),
        '--batch-size', '16',
        '--eval-batch-size', '32',
        '--no-resume',
        '--mixed-precision',
        '--no-xla',
        '--memory-growth',
    ]


def failure_hints(state: dict[str, object], exc: BaseException) -> list[str]:
    text = '{} {}'.format(state.get('phase', ''), exc).lower()
    hints = [
        'Open launcher.log first, then failure_report.json in the same directory.',
        'Do not change model or config to bypass a failed preflight.',
    ]
    if 'gpu' in text or 'cuda' in text or 'cudnn' in text or 'nvidia' in text:
        hints.extend([
            'Confirm nvidia-smi works and the selected GPU is not occupied.',
            'Confirm the environment contains CUDA 11.8 and cuDNN 8.6 user libraries.',
            'Check LD_LIBRARY_PATH and CONDA_PREFIX in failure_report.json.',
        ])
    if 'checksum' in text:
        hints.append('Restore the package; do not regenerate checksums on the teacher host.')
    if 'cache' in text:
        hints.append('Verify CACHE_COMPLETE.json and copy the full records cache to local SSD.')
    if 'memory' in text or 'resourceexhausted' in text or 'oom' in text:
        hints.append('Stop other GPU processes and keep the locked batch size at 16.')
    return hints


def execute(args: argparse.Namespace, session_dir: Path, log: RunLog) -> None:
    package_root = Path(__file__).resolve().parents[1]
    state: dict[str, object] = {'phase': 'startup', 'last_command': None}
    write_json(session_dir / 'diagnostics.json', collect_diagnostics(args))
    log.write('logs: {}'.format(session_dir))
    log.write('python: {}'.format(sys.executable))

    try:
        with phase('validate_platform_and_inputs', state, log):
            if platform.system() != 'Linux':
                raise RuntimeError('This launcher is intentionally Linux-only')
            cache = validate_inputs(args, package_root)

        with phase('select_gpu_and_numa', state, log):
            inventory = gpu_inventory()
            selected = next(
                (gpu for gpu in inventory if int(gpu['index']) == args.gpu_index),
                None,
            )
            if selected is None:
                raise RuntimeError('GPU index {} not found: {}'.format(args.gpu_index, inventory))
            if not driver_is_supported(selected['driver_version']):
                raise RuntimeError(
                    'CUDA 11 compatibility requires NVIDIA driver branch {} or newer; found {}'.format(
                        MINIMUM_CUDA11_DRIVER_MAJOR, selected['driver_version']
                    )
                )
            detected_numa = detect_numa_node(selected['pci_bus_id'])
            numa_node = detected_numa if args.numa_node == 'auto' else int(args.numa_node)
            prefix: list[str] = []
            if numa_node is not None:
                if not shutil.which('numactl'):
                    raise RuntimeError('numactl is required for NUMA pinning but was not found')
                prefix = [
                    'numactl',
                    '--cpunodebind={}'.format(numa_node),
                    '--membind={}'.format(numa_node),
                ]

        env = os.environ.copy()
        env.update({
            'CUDA_VISIBLE_DEVICES': str(args.gpu_index),
            'OMP_NUM_THREADS': str(args.intra_op_threads),
            'MKL_NUM_THREADS': str(args.intra_op_threads),
            'TF_NUM_INTRAOP_THREADS': str(args.intra_op_threads),
            'TF_NUM_INTEROP_THREADS': str(args.inter_op_threads),
            'TF_CPP_MIN_LOG_LEVEL': '1',
            'PYTHONUNBUFFERED': '1',
        })

        plan = {
            'decision': 'DRY_RUN' if args.dry_run else 'READY_TO_RUN',
            'execution': 'single_gpu_single_process',
            'tensorflow_expected': EXPECTED_TENSORFLOW_VERSION,
            'keras_expected': EXPECTED_KERAS_VERSION,
            'cuda_userspace_expected': '11.8',
            'cudnn_expected': '8.6',
            'gpu': selected,
            'visible_gpu': args.gpu_index,
            'detected_numa_node': detected_numa,
            'selected_numa_node': numa_node,
            'graph_workers': args.graph_workers,
            'intra_op_threads': args.intra_op_threads,
            'inter_op_threads': args.inter_op_threads,
            'tf_data_prefetch': args.tf_data_prefetch,
            'batch_size': 16,
            'eval_batch_size': 32,
            'mixed_precision': True,
            'xla': False,
            'resume': False,
            'cache_schema': cache.get('schema_version'),
            'cache_graph_config_sha256': cache.get('graph_config_sha256'),
            'command': prefix + training_command(args),
        }
        write_json(session_dir / 'runtime_plan.json', plan)
        log.raw(json.dumps(plan, indent=2, default=str))
        if args.dry_run:
            write_json(session_dir / 'launcher_summary.json', {
                'decision': 'DRY_RUN_PASS',
                'completed_at_utc': utc_now(),
            })
            return

        with phase('verify_package_checksums', state, log):
            run_streamed(
                [sys.executable, '-B', 'tools/verify_checksums.py'],
                cwd=package_root, env=env, state=state, log=log,
            )

        probe = (
            'import importlib.metadata as md, json, tensorflow as tf; '
            'g=tf.config.list_physical_devices(\'GPU\'); '
            'assert tf.__version__ == \'{}\', tf.__version__; '
            'assert md.version(\'keras\') == \'{}\', md.version(\'keras\'); '
            'assert len(g) == 1, g; '
            'print(json.dumps({\'tensorflow\': tf.__version__, '
            '\'keras\': md.version(\'keras\'), \'gpu\': [x.name for x in g], '
            '\'build\': tf.sysconfig.get_build_info()}, default=str))'
        ).format(EXPECTED_TENSORFLOW_VERSION, EXPECTED_KERAS_VERSION)
        with phase('probe_tensorflow_cuda_runtime', state, log):
            run_streamed(
                prefix + [sys.executable, '-B', '-c', probe],
                cwd=package_root, env=env, state=state, log=log,
            )

        if not args.skip_bounded_validation:
            with phase('run_bounded_scientific_validation', state, log):
                run_streamed(prefix + [
                    sys.executable,
                    '-B',
                    '-m',
                    'lap_gnn_tf.cli.validate',
                    '--config', str(args.config.resolve()),
                    '--fer-csv', str(args.fer_csv.resolve()),
                    '--prior-root', str(args.prior_root.resolve()),
                    '--package-root', str(package_root),
                    '--golden',
                    '--require-gpu',
                ], cwd=package_root, env=env, state=state, log=log)

        with phase('train_seed42', state, log):
            run_streamed(
                prefix + training_command(args),
                cwd=package_root, env=env, state=state, log=log,
            )
        write_json(session_dir / 'launcher_summary.json', {
            'decision': 'TRAINING_COMPLETED',
            'completed_at_utc': utc_now(),
            'output_root': str(args.output_root),
            'runtime_plan': str(session_dir / 'runtime_plan.json'),
            'log': str(log.path),
        })
        log.write('[PASS] launcher completed')
    except BaseException as exc:
        report = {
            'decision': 'FAILED',
            'failed_at_utc': utc_now(),
            'phase': state.get('phase'),
            'last_command': state.get('last_command'),
            'exception_type': type(exc).__name__,
            'exception': str(exc),
            'traceback': traceback.format_exc(),
            'diagnostics': str(session_dir / 'diagnostics.json'),
            'runtime_plan': str(session_dir / 'runtime_plan.json'),
            'launcher_log': str(log.path),
            'hints': failure_hints(state, exc),
        }
        report_path = session_dir / 'failure_report.json'
        write_json(report_path, report)
        log.write('[FAIL] launcher stopped in phase: {}'.format(state.get('phase')))
        log.write('[FAIL] {}: {}'.format(type(exc).__name__, exc))
        log.write('failure report: {}'.format(report_path))
        raise


def main() -> None:
    args = parse_args()
    log_root = args.log_dir or (
        args.output_root.parent / '{}_launcher_logs'.format(args.output_root.name)
    )
    session_dir = log_root / datetime.now().strftime('%Y%m%d_%H%M%S')
    session_dir.mkdir(parents=True, exist_ok=False)
    log = RunLog(session_dir / 'launcher.log')
    try:
        execute(args, session_dir, log)
    finally:
        log.close()


if __name__ == '__main__':
    main()
