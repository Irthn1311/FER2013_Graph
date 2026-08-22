'''Build the self-contained Google Drive bundle for the teacher Linux host.'''

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_PRIOR_ROOT = REPO_ROOT / 'outputs/d16_mediapipe_pixel_priors_best_retry_rescue'
DEFAULT_CACHE_ROOT = REPO_ROOT / 'outputs/tensorflow_clean_graph_cache/ofix7_mid_seed42_records'
DEFAULT_CSV_ROOT = REPO_ROOT / 'data'
DEFAULT_OUTPUT = REPO_ROOT / 'outputs/teacher_drive_bundle/ofix7_mid_seed42_teacher_bundle'
MINIFORGE_VERSION = '26.3.2-2'
MINIFORGE_LINUX_X86_64_SHA256 = (
    '42260ffe3830fb953d5eee1bbb32229ff06aa7c3833c1ed7a9a0420a95685d94'
)


@dataclass
class CopyStats:
    files: int = 0
    bytes: int = 0
    hardlinks: int = 0
    copies: int = 0

    def merge(self, other: 'CopyStats') -> None:
        self.files += other.files
        self.bytes += other.bytes
        self.hardlinks += other.hardlinks
        self.copies += other.copies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Create the one-command OFIX7-mid teacher Drive bundle.'
    )
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--package-dir', type=Path, default=PACKAGE_ROOT)
    parser.add_argument('--csv-root', type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument('--prior-root', type=Path, default=DEFAULT_PRIOR_ROOT)
    parser.add_argument('--graph-cache-dir', type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument('--miniforge-installer', type=Path)
    parser.add_argument('--mode', choices=('hardlink', 'copy'), default='hardlink')
    parser.add_argument('--progress-interval', type=int, default=1000)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block = stream.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def excluded(relative: Path, kind: str) -> bool:
    parts = set(relative.parts)
    if parts.intersection({'__pycache__', '.pytest_cache'}):
        return True
    if any(part.endswith('.egg-info') for part in relative.parts):
        return True
    if relative.suffix in {'.pyc', '.pyo'}:
        return True
    if kind == 'cache' and relative.name == 'ofix7_mid_seed42_records.zip':
        return True
    if kind == 'package' and relative.name.startswith('parity_'):
        return True
    return False


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'hardlink':
        try:
            os.link(source, destination)
            return 'hardlink'
        except OSError:
            pass
    shutil.copy2(source, destination)
    return 'copy'


def materialize_tree(
    source: Path,
    destination: Path,
    *,
    mode: str,
    kind: str,
    progress_interval: int,
) -> CopyStats:
    stats = CopyStats()
    for path in source.rglob('*'):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if excluded(relative, kind):
            continue
        method = materialize_file(path, destination / relative, mode)
        stats.files += 1
        stats.bytes += path.stat().st_size
        stats.hardlinks += int(method == 'hardlink')
        stats.copies += int(method == 'copy')
        if progress_interval > 0 and stats.files % progress_interval == 0:
            print(
                '{}: files={} GiB={:.2f} hardlinks={} copies={}'.format(
                    kind,
                    stats.files,
                    stats.bytes / (1024 ** 3),
                    stats.hardlinks,
                    stats.copies,
                ),
                flush=True,
            )
    return stats


def require_inputs(args: argparse.Namespace) -> None:
    required_files = [
        args.package_dir / 'CHECKSUMS.sha256',
        args.package_dir / 'tools/verify_checksums.py',
        args.package_dir / 'tools/run_teacher_drive_bundle.sh',
        args.csv_root / 'train.csv',
        args.csv_root / 'val.csv',
        args.csv_root / 'test.csv',
        args.prior_root / 'prior_schema.json',
        args.graph_cache_dir / 'CACHE_COMPLETE.json',
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    for split in ('train', 'val', 'test'):
        for root in (args.prior_root, args.graph_cache_dir):
            if not (root / split).is_dir():
                missing.append(str(root / split))
    if missing:
        raise FileNotFoundError('Missing bundle inputs: {}'.format(missing))
    if args.miniforge_installer and not args.miniforge_installer.is_file():
        raise FileNotFoundError(args.miniforge_installer)
    if args.miniforge_installer:
        installer_sha = sha256(args.miniforge_installer)
        if installer_sha != MINIFORGE_LINUX_X86_64_SHA256:
            raise RuntimeError(
                'Miniforge Linux x86_64 checksum mismatch: expected={} actual={}'.format(
                    MINIFORGE_LINUX_X86_64_SHA256, installer_sha
                )
            )
    if args.output_dir.exists():
        raise FileExistsError('Output bundle already exists: {}'.format(args.output_dir))


def write_lf(path: Path, text: str) -> None:
    path.write_text(text.replace('\r\n', '\n'), encoding='utf-8', newline='\n')


def main() -> None:
    args = parse_args()
    require_inputs(args)
    args.output_dir.mkdir(parents=True)
    total = CopyStats()
    sections: dict[str, dict[str, int]] = {}

    package_stats = materialize_tree(
        args.package_dir,
        args.output_dir / 'package',
        mode=args.mode,
        kind='package',
        progress_interval=args.progress_interval,
    )
    total.merge(package_stats)
    sections['package'] = package_stats.__dict__

    csv_stats = CopyStats()
    for name in ('train.csv', 'val.csv', 'test.csv'):
        source = args.csv_root / name
        method = materialize_file(
            source, args.output_dir / 'inputs/fer13-split' / name, args.mode
        )
        csv_stats.files += 1
        csv_stats.bytes += source.stat().st_size
        csv_stats.hardlinks += int(method == 'hardlink')
        csv_stats.copies += int(method == 'copy')
    total.merge(csv_stats)
    sections['fer_csv'] = csv_stats.__dict__

    prior_stats = materialize_tree(
        args.prior_root,
        args.output_dir / 'inputs/priors',
        mode=args.mode,
        kind='priors',
        progress_interval=args.progress_interval,
    )
    total.merge(prior_stats)
    sections['priors'] = prior_stats.__dict__

    cache_stats = materialize_tree(
        args.graph_cache_dir,
        args.output_dir / 'inputs/graph_cache',
        mode=args.mode,
        kind='cache',
        progress_interval=args.progress_interval,
    )
    total.merge(cache_stats)
    sections['graph_cache'] = cache_stats.__dict__

    source_run = args.package_dir / 'tools/run_teacher_drive_bundle.sh'
    write_lf(args.output_dir / 'run.sh', source_run.read_text(encoding='utf-8'))
    readme = '''# Run this experiment\n\nOpen a terminal in this directory and run exactly:\n\n```bash\nbash run.sh\n```\n\nDo not move or rename the `package` or `inputs` directories. Logs are written\nto `logs`, training output to `results`, and the final archive is\n`ofix7_mid_seed42_results.tar.gz`.\n'''
    write_lf(args.output_dir / 'README_FIRST.md', readme)

    installer_record = None
    if args.miniforge_installer:
        installer_dir = args.output_dir / 'installers'
        installer_dir.mkdir(parents=True, exist_ok=True)
        installer_method = materialize_file(
            args.miniforge_installer,
            installer_dir / 'Miniforge3-Linux-x86_64.sh',
            args.mode,
        )
        installer_size = args.miniforge_installer.stat().st_size
        total.files += 1
        total.bytes += installer_size
        total.hardlinks += int(installer_method == 'hardlink')
        total.copies += int(installer_method == 'copy')
        installer_record = {
            'included': True,
            'version': MINIFORGE_VERSION,
            'bytes': installer_size,
            'sha256': sha256(args.miniforge_installer),
        }

    package_manifest = json.loads(
        (args.package_dir / 'package_manifest.json').read_text(encoding='utf-8')
    )
    cache_complete = json.loads(
        (args.graph_cache_dir / 'CACHE_COMPLETE.json').read_text(encoding='utf-8')
    )
    manifest = {
        'schema_version': 'ofix7_mid_teacher_drive_bundle_v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'storage_mode_requested': args.mode,
        'files': total.files,
        'logical_bytes': total.bytes,
        'logical_gib': round(total.bytes / (1024 ** 3), 3),
        'hardlinks': total.hardlinks,
        'copies': total.copies,
        'sections': sections,
        'scientific_payload_sha256': package_manifest['scientific_payload_sha256'],
        'execution_contract_sha256': package_manifest['execution_contract_sha256'],
        'package_checksums_sha256': sha256(args.package_dir / 'CHECKSUMS.sha256'),
        'train_csv_sha256': sha256(args.csv_root / 'train.csv'),
        'val_csv_sha256': sha256(args.csv_root / 'val.csv'),
        'test_csv_sha256': sha256(args.csv_root / 'test.csv'),
        'prior_schema_sha256': sha256(args.prior_root / 'prior_schema.json'),
        'cache_manifest_sha256': cache_complete.get('cache_manifest_sha256'),
        'cache_graph_config_sha256': cache_complete.get('graph_config_sha256'),
        'miniforge_installer': installer_record or {'included': False},
        'run_command': 'bash run.sh',
    }
    write_lf(
        args.output_dir / 'BUNDLE_COMPLETE.json',
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
    )

    subprocess.run(
        [sys.executable, '-B', 'tools/verify_checksums.py'],
        cwd=args.output_dir / 'package',
        check=True,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print('bundle={}'.format(args.output_dir.resolve()), flush=True)


if __name__ == '__main__':
    main()
