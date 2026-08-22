from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location(
        'teacher_bundle_builder', ROOT / 'tools/build_teacher_drive_bundle.py'
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cache_archive_is_excluded_from_drive_bundle():
    module = load_builder()
    assert module.excluded(Path('ofix7_mid_seed42_records.zip'), 'cache')
    assert not module.excluded(Path('CACHE_COMPLETE.json'), 'cache')


def test_one_command_script_is_fail_closed_and_trains():
    text = (ROOT / 'tools/run_teacher_drive_bundle.sh').read_text(encoding='utf-8')
    assert 'set -Eeuo pipefail' in text
    assert 'trap on_error ERR' in text
    assert 'MINIFORGE_VERSION="26.3.2-2"' in text
    assert 'verify_sha256 "$INSTALLER" "$MINIFORGE_SHA256"' in text
    assert '/releases/latest/' not in text
    assert '--dry-run' not in text
    assert 'run_teacher_linux_seed42.py' in text
    assert 'TRAINING_COMPLETE.json' in text
    assert 'ofix7_mid_seed42_results.tar.gz' in text
