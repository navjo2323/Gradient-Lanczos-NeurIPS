"""Ensure runner options do not leak into experiment CLIs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_dry_run_before_or_after_config():
    for args in [['--dry-run', 'configs/graph_smoke.json'],
                 ['configs/graph_smoke.json', '--dry-run']]:
        result = subprocess.run([sys.executable, str(ROOT/'scripts/run_experiment.py'), *args],
                                text=True, capture_output=True, check=True)
        assert '--dry-run' not in result.stdout
        assert '--mode smoke' in result.stdout
        assert 'Run record:' not in result.stdout


def test_external_config_and_explicit_child_arguments(tmp_path):
    config = tmp_path/'custom.json'
    config.write_text(json.dumps({'script': 'experiments/graph/reproduce.py', 'args': ['--mode', 'smoke']}))
    result = subprocess.run([sys.executable, str(ROOT/'scripts/run_experiment.py'), str(config),
                             '--dry-run', '--', '--repeats', '1'], text=True, capture_output=True, check=True)
    assert '--mode smoke --repeats 1' in result.stdout
