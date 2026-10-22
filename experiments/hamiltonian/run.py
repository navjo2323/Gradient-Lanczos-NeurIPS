"""Run the unchanged Hamiltonian experiment from a checked-in configuration."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
from . import train

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/hamiltonian_paper.json')
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.out_dir is not None:
        config['out_dir'] = str(args.out_dir)
    output = Path(config['out_dir'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                'platform': platform.platform(), 'status': 'started'}
    metadata_path = output / 'run_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    cli = []
    for key, value in config.items():
        cli.append('--' + key.replace('_', '-'))
        cli.extend(str(x) for x in value) if isinstance(value, list) else cli.append(str(value))
    old_argv = sys.argv
    try:
        sys.argv = [train.__file__, *cli]
        train.main()
    finally:
        sys.argv = old_argv
    metadata.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
