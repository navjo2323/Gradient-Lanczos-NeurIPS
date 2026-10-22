"""Run an explicit affine experiment configuration and record its environment."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from . import gp_fit

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/affine_gpu_quick.json')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.device is not None:
        config['device'] = args.device
        config['require_gpu'] = args.device in ['mps', 'cuda']
    if args.output is not None:
        config['csv'] = str(args.output)
    output = Path(config['csv'])
    output.parent.mkdir(parents=True, exist_ok=True)
    device = gp_fit.choose_device(config['device'])
    metadata = {'started_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
                'configuration_source': args.config.name, 'actual_device': str(device),
                'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__,
                'platform': platform.platform(), 'torch_threads': torch.get_num_threads(),
                'mps_available': torch.backends.mps.is_available(),
                'cuda_available': torch.cuda.is_available(),
                'environment': {k: os.environ.get(k) for k in ['PYTORCH_ENABLE_MPS_FALLBACK',
                    'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO',
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']},
                'status': 'started',
                'historical_reproduction': 'Reconstructed settings; see docs/affine_gpu.md.'}
    metadata_path = output.with_suffix('.metadata.json')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    cli = []
    for key, value in config.items():
        if isinstance(value, bool):
            if value:
                cli.append('--' + key.replace('_', '-'))
        else:
            cli.extend(['--' + key.replace('_', '-'), str(value)])
    old_argv = sys.argv
    try:
        sys.argv = [gp_fit.__file__, *cli]
        gp_fit.main()
    finally:
        sys.argv = old_argv
    metadata.update(status='completed', finished_utc=datetime.now(timezone.utc).isoformat())
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
