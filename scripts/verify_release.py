#!/usr/bin/env python3
"""Check frozen record hashes and every mapped paper figure/table output."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/paper')
    parser.add_argument('--records-only', action='store_true')
    args = parser.parse_args()
    inventory = json.loads((ROOT / 'results/paper/SHA256SUMS.json').read_text())
    for name, expected in inventory.items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Missing or changed frozen paper record: {name}')
    manifest = json.loads((ROOT / 'docs/paper_manifest.json').read_text())
    labels = {item['item'] for item in manifest['items']}
    assert labels == {f'{kind} {i}' for kind in ['Figure', 'Table'] for i in range(1, 9)}
    checked = set()
    for item in manifest['items']:
        for name in [item['generator'], *item['records']]:
            if not (ROOT / name).is_file():
                raise RuntimeError(f"{item['item']} refers to a missing file: {name}")
        if not args.records_only:
            for name in item['outputs']:
                p = args.output / name
                if not p.is_file() or p.stat().st_size == 0:
                    raise RuntimeError(f"{item['item']} output missing: {p}; run scripts/reproduce_paper.py")
                checked.add(name)
    print(f'PASS: {len(inventory)} frozen record hashes; 8 figures and 8 tables mapped; '
          f'{len(checked)} generated files checked.')


if __name__ == '__main__':
    main()
