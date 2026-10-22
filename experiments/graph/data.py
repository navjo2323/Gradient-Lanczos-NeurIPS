"""Download the exact SNAP edge lists used by the paper and verify SHA-256."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from ._bootstrap import ROOT

DATASETS = {
    "hepth": {
        "filename": "ca-HepTh.txt", "url": "https://snap.stanford.edu/data/ca-HepTh.txt.gz",
        "sha256": "8ce8a9aa75f0617d574a8109a6f8b189a9e38d214a6698ceaf25b2eecc873e90",
        "n": 9877, "nnz": 51946, "i": 7200, "j": 6969,
    },
    "amazon": {
        "filename": "com-amazon.ungraph.txt", "url": "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz",
        "sha256": "9c403688d01417ca452910947a7000bcd61965fc583f4afe57eb209c8b057b7a",
        "n": 334863, "nnz": 1851744, "i": 0, "j": 53525,
    },
}


def verify(graph: str, directory: Path) -> Path:
    spec = DATASETS[graph]
    path = directory / spec["filename"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec["sha256"]:
        raise ValueError(f"Unexpected SHA-256 for {path}: {digest}; expected {spec['sha256']}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/graphs")
    parser.add_argument("--graphs", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for graph in args.graphs:
        spec = DATASETS[graph]
        path = args.data_dir / spec["filename"]
        if not path.exists() and not args.verify_only:
            with urlopen(spec["url"], timeout=120) as response:
                raw = gzip.decompress(response.read())
            digest = hashlib.sha256(raw).hexdigest()
            if digest != spec["sha256"]:
                raise ValueError(f"Downloaded {graph} differs from the paper data: {digest}")
            path.write_bytes(raw)
        verify(graph, args.data_dir)
        print(json.dumps({"graph": graph, "path": str(path), "sha256": spec["sha256"]}))


if __name__ == "__main__":
    main()
