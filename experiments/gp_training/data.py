"""Download and verify the exact Protein bytes used by the paper experiments."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import urllib.request

import numpy as np

COMMIT = "f0737523712fe66a1640e4b980b927e8490b58eb"
URL = f"https://raw.githubusercontent.com/treforevans/uci_datasets/{COMMIT}/uci_datasets/protein/data.csv.gz"
SHA256 = "f709d0107c3ee90ab019b9e5d77fc2ae182158958cb70b1903236a51334fd01b"


def cache_path() -> Path:
    root = Path(os.environ.get("LANCZOS_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
    return root / "protein" / "data.csv.gz"


def verify(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Protein checksum mismatch for {path}: {digest}; expected {SHA256}")


def download(path: Path | None = None) -> Path:
    path = path or cache_path()
    if path.exists():
        verify(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".download")
    try:
        urllib.request.urlretrieve(URL, temporary)
        verify(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_protein() -> tuple[np.ndarray, np.ndarray]:
    path = cache_path()
    if not path.exists():
        raise FileNotFoundError("Protein data is missing. Run: python experiments/gp_training/data.py")
    verify(path)
    values = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if values.shape != (45730, 10):
        raise ValueError(f"Unexpected Protein shape: {values.shape}")
    return values[:, :-1], values[:, -1, None]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Cache location; default data/protein/data.csv.gz")
    args = parser.parse_args()
    print(download(args.output))
