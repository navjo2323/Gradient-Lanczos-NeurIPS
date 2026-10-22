"""Locate unchanged paper implementations and the separately licensed baseline."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in [ROOT / "third_party/fAb-frechet", Path(__file__).parent / "implementation"]:
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
