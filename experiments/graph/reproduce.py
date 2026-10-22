"""Direct-file entry point used by the repository configuration runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.graph.run import main

if __name__ == "__main__":
    main()
