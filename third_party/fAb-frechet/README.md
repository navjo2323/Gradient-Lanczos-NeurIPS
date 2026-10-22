# Modified Arnoldi baseline

`lib.py` is the unchanged `fAb-frechet` implementation used in the graph experiments, associated with Daniel Kressner and Peter Oehme, “A novel Krylov subspace method for approximating Fréchet derivatives of large-scale matrix functions,” Electronic Transactions on Numerical Analysis 65, 110–122 (2026), https://doi.org/10.1553/etna_vol65s110.

The upstream GPL version 2 license is reproduced in `LICENSE`. This component is **not covered by the repository’s MIT license**. The original local distribution contained no upstream revision identifier or separate author copyright notice; no revision is inferred. The exact source hash and local collection provenance are recorded in `results/paper/graph/provenance.json`.

Only `lib.py` is needed for the baseline. Its source and license have been copied without changes. The release adapter loads it from this directory.
