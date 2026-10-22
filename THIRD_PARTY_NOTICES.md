# Third-party code and data

The root MIT license covers the authors' original contributions. It does not
replace the licenses listed here.

| Component | Use | License / notice |
|---|---|---|
| `third_party/fAb-frechet/` | Kressner–Oehme modified-Arnoldi graph baseline | GPLv2; see its `LICENSE` and attribution |
| Matrix-log/GP source provenance | Adapted experiment code from Krämer et al., *Gradients of functions of large matrices* | MIT upstream notice retained with the extracted files |
| SNAP ca-HepTh and com-Amazon | Graph experiments | Download from SNAP; retain source headers and cite dataset publications; see `docs/graph.md` |
| Protein (CASP) dataset | GP training and diagnostic | Download and verify the documented archive; see `docs/gp_training.md` |

The public paper repository bundles the GPLv2 baseline as a separate component;
original-code MIT permission does not remove its copyleft terms. The graph
comparison imports that component. Consult its license when redistributing a
combined or modified program.

Python dependencies are installed from their distributions and retain their
respective licenses. Raw experiment records are the authors' recorded outputs;
source and artifact hashes distinguish those records from regenerated outputs.
