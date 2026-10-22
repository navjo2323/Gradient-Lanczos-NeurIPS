# Tested environments

`tested-macos-arm64-py311.txt` records the exact package versions installed during
release validation on Python 3.11/macOS arm64. These are release-test versions,
not a reconstruction of missing historical environments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/tested-macos-arm64-py311.txt
python -m pip install --no-deps -e .
```

For another platform, install the appropriate PyTorch/JAX wheels and use the
optional dependencies in `pyproject.toml`. CPU tests are portable; MPS requires
Apple hardware and an MPS-enabled PyTorch build. A CUDA path exists in the affine
driver, but CUDA was not used for the paper and is not validated by this release.
