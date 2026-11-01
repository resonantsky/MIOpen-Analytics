# AMD ROCm MIOpen Analytics Platform for Windows

# ROCm & PyTorch Performance & Package Comparison

## Run a stable/nightly comparison

Run `python MIOpen-Check.py` and enter a supported GFX architecture.

The runner creates `venv-stable` and `venv-nightly`, installs the requirements from the corresponding AMD repository, runs the normal benchmark in each environment, and writes `stable-results.txt`, `nightly-results.txt`, and `comparison.md` in the repository root.

Shared Python dependencies are installed from PyPI; ROCm and PyTorch packages are resolved independently from each repository's PyTorch and core wheel indexes so their versions match that channel.

For stable runs on gfx1010–gfx1012 and gfx1030–gfx1036, the runner pins Torch and ROCm packages to the available ROCm 10.0.0 builds.

Nightly runs remain unpinned and install the latest available packages. The benchmark installs torchvision and its architecture-specific AMD package in both environments.

A pip recommendation is shown only when both environments pass torchvision CPU/GPU NMS checks and the selected package inventory includes those matching packages.

Before prompting, the selector checks all architectures against the stable and nightly indexes for the current Python and platform.

Benchmark output is streamed to the terminal and recorded in each results file. At completion, the runner prints matched stable/nightly total and average latency and, when torchvision validation succeeds, a pip install command for the lower-average environment.

Use `python MIOpen-Check.py --mode verbose` to capture the verbose benchmark instead. Both environments are installed from the stable and nightly AMD wheel indexes, respectively.

Example output:

```
➜ python .\MIOpen-Check.py
Package installer: uv
Supported architectures:
  gfx10: gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036
  gfx11: gfx1100, gfx1101, gfx1102, gfx1103, gfx110x, gfx1150, gfx1151, gfx1152, gfx1153, gfx115x
  gfx12: gfx12-0, gfx1200, gfx1201, gfx1250
  gfx9: gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx942, gfx950
Checking required AMD package artifacts for Python 3.12 (win-amd64)...
AMD package availability (stable + nightly):
  gfx1010  PACKAGE SET OK
  gfx1011  PACKAGE SET OK
  gfx1012  PACKAGE SET OK
  gfx1030  PACKAGE SET OK
  gfx1031  PACKAGE SET OK
  gfx1032  PACKAGE SET OK
  gfx1033  PACKAGE SET OK
  gfx1034  PACKAGE SET OK
  gfx1035  PACKAGE SET OK
  gfx1036  PACKAGE SET OK
  gfx1100  PACKAGE SET OK
  gfx1101  PACKAGE SET OK
  gfx1102  PACKAGE SET OK
  gfx1103  PACKAGE SET OK
  gfx110x  PACKAGES MISSING
  gfx1150  PACKAGE SET OK
  gfx1151  PACKAGE SET OK
  gfx1152  PACKAGE SET OK
  gfx1153  PACKAGE SET OK
  gfx115x  PACKAGES MISSING
  gfx12-0  PACKAGES MISSING
  gfx1200  PACKAGE SET OK
  gfx1201  PACKAGE SET OK
  gfx1250  PACKAGES MISSING
  gfx900   PACKAGES MISSING
  gfx906   PACKAGES MISSING
  gfx908   PACKAGE SET OK
  gfx90a   PACKAGE SET OK
  gfx90c   PACKAGES MISSING
  gfx942   PACKAGES MISSING
  gfx950   PACKAGES MISSING
Enter the GFX architecture (for example, gfx1030): gfx1030
Pinning stable ROCm packages to 10.0.0 for gfx1030
Installing stable requirements for gfx1030
Using Python 3.12.9 environment at: venv-stable
Audited 14 packages in 6ms
Using Python 3.12.9 environment at: venv-stable
Audited 11 packages in 8ms
Installing nightly requirements for gfx1030
Using Python 3.12.9 environment at: venv-nightly
Audited 14 packages in 6ms
Using Python 3.12.9 environment at: venv-nightly
Audited 11 packages in 9ms
Running stable normal benchmark; recording stable-results.txt

=== Stable benchmark output (also saved to stable-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.782 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.205 ms
  Torch / included   0.90x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.356 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.996 ms
  Torch / included   1.67x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.474 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.956 ms
  Torch / included   1.42x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.359 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.921 ms
  Torch / included   0.96x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.781 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.797 ms
  Torch / included   1.54x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.581 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.851 ms
  Torch / included   0.85x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.532 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.257 ms
  Torch / included   1.56x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.471 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.541 ms
  Torch / included   1.37x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.665 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.962 ms
  Torch / included   0.93x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   7.116 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.507 ms
  Torch / included   1.58x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   0.888 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.029 ms
  Torch / included   0.86x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.190 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.289 ms
  Torch / included   1.70x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.395 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.514 ms
  Torch / included   1.58x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.500 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.548 ms
  Torch / included   0.97x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.873 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.801 ms
  Torch / included   2.15x
Running nightly normal benchmark; recording nightly-results.txt

=== Nightly benchmark output (also saved to nightly-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.778 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.124 ms
  Torch / included   0.92x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.127 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.965 ms
  Torch / included   1.64x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.453 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.772 ms
  Torch / included   1.46x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.220 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.891 ms
  Torch / included   0.95x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.554 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.771 ms
  Torch / included   1.53x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.563 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.799 ms
  Torch / included   0.87x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.310 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.284 ms
  Torch / included   1.45x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.485 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.582 ms
  Torch / included   1.35x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.565 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.829 ms
  Torch / included   0.93x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   6.670 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.346 ms
  Torch / included   1.53x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.045 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.187 ms
  Torch / included   0.88x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.281 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.387 ms
  Torch / included   1.64x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.292 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.437 ms
  Torch / included   1.59x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.494 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.665 ms
  Torch / included   0.90x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.854 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.906 ms
  Torch / included   2.02x

==============================================================================
Torchvision validation
Torchvision validation (stable): PASS - torch=2.13.0+rocm10.0.0, torchvision=0.28.0+rocm10.0.0, amd-torchvision-device=0.28.0+rocm10.0.0; CPU and GPU NMS passed
Torchvision validation (nightly): PASS - torch=2.16.0a0+rocm10.3.0a20261031, torchvision=0.30.0a0+rocm10.3.0a20261031, amd-torchvision-device=0.30.0a0+rocm10.3.0a20261031; CPU and GPU NMS passed
==============================================================================

==============================================================================
Stable / nightly benchmark summary
  Matched configurations 15
  Stable total / average   86.963 ms / 5.798 ms
  Nightly total / average  85.691 ms / 5.713 ms
  Preferred packages       nightly
  Comparison report        E:\ROCm-Check\comparison.md

Recommended pip install command:
python -m pip install --upgrade --pre --index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261031 amd-torchvision-device-gfx1030==0.30.0a0+rocm10.3.0a20261031 rocm==10.3.0a20261031 rocm-bootstrap==0.3.0 rocm-sdk-core==10.3.0a20261031 rocm-sdk-devel==10.3.0a20261101 rocm-sdk-device-gfx1030==10.3.0a20261031 rocm-sdk-libraries==10.3.0a20261031 torch==2.16.0a0+rocm10.3.0a20261031 torchaudio==2.11.1+rocm10.3.0a20261031 torchvision==0.30.0a0+rocm10.3.0a20261031
==============================================================================
```