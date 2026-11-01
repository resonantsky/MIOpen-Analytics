# AMD ROCm MIOpen Analytics Platform for Windows

# ROCm & PyTorch Performance & Package Comparison

## Run a stable/nightly comparison

Run `python MIOpen-Check.py` and enter a supported GFX architecture.

The runner creates `venv-stable` and `venv-nightly`, installs the requirements from the corresponding AMD repository, runs the normal benchmark in each environment, and writes `stable-results.txt`, `nightly-results.txt`, and `comparison.md` in the repository root.

Shared Python dependencies are installed from PyPI; ROCm and PyTorch packages are resolved independently from each repository's PyTorch and core wheel indexes so their versions match that channel.

For stable runs on gfx1010–gfx1012 and gfx1030–gfx1036, the runner pins Torch and ROCm packages to the available ROCm 10.0.0 builds.

Nightly runs remain unpinned and install the latest available packages. The benchmark installs torchvision and its architecture-specific AMD package in both environments.

A pip recommendation is shown only when both environments pass torchvision CPU/GPU NMS checks and the selected package inventory includes those matching packages.

Before prompting, the selector checks all architectures against the stable and nightly indexes for the current Python and platform. Green means every required AMD package has a compatible artifact (wheel, or the ROCm source archive where applicable); red means at least one is unavailable; yellow means an index could not be checked.

The check is informational and does not disable choices or claim that a GPU runtime test passed. When `uv` is available, it handles installation with a concise progress display; otherwise the runner falls back to pip. Benchmark output is streamed to the terminal and rcodd in each results file. At completion, the runner prints matched stable/nightly total and average latency and, when torchvision validation succeeds, a pip install command for the lower-average environment.

Use `python MIOpen-Check.py --mode verbose` to capture the verbose benchmark instead. Both environments are installed from the stable and nightly AMD wheel indexes, respectively.

Example output:

```
➜ python .\MIOpen-Check.py
Package installer: uv
  gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036
  gfx1100, gfx1101, gfx1102, gfx1103, gfx110x, gfx1150, gfx1151, gfx1152, gfx1153, gfx115x
  gfx12-0, gfx1200, gfx1201, gfx1250
  gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx942, gfx950
Supported architectures:
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
OK = all required packages found; MISSING = one or more unavailable; UNKNOWN = index check failed.
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
  Measured latency   3.924 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.411 ms
  Torch / included   0.89x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.922 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.238 ms
  Torch / included   1.70x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.760 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   6.082 ms
  Torch / included   1.44x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.562 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.195 ms
  Torch / included   0.96x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   23.373 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   15.071 ms
  Torch / included   1.55x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.665 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.777 ms
  Torch / included   0.94x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.513 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.228 ms
  Torch / included   1.58x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.467 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.529 ms
  Torch / included   1.37x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.590 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.987 ms
  Torch / included   0.90x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   7.056 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.509 ms
  Torch / included   1.56x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   0.922 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.079 ms
  Torch / included   0.85x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.315 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.351 ms
  Torch / included   1.71x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.279 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.464 ms
  Torch / included   1.56x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.518 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.610 ms
  Torch / included   0.94x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   4.065 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.858 ms
  Torch / included   2.19x
Running nightly normal benchmark; recording nightly-results.txt

=== Nightly benchmark output (also saved to nightly-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.913 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.267 ms
  Torch / included   0.92x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.306 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.367 ms
  Torch / included   1.55x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.519 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.950 ms
  Torch / included   1.43x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.327 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.097 ms
  Torch / included   0.95x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.731 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.942 ms
  Torch / included   1.52x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.629 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.888 ms
  Torch / included   0.86x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.484 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.370 ms
  Torch / included   1.47x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.551 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.709 ms
  Torch / included   1.31x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.570 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.894 ms
  Torch / included   0.92x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   6.787 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.632 ms
  Torch / included   1.47x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.264 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.227 ms
  Torch / included   1.03x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.345 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.368 ms
  Torch / included   1.71x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.330 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.560 ms
  Torch / included   1.49x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.539 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.727 ms
  Torch / included   0.89x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.921 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.928 ms
  Torch / included   2.03x

==============================================================================
Torchvision validation
Torchvision validation (stable): PASS - torch=2.13.0+rocm10.0.0, torchvision=0.28.0+rocm10.0.0, amd-torchvision-device=0.28.0+rocm10.0.0; CPU and GPU NMS passed
Torchvision validation (nightly): PASS - torch=2.16.0a0+rocm10.3.0a20261031, torchvision=0.30.0a0+rocm10.3.0a20261031, amd-torchvision-device=0.30.0a0+rocm10.3.0a20261031; CPU and GPU NMS passed
==============================================================================

==============================================================================
Stable / nightly benchmark summary
  Matched configurations 15
  Stable total / average   88.931 ms / 5.929 ms
  Nightly total / average  87.216 ms / 5.814 ms
  Preferred packages       nightly
  Comparison report        E:\ROCm-Check\comparison.md

Recommended pip install command:
python -m pip install --upgrade --pre --index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261031 amd-torchvision-device-gfx1030==0.30.0a0+rocm10.3.0a20261031 rocm==10.3.0a20261031 rocm-bootstrap==0.3.0 rocm-sdk-core==10.3.0a20261031 rocm-sdk-devel==10.3.0a20261101 rocm-sdk-device-gfx1030==10.3.0a20261031 rocm-sdk-libraries==10.3.0a20261031 torch==2.16.0a0+rocm10.3.0a20261031 torchaudio==2.11.1+rocm10.3.0a20261031 torchvision==0.30.0a0+rocm10.3.0a20261031
==============================================================================
```