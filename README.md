# AMD ROCm MIOpen Analytics Platform

# ROCm & PyTorch Performance & Package Comparison

## Run a stable/nightly comparison

Run `python MIOpen-Check.py` and enter a supported GFX architecture (the package name
`amd-torch-device-gfx...` is also accepted). The runner creates `venv-stable` and
`venv-nightly`, installs the requirements from the corresponding AMD repository, runs the
normal benchmark in each environment, and writes `stable-results.txt`,
`nightly-results.txt`, and `comparison.md` in the repository root.
Shared Python dependencies are installed from PyPI; ROCm and PyTorch packages are
resolved independently from each repository's PyTorch and core wheel indexes so
their versions match that channel.
For stable runs on gfx1010–gfx1012 and gfx1030–gfx1036, the runner pins Torch and ROCm
packages to the available ROCm 10.0.0 builds. Nightly runs remain unpinned and install
the latest available packages.
When `uv` is available, it handles installation with a concise progress display; otherwise
the runner falls back to pip. Benchmark output is streamed to the terminal and recorded
in each results file. At completion, the runner prints matched stable/nightly total and
average latency and a pip install command for the lower-average environment.

Use `python MIOpen-Check.py --mode verbose` to capture the verbose benchmark instead.
Both environments are installed from the stable and nightly AMD wheel indexes,
respectively.

Example output:
```shell
➜ python .\MIOpen-Check.py
Package installer: uv
Supported architectures: gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx110x, gfx1150, gfx1151, gfx1152, gfx1153, gfx115x, gfx12-0, gfx1200, gfx1201, gfx1250, gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx942, gfx950
Enter the GFX architecture (for example, gfx1030): gfx1030
Pinning stable ROCm packages to 10.0.0 for gfx1030
Installing stable requirements for gfx1030
Using Python 3.12.9 environment at: venv-stable
Audited 14 packages in 6ms
Using Python 3.12.9 environment at: venv-stable
Audited 9 packages in 7ms
Installing nightly requirements for gfx1030
Using Python 3.12.9 environment at: venv-nightly
Audited 14 packages in 5ms
Using Python 3.12.9 environment at: venv-nightly
Audited 9 packages in 7ms
Running stable normal benchmark; recording stable-results.txt

=== Stable benchmark output (also saved to stable-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.925 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.379 ms
  Torch / included   0.90x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.532 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.994 ms
  Torch / included   1.71x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.740 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   6.105 ms
  Torch / included   1.43x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.631 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.269 ms
  Torch / included   0.96x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   23.180 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   15.091 ms
  Torch / included   1.54x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.701 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.736 ms
  Torch / included   0.98x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.659 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.198 ms
  Torch / included   1.66x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.620 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.558 ms
  Torch / included   1.41x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.680 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.148 ms
  Torch / included   0.89x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   7.269 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.575 ms
  Torch / included   1.59x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   0.885 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.049 ms
  Torch / included   0.84x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.232 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.277 ms
  Torch / included   1.75x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.383 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.517 ms
  Torch / included   1.57x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.534 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.703 ms
  Torch / included   0.90x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   4.036 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.959 ms
  Torch / included   2.06x
Running nightly normal benchmark; recording nightly-results.txt

=== Nightly benchmark output (also saved to nightly-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.744 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.383 ms
  Torch / included   0.85x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.335 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.016 ms
  Torch / included   1.66x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.618 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.834 ms
  Torch / included   1.48x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.562 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.033 ms
  Torch / included   0.97x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.937 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   15.014 ms
  Torch / included   1.53x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.580 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.742 ms
  Torch / included   0.91x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.446 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.294 ms
  Torch / included   1.50x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.637 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.532 ms
  Torch / included   1.44x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.492 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.989 ms
  Torch / included   0.88x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   6.710 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.384 ms
  Torch / included   1.53x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.030 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.100 ms
  Torch / included   0.94x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.305 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.275 ms
  Torch / included   1.81x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.291 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.507 ms
  Torch / included   1.52x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.499 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.715 ms
  Torch / included   0.87x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.793 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.910 ms
  Torch / included   1.99x

==============================================================================
Stable / nightly benchmark summary
  Matched configurations 15
  Stable total / average   89.007 ms / 5.934 ms
  Nightly total / average  86.979 ms / 5.799 ms
  Preferred packages       nightly
  Comparison report        E:\ROCm-Check\comparison.md

Recommended pip install command:
python -m pip install --pre --index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261031 rocm==10.3.0a20261031 rocm-bootstrap==0.3.0 rocm-sdk-core==10.3.0a20261031 rocm-sdk-devel==10.3.0a20261031 rocm-sdk-device-gfx1030==10.3.0a20261031 rocm-sdk-libraries==10.3.0a20261031 torch==2.16.0a0+rocm10.3.0a20261031 torchaudio==2.11.1+rocm10.3.0a20261031
==============================================================================
```
