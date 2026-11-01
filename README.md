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
Supported architectures: gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036, gfx1100, gfx1101, gfx1102, gfx1103, gfx110x, gfx1150, gfx1151, gfx1152, gfx1153, gfx115x, gfx12-0, gfx1200, gfx1201, gfx1250, gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx942, gfx950
Checking required AMD package artifacts for Python 3.12 (win-amd64)...
AMD package availability by architecture (checks compatible wheels and the ROCm source archive in stable and nightly indexes for this Python/platform):
  gfx1010  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1011  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1012  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1030  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1031  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1032  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1033  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1034  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1035  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1036  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1100  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1101  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1102  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1103  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx110x  [PACKAGES MISSING] stable: Torch OK, SDK missing | nightly: Torch OK, SDK missing - stable rocm-sdk-device-gfx110x: package not published; stable amd-torchvision-device-gfx110x: package not published; nightly rocm-sdk-device-gfx110x: package not published; and 1 more
  gfx1150  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1151  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1152  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1153  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx115x  [PACKAGES MISSING] stable: Torch OK, SDK missing | nightly: Torch OK, SDK missing - stable rocm-sdk-device-gfx115x: package not published; stable amd-torchvision-device-gfx115x: package not published; nightly rocm-sdk-device-gfx115x: package not published; and 1 more
  gfx12-0  [PACKAGES MISSING] stable: Torch OK, SDK missing | nightly: Torch OK, SDK missing - stable rocm-sdk-device-gfx12-0: package not published; stable amd-torchvision-device-gfx12-0: package not published; nightly rocm-sdk-device-gfx12-0: package not published; and 1 more
  gfx1200  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1201  [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx1250  [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch missing, SDK missing - stable amd-torch-device-gfx1250: no compatible wheel published for this Python/platform; stable rocm-sdk-device-gfx1250: no compatible wheel published for this Python/platform; stable amd-torchvision-device-gfx1250: no compatible wheel published for this Python/platform; and 3 more
  gfx900   [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch OK, SDK OK - stable amd-torch-device-gfx900: package not published; stable rocm-sdk-device-gfx900: package not published; stable amd-torchvision-device-gfx900: package not published
  gfx906   [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch OK, SDK OK - stable amd-torch-device-gfx906: package not published; stable rocm-sdk-device-gfx906: package not published; stable amd-torchvision-device-gfx906: package not published
  gfx908   [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx90a   [PACKAGE SET OK] stable: Torch OK, SDK OK | nightly: Torch OK, SDK OK
  gfx90c   [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch OK, SDK OK - stable amd-torch-device-gfx90c: package not published; stable rocm-sdk-device-gfx90c: package not published; stable amd-torchvision-device-gfx90c: package not published
  gfx942   [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch missing, SDK missing - stable amd-torch-device-gfx942: no compatible wheel published for this Python/platform; stable rocm-sdk-device-gfx942: no compatible wheel published for this Python/platform; stable amd-torchvision-device-gfx942: no compatible wheel published for this Python/platform; and 3 more
  gfx950   [PACKAGES MISSING] stable: Torch missing, SDK missing | nightly: Torch missing, SDK missing - stable amd-torch-device-gfx950: no compatible wheel published for this Python/platform; stable rocm-sdk-device-gfx950: no compatible wheel published for this Python/platform; stable amd-torchvision-device-gfx950: no compatible wheel published for this Python/platform; and 3 more
Green: required packages found. Red: one or more are missing. Yellow: index could not be checked.
This is package availability, not an official hardware support or runtime test. No architectures are disabled.
Enter the GFX architecture (for example, gfx1030): gfx1030
Pinning stable ROCm packages to 10.0.0 for gfx1030
Creating stable virtual environment at E:\ROCm-Check\venv-stable
Installing stable requirements for gfx1030
Using Python 3.12.9 environment at: venv-stable
Resolved 14 packages in 129ms
░░░░░░░░░░░░░░░░░░░░ [0/14] Installing wheels...                                                                                                                                                                                               warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 14 packages in 1.23s
 + filelock==3.32.6
 + fsspec==2026.7.0
 + jinja2==3.1.6
 + markdown-it-py==4.2.0
 + markupsafe==3.0.3
 + mdurl==0.1.2
 + mpmath==1.3.0
 + networkx==3.6.1
 + numpy==2.5.3
 + pygments==2.21.0
 + rich==15.0.0
 + setuptools==81.0.0
 + sympy==1.14.0
 + typing-extensions==4.16.0
Using Python 3.12.9 environment at: venv-stable
Resolved 22 packages in 3.46s
Prepared 3 packages in 2.26s
░░░░░░░░░░░░░░░░░░░░ [0/12] Installing wheels...                                                                                                                                                                                               warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 12 packages in 9.42s
 + amd-torch-device-gfx1030==2.13.0+rocm10.0.0
 + amd-torchvision-device-gfx1030==0.28.0+rocm10.0.0
 + pillow==12.3.0
 + rocm==10.0.0
 + rocm-bootstrap==0.3.0
 + rocm-sdk-core==10.0.0
 + rocm-sdk-devel==10.0.0
 + rocm-sdk-device-gfx1030==10.0.0
 + rocm-sdk-libraries==10.0.0
 + torch==2.13.0+rocm10.0.0
 + torchaudio==2.11.0.2+rocm10.0.0
 + torchvision==0.28.0+rocm10.0.0
Creating nightly virtual environment at E:\ROCm-Check\venv-nightly
Installing nightly requirements for gfx1030
Using Python 3.12.9 environment at: venv-nightly
Resolved 14 packages in 12ms
░░░░░░░░░░░░░░░░░░░░ [0/14] Installing wheels...                                                                                                                                                                                               warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 14 packages in 1.22s
 + filelock==3.32.6
 + fsspec==2026.7.0
 + jinja2==3.1.6
 + markdown-it-py==4.2.0
 + markupsafe==3.0.3
 + mdurl==0.1.2
 + mpmath==1.3.0
 + networkx==3.6.1
 + numpy==2.5.3
 + pygments==2.21.0
 + rich==15.0.0
 + setuptools==81.0.0
 + sympy==1.14.0
 + typing-extensions==4.16.0
Using Python 3.12.9 environment at: venv-nightly
Resolved 22 packages in 7.58s
Prepared 3 packages in 2.39s
░░░░░░░░░░░░░░░░░░░░ [0/12] Installing wheels...                                                                                                                                                                                               warning: Failed to hardlink files; falling back to full copy. This may lead to degraded performance.
         If the cache and target directories are on different filesystems, hardlinking may not be supported.
         If this is intentional, set `export UV_LINK_MODE=copy` or use `--link-mode=copy` to suppress this warning.
Installed 12 packages in 10.27s
 + amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261031
 + amd-torchvision-device-gfx1030==0.30.0a0+rocm10.3.0a20261031
 + pillow==12.3.0
 + rocm==10.3.0a20261031
 + rocm-bootstrap==0.3.0
 + rocm-sdk-core==10.3.0a20261031
 + rocm-sdk-devel==10.3.0a20261101
 + rocm-sdk-device-gfx1030==10.3.0a20261031
 + rocm-sdk-libraries==10.3.0a20261031
 + torch==2.16.0a0+rocm10.3.0a20261031
 + torchaudio==2.11.1+rocm10.3.0a20261031
 + torchvision==0.30.0a0+rocm10.3.0a20261031
Running stable normal benchmark; recording stable-results.txt

=== Stable benchmark output (also saved to stable-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.841 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.243 ms
  Torch / included   0.91x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.380 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.001 ms
  Torch / included   1.68x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.497 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   6.004 ms
  Torch / included   1.42x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.320 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.804 ms
  Torch / included   0.96x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.809 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.754 ms
  Torch / included   1.55x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.658 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.766 ms
  Torch / included   0.94x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.526 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.291 ms
  Torch / included   1.54x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.566 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.551 ms
  Torch / included   1.40x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.656 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.883 ms
  Torch / included   0.94x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   7.003 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.523 ms
  Torch / included   1.55x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   0.993 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.196 ms
  Torch / included   0.83x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.394 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.278 ms
  Torch / included   1.87x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.350 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.538 ms
  Torch / included   1.53x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.412 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.553 ms
  Torch / included   0.91x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   4.024 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.824 ms
  Torch / included   2.21x
Running nightly normal benchmark; recording nightly-results.txt

=== Nightly benchmark output (also saved to nightly-results.txt) ===
==============================================================================
conv2d ch256-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.904 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.174 ms
  Torch / included   0.94x
==============================================================================
conv2d ch256-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.245 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.998 ms
  Torch / included   1.65x
==============================================================================
conv2d ch256-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   8.385 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   5.894 ms
  Torch / included   1.42x
==============================================================================
conv2d ch256-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.147 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   13.862 ms
  Torch / included   0.95x
==============================================================================
conv2d ch256-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   22.578 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   14.788 ms
  Torch / included   1.53x
==============================================================================
conv2d ch128-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.605 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.807 ms
  Torch / included   0.89x
==============================================================================
conv2d ch128-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.351 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.217 ms
  Torch / included   1.51x
==============================================================================
conv2d ch128-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.459 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   2.549 ms
  Torch / included   1.36x
==============================================================================
conv2d ch128-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.567 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   3.769 ms
  Torch / included   0.95x
==============================================================================
conv2d ch128-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   6.675 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   4.316 ms
  Torch / included   1.55x
==============================================================================
conv2d ch64-k(3, 3)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   0.974 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.049 ms
  Torch / included   0.93x
==============================================================================
conv2d ch64-k(3, 3)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.329 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.391 ms
  Torch / included   1.67x
==============================================================================
conv2d ch64-k(3, 3)-d(3, 3)-2
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   2.368 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.465 ms
  Torch / included   1.62x
==============================================================================
conv2d ch64-k(5, 5)-d(1, 1)-0
native: miopen torch
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.411 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.555 ms
  Torch / included   0.91x
==============================================================================
conv2d ch64-k(5, 5)-d(2, 2)-1
native: miopen torch
  Algorithm          GemmFwdRest
  Measured latency   3.781 ms
phased included
  Algorithm          ConvBinWinogradRxSf2x3g1
  Measured latency   1.900 ms
  Torch / included   1.99x
Torchvision validation (stable): PASS - torch=2.13.0+rocm10.0.0, torchvision=0.28.0+rocm10.0.0, amd-torchvision-device=0.28.0+rocm10.0.0; CPU and GPU NMS passed
Torchvision validation (nightly): PASS - torch=2.16.0a0+rocm10.3.0a20261031, torchvision=0.30.0a0+rocm10.3.0a20261031, amd-torchvision-device=0.30.0a0+rocm10.3.0a20261031; CPU and GPU NMS passed

==============================================================================
Stable / nightly benchmark summary
  Matched configurations 15
  Stable total / average   87.429 ms / 5.829 ms
  Nightly total / average  85.779 ms / 5.719 ms
  Preferred packages       nightly
  Comparison report        E:\ROCm-Check\comparison.md

Recommended pip install command:
python -m pip install --pre --index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261031 amd-torchvision-device-gfx1030==0.30.0a0+rocm10.3.0a20261031 rocm==10.3.0a20261031 rocm-bootstrap==0.3.0 rocm-sdk-core==10.3.0a20261031 rocm-sdk-devel==10.3.0a20261101 rocm-sdk-device-gfx1030==10.3.0a20261031 rocm-sdk-libraries==10.3.0a20261031 torch==2.16.0a0+rocm10.3.0a20261031 torchaudio==2.11.1+rocm10.3.0a20261031 torchvision==0.30.0a0+rocm10.3.0a20261031
==============================================================================
```