# AMD ROCm MIOpen Analytics Platform for Windows

# ROCm & PyTorch Performance & Package Comparison

## Run a stable/nightly comparison

Run `python MIOpen-Check.py` and enter a supported GFX architecture.

The runner creates `venv-pytorch` and `venv-nightly`, runs the normal benchmark in each environment, and writes `pytorch-results.md`, `nightly-results.md`, and `comparison.md` in the repository root. On Linux, `venv-pytorch` uses the official PyTorch ROCm index; other supported platforms retain AMD's stable index under that environment name.

Shared Python dependencies are installed from PyPI. On Linux, the PyTorch environment uses the official PyTorch ROCm wheel index, which mirrors AMD's stable packages; the nightly environment uses AMD's nightly wheel indexes.

For gfx1010–gfx1012 and gfx1030–gfx1036, the official PyTorch environment uses the available ROCm 10.0.0 builds.

Nightly runs remain unpinned and install the latest available packages. The benchmark installs torchvision and its architecture-specific AMD package in both environments.

A pip recommendation is shown only when both environments pass torchvision CPU/GPU NMS checks and the selected package inventory includes those matching packages.

Before prompting, the selector checks all architectures against the official PyTorch and nightly indexes for the current Python and platform.

Benchmark output is streamed to the terminal and recorded in each results file. At completion, the runner prints matched PyTorch/nightly total and average latency and, when runtime validation succeeds, a pip install command for the lower-average environment.

Use `python MIOpen-Check.py --mode verbose` to capture the verbose benchmark instead. The environments are installed from the official PyTorch and AMD nightly wheel indexes, respectively.

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
AMD package availability (PyTorch + nightly):
  gfx10: gfx1010, gfx1011, gfx1012, gfx1030, gfx1031, gfx1032, gfx1033, gfx1034, gfx1035, gfx1036
  gfx11: gfx1100, gfx1101, gfx1102, gfx1103, gfx110x, gfx1150, gfx1151, gfx1152, gfx1153, gfx115x
  gfx12: gfx12-0, gfx1200, gfx1201, gfx1250
  gfx9: gfx900, gfx906, gfx908, gfx90a, gfx90c, gfx942, gfx950
Enter the GFX architecture (for example, gfx1030): gfx1030
Installing pytorch requirements for gfx1030
Using Python 3.12.9 environment at: venv-pytorch
Audited 14 packages in 7ms
Using Python 3.12.9 environment at: venv-pytorch
Audited 11 packages in 8ms
Installing nightly requirements for gfx1030
Using Python 3.12.9 environment at: venv-nightly
Audited 14 packages in 7ms
Using Python 3.12.9 environment at: venv-nightly
Audited 11 packages in 9ms

Running pytorch normal benchmark; recording pytorch-results.md

=== Pytorch benchmark output (also saved to pytorch-results.md) ===
=======================================================

conv2d ch256-k(3, 3)-d(1, 1)-0

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           3.831 ms

=======================================================

conv2d ch256-k(3, 3)-d(2, 2)-1

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           8.384 ms

=======================================================

conv2d ch256-k(3, 3)-d(3, 3)-2

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           8.591 ms

=======================================================

conv2d ch256-k(5, 5)-d(1, 1)-0

A 2D convolution with 256 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           13.155 ms

=======================================================

conv2d ch256-k(5, 5)-d(2, 2)-1

A 2D convolution with 256 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           22.866 ms

=======================================================

conv2d ch128-k(3, 3)-d(1, 1)-0

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.665 ms

=======================================================

conv2d ch128-k(3, 3)-d(2, 2)-1

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.378 ms

=======================================================

conv2d ch128-k(3, 3)-d(3, 3)-2

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.456 ms

=======================================================

conv2d ch128-k(5, 5)-d(1, 1)-0

A 2D convolution with 128 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           3.604 ms

=======================================================

conv2d ch128-k(5, 5)-d(2, 2)-1

A 2D convolution with 128 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           7.151 ms

=======================================================

conv2d ch64-k(3, 3)-d(1, 1)-0

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.003 ms

=======================================================

conv2d ch64-k(3, 3)-d(2, 2)-1

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           2.354 ms

=======================================================

conv2d ch64-k(3, 3)-d(3, 3)-2

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           2.264 ms

=======================================================

conv2d ch64-k(5, 5)-d(1, 1)-0

A 2D convolution with 64 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.522 ms

=======================================================

conv2d ch64-k(5, 5)-d(2, 2)-1

A 2D convolution with 64 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.883 ms


Running nightly normal benchmark; recording nightly-results.md

=== Nightly benchmark output (also saved to nightly-results.md) ===
=======================================================

conv2d ch256-k(3, 3)-d(1, 1)-0

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           3.759 ms

=======================================================

conv2d ch256-k(3, 3)-d(2, 2)-1

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           8.192 ms

=======================================================

conv2d ch256-k(3, 3)-d(3, 3)-2

A 2D convolution with 256 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           8.421 ms

=======================================================

conv2d ch256-k(5, 5)-d(1, 1)-0

A 2D convolution with 256 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           12.970 ms

=======================================================

conv2d ch256-k(5, 5)-d(2, 2)-1

A 2D convolution with 256 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 256, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           22.568 ms

=======================================================

conv2d ch128-k(3, 3)-d(1, 1)-0

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.608 ms

=======================================================

conv2d ch128-k(3, 3)-d(2, 2)-1

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.293 ms

=======================================================

conv2d ch128-k(3, 3)-d(3, 3)-2

A 2D convolution with 128 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.431 ms

=======================================================

conv2d ch128-k(5, 5)-d(1, 1)-0

A 2D convolution with 128 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           3.620 ms

=======================================================

conv2d ch128-k(5, 5)-d(2, 2)-1

A 2D convolution with 128 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 128, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           6.653 ms

=======================================================

conv2d ch64-k(3, 3)-d(1, 1)-0

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.011 ms

=======================================================

conv2d ch64-k(3, 3)-d(2, 2)-1

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           2.309 ms

=======================================================

conv2d ch64-k(3, 3)-d(3, 3)-2

A 2D convolution with 64 channels, a 3x3 kernel, and dilation 3x3.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           2.182 ms

=======================================================

conv2d ch64-k(5, 5)-d(1, 1)-0

A 2D convolution with 64 channels, a 5x5 kernel, and dilation 1x1.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  ConvBinWinogradRxSf2x3g1

  Measured latency:           1.543 ms

=======================================================

conv2d ch64-k(5, 5)-d(2, 2)-1

A 2D convolution with 64 channels, a 5x5 kernel, and dilation 2x2.

MIOpen:
  input:                      (8, 64, 128, 128)
  torch.dtype:                torch.float16
  Algorithm:                  GemmFwdRest

  Measured latency:           3.890 ms


==============================================================================
Torchvision validation
Torchvision validation (stable): PASS - torch=2.13.0+rocm10.0.0, torchvision=0.28.0+rocm10.0.0, amd-torchvision-device=0.28.0+rocm10.0.0; CPU and GPU NMS passed
Torchvision validation (nightly): PASS - torch=2.16.0a0+rocm10.3.0a20261101, torchvision=0.30.0a0+rocm10.3.0a20261101, amd-torchvision-device=0.30.0a0+rocm10.3.0a20261101; CPU and GPU NMS passed
==============================================================================

==============================================================================
Stable / nightly benchmark summary
  Matched configurations 15
  Stable total / average   87.107 ms / 5.807 ms
  Nightly total / average  85.450 ms / 5.697 ms
  Preferred packages       nightly
  Comparison report        E:\ROCm-Check\comparison.md

Recommended pip install command:
python -m pip install --upgrade --pre --index-url https://nightly.repo.amd.com/rocm/pytorch/whl-next --extra-index-url https://nightly.repo.amd.com/rocm/core/whl-next amd-torch-device-gfx1030==2.16.0a0+rocm10.3.0a20261101 amd-torchvision-device-gfx1030==0.30.0a0+rocm10.3.0a20261101 rocm==10.3.0a20261101 rocm-bootstrap==0.3.0 rocm-sdk-core==10.3.0a20261101 rocm-sdk-devel==10.3.0a20261102 rocm-sdk-device-gfx1030==10.3.0a20261101 rocm-sdk-libraries==10.3.0a20261101 torch==2.16.0a0+rocm10.3.0a20261101 torchaudio==2.11.1+rocm10.3.0a20261101 torchvision==0.30.0a0+rocm10.3.0a20261101
==============================================================================
```