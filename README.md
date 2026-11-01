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

## 1. Package Version Differences (`pip list`)

| Package / Environment | Document 1 (`test1.txt`) | Document 2 (`test2.txt`) | Notes |
| --- | --- | --- | --- |
| `torch` | `2.13.0+rocm10.0.0` | `2.16.0a0+rocm10.3.0a20261031` | Doc 2 running newer dev build |
| `torchaudio` | `2.11.0.3+rocm10.1.0` | `2.11.1+rocm10.3.0a20261031` | Updated patch & ROCm alignment |
| `torchvision` | `0.28.0+rocm10.0.0` | `0.30.0a0+rocm10.3.0a20261031` | Doc 2 running newer dev build |
| `rocm` | `10.0.0` | `10.3.0a20261031` | ROCm core stack upgraded |
| `rocm-sdk-devel` | `10.2.0a20260926` | `10.3.0a20261031` | Build environment upgraded |
| `amd-torch-device-gfx1030` | `2.13.0+rocm10.0.0` | `2.16.0a0+rocm10.3.0a20261031` | Matching PyTorch build target |
| `amd-torchvision-device-gfx1030` | *Not present* | `0.30.0a0+rocm10.3.0a20261031` | Added in Doc 2 environment |

---

## 2. Solver Latency Comparison (Native MIOpen Torch)

Below is the side-by-side native measured latency across all 15 convolution configurations:

| Benchmark Config | Doc 1 Latency (ms) | Doc 2 Run 1 (ms) | Doc 2 Run 2 (ms) | Doc 2 Avg (ms) | Net Diff (Avg) |
| --- | --- | --- | --- | --- | --- |
| `ch256-k(3,3)-d(1,1)-0` | 3.813 | 3.754 | 3.739 | 3.747 | \-0.066 ms (-1.73%) |
| `ch256-k(3,3)-d(2,2)-1` | 8.430 | 8.134 | 8.178 | 8.156 | \-0.274 ms (-3.25%) |
| `ch256-k(3,3)-d(3,3)-2` | 8.451 | 8.370 | 8.405 | 8.388 | \-0.063 ms (-0.75%) |
| `ch256-k(5,5)-d(1,1)-0` | 13.371 | 12.742 | 13.178 | 12.960 | \-0.411 ms (-3.07%) |
| `ch256-k(5,5)-d(2,2)-1` | 22.739 | 20.923 | 22.480 | 21.702 | \-1.037 ms (-4.56%) |
| `ch128-k(3,3)-d(1,1)-0` | 1.477 | 1.475 | 1.539 | 1.507 | +0.030 ms (+2.03%) |
| `ch128-k(3,3)-d(2,2)-1` | 3.435 | 3.268 | 3.257 | 3.263 | \-0.172 ms (-5.01%) |
| `ch128-k(3,3)-d(3,3)-2` | 3.492 | 3.303 | 3.324 | 3.314 | \-0.178 ms (-5.10%) |
| `ch128-k(5,5)-d(1,1)-0` | 3.613 | 3.544 | 3.514 | 3.529 | \-0.084 ms (-2.32%) |
| `ch128-k(5,5)-d(2,2)-1` | 7.058 | 6.655 | 6.633 | 6.644 | \-0.414 ms (-5.87%) |
| `ch64-k(3,3)-d(1,1)-0` | 0.935 | 0.945 | 0.953 | 0.949 | +0.014 ms (+1.50%) |
| `ch64-k(3,3)-d(2,2)-1` | 2.272 | 2.227 | 2.150 | 2.189 | \-0.083 ms (-3.65%) |
| `ch64-k(3,3)-d(3,3)-2` | 2.217 | 2.231 | 2.227 | 2.229 | +0.012 ms (+0.54%) |
| `ch64-k(5,5)-d(1,1)-0` | 1.380 | 1.457 | 1.413 | 1.435 | +0.055 ms (+3.99%) |
| `ch64-k(5,5)-d(2,2)-1` | 3.857 | 3.789 | 3.817 | 3.803 | \-0.054 ms (-1.40%) |
| **TOTAL** | **86.545 ms** | **82.817 ms** | **84.807 ms** | **83.812 ms** | **-2.733 ms (-3.16%)** |

---

## 3. Findings & Calculated Percentage Gain

- **Best Performing Set:** **Document 2 (**`test2.txt`**)** using **PyTorch 2.16.0a0 + ROCm 10.3.0**.
- **Total Latency Gain (Run 1):**$$\\text{Gain} = \\frac{86.545\\text{ ms} - 82.817\\text{ ms}}{86.545\\text{ ms}} \\times 100 = \\mathbf{4.31%}$$
- **Total Latency Gain (Averaged across Run 1 & Run 2):**$$\\text{Gain} = \\frac{86.545\\text{ ms} - 83.812\\text{ ms}}{86.545\\text{ ms}} \\times 100 = \\mathbf{3.16%}$$

### Summary

The updated ROCm 10.3.0 and PyTorch 2.16 stack in Document 2 yields up to a **4.31% overall reduction in native solver latency** across the benchmark suite compared to Document 1.