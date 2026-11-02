import os
import operator
import re
import threading
import time

# -----------------------------------------------------------------------------
# Benchmark metadata and suite switches
# -----------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE = "float16"
RUN_CONV1D = False
RUN_CONV2D = True
RUN_CONV3D = False
CONV1D_BATCH_SIZE = 128
CONV1D_CHANNELS = [512, 256, 128, 64, 32]
CONV1D_LENGTH = 400
CONV2D_BATCH_SIZE = 8
CONV2D_CHANNELS = [256, 128, 64]
CONV2D_SIZE = (128, 128)
CONV3D_BATCH_SIZE = 2
CONV3D_CHANNELS = [128, 64, 32]
CONV3D_SIZE = (8, 64, 64)
WARMUP_RUNS = 2
BENCHMARK_RUNS = 5

CONV1D_CONFIGS = [
    (3, ((1, 1), (3, 3), (5, 5))),
    (7, ((3, 1), (9, 3), (15, 5))),
    (11, ((5, 1), (15, 3), (25, 5))),
]
CONV2D_CONFIGS = [
    ((3, 3), ((1, 1), (2, 2), (3, 3))),
    ((5, 5), ((1, 1), (2, 2))),
]
CONV3D_CONFIGS = [
    ((3, 3, 3), ((1, 1, 1), (2, 1, 1), (1, 2, 2))),
    ((3, 3, 1), ((1, 1, 1), (2, 1, 1))),
]

# -----------------------------------------------------------------------------
# MIOpen Environment Setup (must precede the Torch import)
# -----------------------------------------------------------------------------
os.environ["MIOPEN_FIND_MODE"] = "2"
os.environ["MIOPEN_FIND_ENFORCE"] = "1"
os.environ["ROCBLAS_LAYER"] = "0"
os.environ["PYTORCH_ROCM_USE_ROCBLAS"] = "0"
os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
os.environ["MIOPEN_ENABLE_LOGGING_ELAPSED_TIME"] = "1"
os.environ["MIOPEN_LOG_LEVEL"] = "5"
os.environ["MIOPEN_DEBUG_ENABLE"] = "0"
import torch
from torch import nn

torch.set_float32_matmul_precision('high')

_ALGORITHM_PATTERN = re.compile(
    r"FW Chosen Algorithm:\s*([^,\s]+)"
)

class _MIOpenStderrCapture:
    """Capture native DLL stderr without relying on Python redirect_stderr."""

    def __enter__(self):
        self.read_fd, write_fd = os.pipe()
        self.saved_stderr = os.dup(2)
        os.dup2(write_fd, 2)
        os.close(write_fd)
        self.lines = []

        def read_output():
            chunks = []
            while True:
                chunk = os.read(self.read_fd, 4096)
                if not chunk:
                    break
                chunks.append(chunk)
            self.lines.extend(b"".join(chunks).decode(errors="replace").splitlines())

        self.reader = threading.Thread(target=read_output, daemon=True)
        self.reader.start()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        os.dup2(self.saved_stderr, 2)
        os.close(self.saved_stderr)
        self.reader.join(timeout=2)
        os.close(self.read_fd)


def _extract_algorithms(lines):
    return [match.group(1) for line in lines if (match := _ALGORITHM_PATTERN.search(line))]


def _spatial_ints(values: str | tuple[int | str, ...], name: str) -> tuple[int, ...]:
    if isinstance(values, str):
        raise ValueError(f"Phased convolution requires numeric {name}")
    integers = []
    for value in values:
        if isinstance(value, str):
            raise ValueError(f"Phased convolution requires numeric {name}")
        integers.append(operator.index(value))
    return tuple(integers)


def _format_spatial(values: tuple[int, ...]) -> str:
    return "x".join(str(value) for value in values)


_dilated_use_phases = {}

def _conv1d_phases(x: torch.Tensor, conv: torch.nn.Conv1d):
    d, = _spatial_ints(conv.dilation, "dilation")
    pad, = _spatial_ints(conv.padding, "padding")
    length = x.shape[-1]
    x = torch.nn.functional.pad(x, (pad, pad))
    # phases must divide evenly; the remainder is padded off the end, so it only ever lands
    # beyond `length` and is dropped by the final slice
    rem = (-x.shape[-1]) % d
    if rem:
        x = torch.nn.functional.pad(x, (0, rem))
    n, c, total = x.shape
    phases = x.view(n, c, total // d, d).permute(0, 3, 1, 2).reshape(n * d, c, total // d)
    out = torch.nn.functional.conv1d(phases, conv.weight, conv.bias)
    chan, per = conv.weight.shape[0], out.shape[-1]
    out = out.view(n, d, chan, per).permute(0, 2, 3, 1).reshape(n, chan, per * d)
    return out[..., :length]


def _conv2d_phases(x: torch.Tensor, conv: torch.nn.Conv2d):
    stride_h, stride_w = _spatial_ints(conv.stride, "stride")
    dilation_h, dilation_w = _spatial_ints(conv.dilation, "dilation")
    padding_h, padding_w = _spatial_ints(conv.padding, "padding")
    kernel_h, kernel_w = _spatial_ints(conv.kernel_size, "kernel size")
    height, width = x.shape[-2:]

    # 1. Pad input tensor
    x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h))

    # 2. Pad remainder to cleanly divide by dilation factors
    rem_h = (-x.shape[-2]) % dilation_h
    rem_w = (-x.shape[-1]) % dilation_w
    if rem_h or rem_w:
        x = torch.nn.functional.pad(x, (0, rem_w, 0, rem_h))

    batch, channels, padded_h, padded_w = x.shape

    # 3. Decompose input into polyphase sub-grids
    phases = (
        x.view(batch, channels, padded_h // dilation_h, dilation_h, padded_w // dilation_w, dilation_w)
        .permute(0, 3, 5, 1, 2, 4)
        .reshape(batch * dilation_h * dilation_w, channels, padded_h // dilation_h, padded_w // dilation_w)
    )

    # 4. Standard un-dilated convolution on sub-sampled phases
    # Note: Stride remains 1 for inner conv because striding is applied on the reconstructed output
    out = torch.nn.functional.conv2d(phases, conv.weight, conv.bias, groups=conv.groups)

    # 5. Calculate target output shape following standard PyTorch Conv2d spatial formulas
    phase_h, phase_w = out.shape[-2:]
    channels_out = conv.weight.shape[0]

    # 6. Reinterlace phase outputs into full spatial grid
    out = (
        out.view(batch, dilation_h, dilation_w, channels_out, phase_h, phase_w)
        .permute(0, 3, 4, 1, 5, 2)
        .reshape(batch, channels_out, phase_h * dilation_h, phase_w * dilation_w)
    )

    # 7. Apply spatial slice matching stride-3 (or arbitrary stride) sub-sampling
    return out[..., :height + 2 * padding_h - dilation_h * (kernel_h - 1):stride_h,
                    :width + 2 * padding_w - dilation_w * (kernel_w - 1):stride_w]


def _conv3d_phases(x: torch.Tensor, conv: torch.nn.Conv3d):
    """Compute a dilated Conv3d with arbitrary stride as independent undilated phases."""
    stride_d, stride_h, stride_w = _spatial_ints(conv.stride, "stride")
    dilation_d, dilation_h, dilation_w = _spatial_ints(conv.dilation, "dilation")
    padding_d, padding_h, padding_w = _spatial_ints(conv.padding, "padding")
    kernel_d, kernel_h, kernel_w = _spatial_ints(conv.kernel_size, "kernel size")
    depth, height, width = x.shape[-3:]

    # 1. Pad input tensor
    x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h, padding_d, padding_d))

    # 2. Pad remainder to cleanly divide by dilation factors
    rem_d = (-x.shape[-3]) % dilation_d
    rem_h = (-x.shape[-2]) % dilation_h
    rem_w = (-x.shape[-1]) % dilation_w
    if rem_d or rem_h or rem_w:
        x = torch.nn.functional.pad(x, (0, rem_w, 0, rem_h, 0, rem_d))

    batch, channels, padded_d, padded_h, padded_w = x.shape

    # 3. Decompose input into polyphase sub-grids across depth, height, and width
    phases = (
        x.view(
            batch,
            channels,
            padded_d // dilation_d,
            dilation_d,
            padded_h // dilation_h,
            dilation_h,
            padded_w // dilation_w,
            dilation_w,
        )
        .permute(0, 3, 5, 7, 1, 2, 4, 6)
        .reshape(batch * dilation_d * dilation_h * dilation_w, channels, padded_d // dilation_d, padded_h // dilation_h, padded_w // dilation_w)
    )

    # 4. Standard un-dilated convolution on sub-sampled phases (inner stride remains 1)
    out = torch.nn.functional.conv3d(phases, conv.weight, conv.bias, groups=conv.groups)

    # 5. Calculate target output shape following standard PyTorch Conv3d spatial formulas
    phase_d, phase_h, phase_w = out.shape[-3:]
    channels_out = conv.weight.shape[0]

    # 6. Reinterlace phase outputs into full spatial grid
    out = (
        out.view(batch, dilation_d, dilation_h, dilation_w, channels_out, phase_d, phase_h, phase_w)
        .permute(0, 4, 5, 1, 6, 2, 7, 3)
        .reshape(batch, channels_out, phase_d * dilation_d, phase_h * dilation_h, phase_w * dilation_w)
    )

    # 7. Compute full un-strided dimensions and apply strided slicing
    full_d = depth + 2 * padding_d - dilation_d * (kernel_d - 1)
    full_h = height + 2 * padding_h - dilation_h * (kernel_h - 1)
    full_w = width + 2 * padding_w - dilation_w * (kernel_w - 1)

    return out[..., :full_d:stride_d, :full_h:stride_h, :full_w:stride_w]

def conv1d_dilated(x: torch.Tensor, conv: torch.nn.Conv1d):
    if conv.dilation[0] == 1 or not x.is_cuda:
        return conv(x)

    key = (
        x.device,
        x.dtype,
        tuple(x.shape),
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
    )
    use_phases = _dilated_use_phases.get(key)
    if use_phases is None:
        def timed(fn):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            return time.perf_counter() - t0

        with torch.no_grad():
            native, phases = timed(lambda: conv(x)), timed(lambda: _conv1d_phases(x, conv))
        # require a clear margin so timing noise cannot pick the rewrite on a card that is fine
        use_phases = phases * 1.5 < native
        _dilated_use_phases[key] = use_phases
    return _conv1d_phases(x, conv) if use_phases else conv(x)

def benchmark(fn, warmup=2, repeats=5):
    with _MIOpenStderrCapture() as capture, torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / repeats
    return elapsed_ms, _extract_algorithms(capture.lines)


def _dtype():
    return getattr(torch, DTYPE)


def print_native_result(name, description, native_ms, native_algorithms, inputs):
    cyan = "\033[36m"
    green = "\033[92m"
    yellow = "\033[93m"
    magenta = "\033[95m"
    dim = "\033[2m"
    reset = "\033[0m"
    algorithm = native_algorithms[-1] if native_algorithms else "not emitted (cached or unavailable)"
    print(f"{cyan}{'=' * 55}{reset}")
    print()
    print(f"{yellow}{name}{reset}")
    print()
    print(f"{dim}{description}{reset}")
    print()
    print(f"{magenta}MIOpen:{reset}")
    print(f"  {dim}{'input:':<28}{reset}{tuple(inputs.shape)}")
    print(f"  {dim}{'torch.dtype:':<28}{reset}{inputs.dtype}")
    print(f"  {dim}{'Algorithm:':<28}{reset}{green}{algorithm}{reset}")
    print()
    print(f"  {dim}{'Measured latency:':<28}{reset}{cyan}{native_ms:.3f} ms{reset}")
    print()


def run_conv1d_suite():
    for channels in CONV1D_CHANNELS:
        for kernel_size, padding_dilations in CONV1D_CONFIGS:
            for index, (padding, dilation) in enumerate(padding_dilations):
                test_input = torch.randn(CONV1D_BATCH_SIZE, channels, CONV1D_LENGTH, device=DEVICE, dtype=_dtype())
                test_conv = nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
                ).to(DEVICE, dtype=_dtype())
                name = f"conv1d ch{channels}-k{kernel_size}-{index}"
                description = (
                    f"A 1D convolution with {channels} channels, a {kernel_size}-element kernel, "
                    f"padding {padding}, and dilation {dilation}."
                )
                native_ms, native_algorithms = benchmark(
                    lambda conv=test_conv, inputs=test_input: conv(inputs),
                    WARMUP_RUNS,
                    BENCHMARK_RUNS,
                )
                print_native_result(name, description, native_ms, native_algorithms, test_input)


def run_conv2d_suite():
    for channels in CONV2D_CHANNELS:
        for kernel_size, dilations in CONV2D_CONFIGS:
            for index, dilation in enumerate(dilations):
                padding = tuple((size // 2) * value for size, value in zip(kernel_size, dilation))
                test_input = torch.randn(CONV2D_BATCH_SIZE, channels, *CONV2D_SIZE, device=DEVICE, dtype=_dtype())
                test_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation).to(DEVICE, dtype=_dtype())
                name = f"conv2d ch{channels}-k{kernel_size}-d{dilation}-{index}"
                description = (
                    f"A 2D convolution with {channels} channels, a {_format_spatial(kernel_size)} kernel, "
                    f"and dilation {_format_spatial(dilation)}."
                )
                native_ms, native_algorithms = benchmark(
                    lambda conv=test_conv, inputs=test_input: conv(inputs),
                    WARMUP_RUNS,
                    BENCHMARK_RUNS,
                )
                print_native_result(name, description, native_ms, native_algorithms, test_input)


def run_conv3d_suite():
    for channels in CONV3D_CHANNELS:
        for kernel_size, dilations in CONV3D_CONFIGS:
            for index, dilation in enumerate(dilations):
                padding = tuple((size // 2) * value for size, value in zip(kernel_size, dilation))
                test_input = torch.randn(CONV3D_BATCH_SIZE, channels, *CONV3D_SIZE, device=DEVICE, dtype=_dtype())
                test_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation).to(DEVICE, dtype=_dtype())
                name = f"conv3d ch{channels}-k{kernel_size}-d{dilation}-{index}"
                description = (
                    f"A 3D convolution with {channels} channels, a {_format_spatial(kernel_size)} kernel, "
                    f"and dilation {_format_spatial(dilation)}."
                )
                native_ms, native_algorithms = benchmark(
                    lambda conv=test_conv, inputs=test_input: conv(inputs),
                    WARMUP_RUNS,
                    BENCHMARK_RUNS,
                )
                print_native_result(name, description, native_ms, native_algorithms, test_input)


if __name__ == "__main__":
    if RUN_CONV1D:
        run_conv1d_suite()
    if RUN_CONV2D:
        run_conv2d_suite()
    if RUN_CONV3D:
        run_conv3d_suite()
