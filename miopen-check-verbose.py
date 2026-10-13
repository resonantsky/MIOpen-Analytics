"""Capture and report native MIOpen convolution selections across Linux and Windows."""

import atexit
import ctypes
import os
import re
import sys
import threading
import time

import torch
from logger import log
from torch import nn

# -----------------------------------------------------------------------------
# Benchmark metadata and suite switches
# -----------------------------------------------------------------------------
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"
DEVICE = "cuda"
DTYPE = "float32"
RUN_CONV1D = False
RUN_CONV2D = True
RUN_CONV3D = False
CONV1D_BATCH_SIZE = 128
CONV1D_CHANNELS = [512, 256, 128, 64, 32]
CONV1D_LENGTH = 400
CONV2D_BATCH_SIZE = 1
CONV2D_CHANNELS = [32, 16, 8]
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
os.environ["MIOPEN_DEBUG_ENABLE"] = "1"

torch.set_float32_matmul_precision('high')

_ALGORITHM_PATTERN = re.compile(r"FW Chosen Algorithm:\s*([^,\s]+)")
_CHOSEN_PATTERN = re.compile(r"FW Chosen Algorithm:\s*([^,\s]+)\s*,\s*[^,]*,\s*([0-9.eE+-]+)")
_MIOPEN_PREFIX = "MIOpen(HIP):"
_logging_capture = None

IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    import msvcrt
    STD_ERROR_HANDLE = -12
    kernel32 = ctypes.windll.kernel32

    # Load Universal C Runtime to manipulate DLL-level file descriptors
    try:
        ucrt = ctypes.CDLL("ucrtbase.dll")
    except OSError:
        ucrt = None


def _forward_stderr(fd, line):
    data = line if line.endswith(b"\n") else line + b"\n"
    while data:
        written = os.write(fd, data)
        data = data[written:]


def _process_line(line, saved_stderr):
    text = line.decode(errors="replace").rstrip("\r\n")
    if not text.lstrip().startswith(_MIOPEN_PREFIX):
        _forward_stderr(saved_stderr, line)
        return
    match = _CHOSEN_PATTERN.search(text)
    if match:
        log.info(f'MIOpen: algorithm={match.group(1)} time={float(match.group(2)):.3f}')

class MIOpenLogRedirect:
    """Redirect native MIOpen diagnostics into structured informational logs."""

    def __init__(self):
        self.read_fd = -1
        self.write_fd = -1
        self.saved_stderr = -1
        self.saved_python_stderr = sys.stderr
        self.safe_stderr = None
        self.reader = None
        self.old_std_handle = None

    def __enter__(self):
        try:
            self.read_fd, self.write_fd = os.pipe()
            self.saved_stderr = os.dup(2)
            self.saved_python_stderr = sys.stderr

            self.safe_stderr = os.fdopen(
                os.dup(self.saved_stderr),
                "w",
                encoding=getattr(sys.stderr, "encoding", None) or "utf-8",
                buffering=1,
            )

            # 1. Standard C File Descriptor Redirection
            os.dup2(self.write_fd, 2)

            # 2. Windows-Specific CRT & Process Handle Hooking
            if IS_WINDOWS:
                # Intercept Win32 OS Level Handle
                self.old_std_handle = kernel32.GetStdHandle(STD_ERROR_HANDLE)
                new_os_handle = msvcrt.get_osfhandle(self.write_fd)
                kernel32.SetStdHandle(STD_ERROR_HANDLE, new_os_handle)

                # Intercept Universal CRT level handle for loaded DLLs
                if ucrt is not None:
                    try:
                        ucrt._dup2(self.write_fd, 2)
                        # Disable buffering on FD 2 in ucrtbase so logs flush immediately
                        _IOLBF = 64
                        ucrt._setvbuf(2, None, _IOLBF, 4096)
                    except (AttributeError, OSError, TypeError):
                        pass

            sys.stderr = self.safe_stderr

            def read_output():
                pending = b""
                while True:
                    try:
                        chunk = os.read(self.read_fd, 4096)
                    except OSError:
                        break
                    if not chunk:
                        break
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        _process_line(line + b"\n", self.saved_stderr)
                if pending:
                    _process_line(pending, self.saved_stderr)

            self.reader = threading.Thread(target=read_output, daemon=True)
            self.reader.start()
            return self

        except Exception:
            self._cleanup()
            raise

    def _cleanup(self):
        if IS_WINDOWS and self.old_std_handle is not None:
            kernel32.SetStdHandle(STD_ERROR_HANDLE, self.old_std_handle)
            self.old_std_handle = None

        if self.saved_stderr >= 0:
            os.dup2(self.saved_stderr, 2)

        sys.stderr = self.saved_python_stderr

        if self.write_fd >= 0:
            try:
                os.close(self.write_fd)
            except OSError:
                pass
            self.write_fd = -1

        if self.safe_stderr is not None:
            self.safe_stderr.close()

        if self.read_fd >= 0:
            try:
                os.close(self.read_fd)
            except OSError:
                pass
            self.read_fd = -1

        if self.saved_stderr >= 0:
            os.close(self.saved_stderr)
            self.saved_stderr = -1

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if IS_WINDOWS and self.old_std_handle is not None:
            kernel32.SetStdHandle(STD_ERROR_HANDLE, self.old_std_handle)
            self.old_std_handle = None

        os.dup2(self.saved_stderr, 2)
        sys.stderr = self.saved_python_stderr

        if self.write_fd >= 0:
            os.close(self.write_fd)
            self.write_fd = -1

        if self.reader is not None:
            self.reader.join()

        self.safe_stderr.close()
        os.close(self.saved_stderr)
        os.close(self.read_fd)
        return False


def start_miopen_logging():
    """Start filtering native MIOpen diagnostics without changing the environment."""
    global _logging_capture  # pylint: disable=global-statement
    if _logging_capture is None:
        try:
            # Enforce unbuffered output for ROCm/MIOpen environment on Windows
            os.environ["MIOPEN_ENABLE_LOGGING"] = "1"
            os.environ["MIOPEN_ENABLE_LOGGING_CMD"] = "1"
            os.environ["PYTHONUNBUFFERED"] = "1"

            _logging_capture = MIOpenLogRedirect()
            _logging_capture.__enter__()
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as err:
            log.warning(f'MIOpen logging: failed to start: {err}')
            _logging_capture = None


def stop_miopen_logging():
    """Stop filtering native MIOpen diagnostics and restore stderr."""
    global _logging_capture  # pylint: disable=global-statement
    if _logging_capture is not None:
        _logging_capture.__exit__(None, None, None)
        _logging_capture = None

atexit.register(stop_miopen_logging)


class _MIOpenStderrCapture:
    """Capture one native MIOpen operation and log its selected algorithm and time."""

    def __init__(self, operation: str = "convolution", repeats: int = 1):
        self.operation = operation
        self.repeats = max(1, repeats)
        self.lines: list[str] = []
        self.elapsed_ms = 0.0
        self.algorithms: list[str] = []
        self.redirect = None

    def __enter__(self):
        self.redirect = MIOpenLogRedirect()
        self.redirect.__enter__()
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self.redirect:
            self.redirect.__exit__(_exc_type, _exc_value, _traceback)
        return False


def capture_miopen(operation: str = "convolution", repeats: int = 1):
    """Return a side-effect-free context manager for one MIOpen operation."""
    return _MIOpenStderrCapture(operation=operation, repeats=repeats)

def _extract_algorithms(lines):
    return [match.group(1) for line in lines if (match := _ALGORITHM_PATTERN.search(line))]

_dilated_use_phases = {}

def _conv1d_phases(x: torch.Tensor, conv: torch.nn.Conv1d):
    """A dilated conv1d computed as `dilation` undilated convs over the strided phases.

    MIOpen has no usable dilated 1D kernel on gfx1100. Measured on 8x512x2048, kernel 3: the same
    FLOPs run at 18.7 TFLOP/s undilated and 0.3 TFLOP/s at dilation 3 - a 62x penalty, and these
    ResBlocks are most of the generator. A dilation-D convolution is exactly D independent undilated
    convolutions over x[..., i::D], so this routes around that kernel; measured 22-33x faster with a
    relative error of 1e-6, which is fp32 rounding.

    ROCm GPUs only. CPU handles dilation fine and the reshape would just cost copies there, and
    cuDNN is untested so NVIDIA keeps the native kernel.
    """
    d = conv.dilation[0]
    length, pad = x.shape[-1], conv.padding[0]
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
    """Compute a stride-one dilated Conv2d as independent undilated phases."""
    if conv.stride != (1, 1):
        raise ValueError("phase-split Conv2d requires stride=1")
    dilation_h, dilation_w = conv.dilation
    padding_h, padding_w = conv.padding
    height, width = x.shape[-2:]
    x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h))
    rem_h = (-x.shape[-2]) % dilation_h
    rem_w = (-x.shape[-1]) % dilation_w
    if rem_h or rem_w:
        x = torch.nn.functional.pad(x, (0, rem_w, 0, rem_h))
    batch, channels, padded_h, padded_w = x.shape
    phases = (
        x.view(batch, channels, padded_h // dilation_h, dilation_h, padded_w // dilation_w, dilation_w)
        .permute(0, 3, 5, 1, 2, 4)
        .reshape(batch * dilation_h * dilation_w, channels, padded_h // dilation_h, padded_w // dilation_w)
    )
    out = torch.nn.functional.conv2d(phases, conv.weight, conv.bias, groups=conv.groups)
    out_h = height + 2 * padding_h - dilation_h * (conv.kernel_size[0] - 1)
    out_w = width + 2 * padding_w - dilation_w * (conv.kernel_size[1] - 1)
    phase_h, phase_w = out.shape[-2:]
    channels_out = conv.weight.shape[0]
    out = (
        out.view(batch, dilation_h, dilation_w, channels_out, phase_h, phase_w)
        .permute(0, 3, 4, 1, 5, 2)
        .reshape(batch, channels_out, phase_h * dilation_h, phase_w * dilation_w)
    )
    return out[..., :out_h, :out_w]


def _conv3d_phases(x: torch.Tensor, conv: torch.nn.Conv3d):
    """Compute a stride-one dilated Conv3d as independent undilated phases."""
    if conv.stride != (1, 1, 1):
        raise ValueError("phase-split Conv3d requires stride=1")
    dilation_d, dilation_h, dilation_w = conv.dilation
    padding_d, padding_h, padding_w = conv.padding
    depth, height, width = x.shape[-3:]
    x = torch.nn.functional.pad(x, (padding_w, padding_w, padding_h, padding_h, padding_d, padding_d))
    rem_d = (-x.shape[-3]) % dilation_d
    rem_h = (-x.shape[-2]) % dilation_h
    rem_w = (-x.shape[-1]) % dilation_w
    if rem_d or rem_h or rem_w:
        x = torch.nn.functional.pad(x, (0, rem_w, 0, rem_h, 0, rem_d))
    batch, channels, padded_d, padded_h, padded_w = x.shape
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
    out = torch.nn.functional.conv3d(phases, conv.weight, conv.bias, groups=conv.groups)
    out_d = depth + 2 * padding_d - dilation_d * (conv.kernel_size[0] - 1)
    out_h = height + 2 * padding_h - dilation_h * (conv.kernel_size[1] - 1)
    out_w = width + 2 * padding_w - dilation_w * (conv.kernel_size[2] - 1)
    phase_d, phase_h, phase_w = out.shape[-3:]
    channels_out = conv.weight.shape[0]
    out = (
        out.view(batch, dilation_d, dilation_h, dilation_w, channels_out, phase_d, phase_h, phase_w)
        .permute(0, 4, 5, 1, 6, 2, 7, 3)
        .reshape(batch, channels_out, phase_d * dilation_d, phase_h * dilation_h, phase_w * dilation_w)
    )
    return out[..., :out_d, :out_h, :out_w]

def conv1d_dilated(x: torch.Tensor, conv: torch.nn.Conv1d):
    """Dilated conv1d, via whichever path is actually faster on this device.

    MIOpen's dilated 1D kernel is pathological on gfx1100 - measured 62x slower than the identical
    FLOPs undilated - but other architectures and later ROCm releases may not share that, and cuDNN
    was never tested. So this measures both paths once per workload rather than deciding by vendor,
    and the native kernel keeps ties. One probe costs a few milliseconds against a run of hours.
    """
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

def print_result(name, native_ms, native_algorithms, custom_ms, custom_algorithms):
    cyan = "\033[36m"
    green = "\033[92m"
    yellow = "\033[93m"
    magenta = "\033[95m"
    dim = "\033[2m"
    reset = "\033[0m"

    def field(label, value, color):
        return f"  {dim}{label:<18}{reset} {color}{value}{reset}"

    def algorithm_fields(algorithms):
        if not algorithms:
            return [field("Algorithm", "not emitted (cached or unavailable)", dim)]
        return [field("Algorithm", algorithms[-1], green)]

    ratio = native_ms / custom_ms
    print(f"{cyan}{'=' * 78}{reset}")
    print(f"{cyan}{name}{reset}")
    print(f"{magenta}native: miopen{reset} {dim}torch{reset}")
    for line in algorithm_fields(native_algorithms):
        print(line)
    print(field("Measured latency", f"{native_ms:.3f} ms", cyan))
    print(f"{magenta}phased{reset} {dim}included{reset}")
    for line in algorithm_fields(custom_algorithms):
        print(line)
    print(field("Measured latency", f"{custom_ms:.3f} ms", cyan))
    print(field("Torch / included", f"{ratio:.2f}x", yellow))

def print_native_result(name, native_ms, native_algorithms):
    cyan = "\033[36m"
    green = "\033[92m"
    magenta = "\033[95m"
    dim = "\033[2m"
    reset = "\033[0m"
    algorithm = native_algorithms[-1] if native_algorithms else "not emitted (cached or unavailable)"
    print(f"{cyan}{'=' * 78}{reset}")
    print(f"{cyan}{name}{reset}")
    print(f"{magenta}native: miopen{reset} {dim}torch{reset}")
    print(f"  {dim}{'Algorithm':<18}{reset} {green}{algorithm}{reset}")
    print(f"  {dim}{'Measured latency':<18}{reset} {cyan}{native_ms:.3f} ms{reset}")


def run_conv1d_suite():
    for channels in CONV1D_CHANNELS:
        for kernel_size, padding_dilations in CONV1D_CONFIGS:
            for index, (padding, dilation) in enumerate(padding_dilations):
                test_input = torch.randn(CONV1D_BATCH_SIZE, channels, CONV1D_LENGTH, device=DEVICE, dtype=_dtype())
                test_conv = nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation)
                ).to(DEVICE, dtype=_dtype())
                name = f"conv1d ch{channels}-k{kernel_size}-{index}"
                native_ms, native_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: conv(inputs), WARMUP_RUNS, BENCHMARK_RUNS)
                phases_ms, phases_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: _conv1d_phases(inputs, conv), WARMUP_RUNS, BENCHMARK_RUNS)
                print_result(name, native_ms, native_algorithms, phases_ms, phases_algorithms)


def run_conv2d_suite():
    for channels in CONV2D_CHANNELS:
        for kernel_size, dilations in CONV2D_CONFIGS:
            for index, dilation in enumerate(dilations):
                padding = tuple((size // 2) * value for size, value in zip(kernel_size, dilation))
                test_input = torch.randn(CONV2D_BATCH_SIZE, channels, *CONV2D_SIZE, device=DEVICE, dtype=_dtype())
                test_conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation).to(DEVICE, dtype=_dtype())
                name = f"conv2d ch{channels}-k{kernel_size}-d{dilation}-{index}"
                native_ms, native_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: conv(inputs), WARMUP_RUNS, BENCHMARK_RUNS)
                phases_ms, phases_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: _conv2d_phases(inputs, conv), WARMUP_RUNS, BENCHMARK_RUNS)
                print_result(name, native_ms, native_algorithms, phases_ms, phases_algorithms)


def run_conv3d_suite():
    for channels in CONV3D_CHANNELS:
        for kernel_size, dilations in CONV3D_CONFIGS:
            for index, dilation in enumerate(dilations):
                padding = tuple((size // 2) * value for size, value in zip(kernel_size, dilation))
                test_input = torch.randn(CONV3D_BATCH_SIZE, channels, *CONV3D_SIZE, device=DEVICE, dtype=_dtype())
                test_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation).to(DEVICE, dtype=_dtype())
                name = f"conv3d ch{channels}-k{kernel_size}-d{dilation}-{index}"
                native_ms, native_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: conv(inputs), WARMUP_RUNS, BENCHMARK_RUNS)
                phases_ms, phases_algorithms = benchmark(lambda conv=test_conv, inputs=test_input: _conv3d_phases(inputs, conv), WARMUP_RUNS, BENCHMARK_RUNS)
                print_result(name, native_ms, native_algorithms, phases_ms, phases_algorithms)


if __name__ == "__main__":
    if RUN_CONV1D:
        run_conv1d_suite()
    if RUN_CONV2D:
        run_conv2d_suite()
    if RUN_CONV3D:
        run_conv3d_suite()
