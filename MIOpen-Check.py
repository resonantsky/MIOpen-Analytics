#!/usr/bin/env python3
"""Install stable/nightly ROCm environments and compare their MIOpen benchmarks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import unquote
from urllib.request import Request, urlopen
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import comparison_generator


ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
IS_LINUX = os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Linux"
PYTORCH_ROCM_REPOSITORIES = {
    "10.0": "https://download.pytorch.org/whl/rocm10.0",
    "10.1": "https://download.pytorch.org/whl/rocm10.1",
}
PYTORCH_ROCM_REQUIRED_PACKAGES = ("torch", "torchvision")
PYTORCH_ROCM_OPTIONAL_PACKAGES = ("torchaudio",)
PYTORCH_ROCM_PACKAGES = PYTORCH_ROCM_REQUIRED_PACKAGES + PYTORCH_ROCM_OPTIONAL_PACKAGES
REPOSITORIES = {
    "stable": {
        "pytorch": "https://stable.repo.amd.com/rocm/pytorch/whl-next",
        "core": "https://stable.repo.amd.com/rocm/core/whl-next",
    },
    "nightly": {
        "pytorch": "https://nightly.repo.amd.com/rocm/pytorch/whl-next",
        "core": "https://nightly.repo.amd.com/rocm/core/whl-next",
    },
}
ARCHITECTURES = (
    "gfx1010",
    "gfx1011",
    "gfx1012",
    "gfx1030",
    "gfx1031",
    "gfx1032",
    "gfx1033",
    "gfx1034",
    "gfx1035",
    "gfx1036",
    "gfx1100",
    "gfx1101",
    "gfx1102",
    "gfx1103",
    "gfx110x",
    "gfx1150",
    "gfx1151",
    "gfx1152",
    "gfx1153",
    "gfx115x",
    "gfx12-0",
    "gfx1200",
    "gfx1201",
    "gfx1250",
    "gfx900",
    "gfx906",
    "gfx908",
    "gfx90a",
    "gfx90c",
    "gfx942",
    "gfx950",
)
STABLE_ROCM_10_0_ARCHITECTURES = frozenset(
    {
        "gfx1010",
        "gfx1011",
        "gfx1012",
        "gfx1030",
        "gfx1031",
        "gfx1032",
        "gfx1033",
        "gfx1034",
        "gfx1035",
        "gfx1036",
    }
)
DEVICE_REQUIREMENT = re.compile(
    r"(?im)^(\s*[A-Za-z0-9_.-]*-device-)gfx[0-9a-z-]+(?=\s*(?:[<>=!~;\[]|$))"
)
REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
AMD_PACKAGE_NAMES = {
    "torch",
    "torchaudio",
    "torchvision",
    "rocm",
    "rocm-bootstrap",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
}
TRITON_PACKAGE_NAMES = {
    "triton",
    "triton-rocm",
    "pytorch-triton",
    "pytorch-triton-rocm",
}
UV_EXECUTABLE = shutil.which("uv")
WHEEL_CHECK_TIMEOUT = 12


@dataclass(frozen=True)
class WheelProbe:
    available: Optional[bool]
    detail: str
    versions: Tuple[str, ...] = ()


class _WheelLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs) -> None:
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href")


def parse_architecture(value: str) -> str:
    architecture = value.strip().lower()
    prefix = "amd-torch-device-"
    if architecture.startswith(prefix):
        architecture = architecture[len(prefix) :]
    if architecture not in ARCHITECTURES:
        choices = ", ".join(ARCHITECTURES)
        raise ValueError(f"Unsupported architecture {value!r}. Choose one of: {choices}")
    return architecture


def pytorch_rocm_release(architecture: str) -> str:
    return "10.0" if architecture in STABLE_ROCM_10_0_ARCHITECTURES else "10.1"


def _linux_platform_tag_compatible(platform_tags: set, runtime_platform: str) -> bool:
    if "any" in platform_tags or runtime_platform in platform_tags:
        return True
    if not IS_LINUX:
        return False

    architecture = runtime_platform.split("linux_", 1)[-1]
    confstr = getattr(os, "confstr", None)
    libc = confstr("CS_GNU_LIBC_VERSION") if confstr is not None else None
    libc_match = re.fullmatch(r"glibc (\d+)\.(\d+)", libc or "")
    if libc_match is None:
        return False
    libc_version = tuple(map(int, libc_match.groups()))

    legacy_manylinux = {
        "manylinux1": (2, 5),
        "manylinux2010": (2, 12),
        "manylinux2014": (2, 17),
    }
    for tag in platform_tags:
        match = re.fullmatch(r"manylinux_(\d+)_(\d+)_(.+)", tag)
        if match and match.group(3) == architecture:
            required_version = (int(match.group(1)), int(match.group(2)))
        else:
            match = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_(.+)", tag)
            if match is None or match.group(2) != architecture:
                continue
            required_version = legacy_manylinux[match.group(1)]
        if libc_version >= required_version:
            return True
    return False


def _wheel_matches_runtime(filename: str) -> Tuple[bool, Optional[str]]:
    try:
        prefix, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
        _, version = prefix.rsplit("-", 1)
    except ValueError:
        return False, None

    major, minor = (3, 12) if IS_LINUX else sys.version_info[:2]
    python_tags = set(python_tag.split("."))
    supported_python_tags = {f"py{major}", f"py{major}{minor}", "py3"}
    if sys.implementation.name == "cpython":
        supported_python_tags.add(f"cp{major}{minor}")
    if not python_tags.intersection(supported_python_tags):
        return False, None

    abi_tags = set(abi_tag.split("."))
    supported_abi_tags = {"none", "abi3"}
    if sys.implementation.name == "cpython":
        supported_abi_tags.add(f"cp{major}{minor}")
    if not abi_tags.intersection(supported_abi_tags):
        return False, None

    runtime_platform = sysconfig.get_platform().lower().replace("-", "_")
    platform_tags = set(platform_tag.lower().split("."))
    if not _linux_platform_tag_compatible(platform_tags, runtime_platform):
        return False, None
    return True, unquote(version)


def _probe_package_artifact(
    index_url: str,
    package: str,
    required_version: Optional[str],
    required_version_fragment: Optional[str] = None,
) -> WheelProbe:
    url = f"{index_url}/{package}/"
    request = Request(url, headers={"Accept": "text/html"})
    try:
        with urlopen(request, timeout=WHEEL_CHECK_TIMEOUT) as response:
            page = response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        if error.code == 404:
            return WheelProbe(False, "package not published")
        return WheelProbe(None, f"index returned HTTP {error.code}")
    except (URLError, TimeoutError, OSError) as error:
        return WheelProbe(None, f"index check failed: {error}")

    parser = _WheelLinkParser()
    parser.feed(page)
    versions = set()
    has_source_distribution = False
    for link in parser.links:
        filename = unquote(link.split("#", 1)[0].split("?", 1)[0]).rsplit("/", 1)[-1]
        if package == "rocm" and filename.endswith(".tar.gz"):
            version_prefix = f"{package}-"
            if filename.startswith(version_prefix):
                versions.add(filename[len(version_prefix) : -len(".tar.gz")])
                has_source_distribution = True
            continue
        if not filename.endswith(".whl"):
            continue
        compatible, version = _wheel_matches_runtime(filename)
        if (
            compatible
            and version is not None
            and (
                required_version_fragment is None
                or required_version_fragment in version
            )
        ):
            versions.add(version)

    if required_version is not None:
        if required_version in versions:
            detail = (
                f"source distribution {required_version} available"
                if package == "rocm" and has_source_distribution
                else f"wheel version {required_version} available"
            )
            return WheelProbe(True, detail, (required_version,))
        return WheelProbe(
            False,
            f"required version {required_version} has no compatible wheel",
            tuple(sorted(versions)),
        )
    if versions:
        detail = (
            "source distribution available"
            if package == "rocm" and has_source_distribution
            else "compatible wheel available"
        )
        return WheelProbe(True, detail, tuple(sorted(versions)))
    return WheelProbe(False, "no compatible wheel published for this Python/platform")


def check_architecture_wheels() -> Dict[str, Dict[Tuple[str, str], WheelProbe]]:
    queries = {}
    architecture_queries = {architecture: [] for architecture in ARCHITECTURES}
    for architecture in ARCHITECTURES:
        _, amd_requirements = requirements_for_architecture(architecture)
        stable_constraints = stable_runtime_constraints(architecture)
        constrained_versions = {}
        if stable_constraints:
            for line in stable_constraints.splitlines():
                match = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)==(.+)\s*", line)
                if match:
                    constrained_versions[match.group(1).lower()] = match.group(2)

        for channel in ("stable", "nightly"):
            for line in amd_requirements.splitlines():
                package = line.strip().lower()
                index_kind = "core" if package.startswith("rocm") else "pytorch"
                required_version = (
                    constrained_versions.get(package) if channel == "stable" else None
                )
                key = (channel, package, required_version)
                index_url = REPOSITORIES[channel][index_kind]
                queries[key] = (index_url, package, required_version)
                architecture_queries[architecture].append(key)

        if IS_LINUX:
            rocm_release = pytorch_rocm_release(architecture)
            channel = f"pytorch-rocm{rocm_release}"
            for package in PYTORCH_ROCM_PACKAGES:
                key = (channel, package, None)
                queries[key] = (
                    f"{PYTORCH_ROCM_REPOSITORIES[rocm_release]}",
                    package,
                    None,
                    f"+rocm{rocm_release}",
                )
                architecture_queries[architecture].append(key)

    results = {}
    print(
        f"Checking required package artifacts for Python "
        f"{3 if IS_LINUX else sys.version_info.major}."
        f"{12 if IS_LINUX else sys.version_info.minor} ({sysconfig.get_platform()})...",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(_probe_package_artifact, *query): key
            for key, query in queries.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return {
        architecture: {
            (key[0], key[1]): results[key]
            for key in architecture_queries[architecture]
        }
        for architecture in ARCHITECTURES
    }


def architecture_wheel_status(
    probes: Mapping[Tuple[str, str], WheelProbe]
) -> str:
    availability = [probe.available for probe in probes.values()]
    if all(value is True for value in availability):
        return "available"
    if any(value is False for value in availability):
        return "missing"
    return "unknown"


def official_pytorch_package_status(
    probes: Mapping[str, WheelProbe]
) -> Tuple[bool, Tuple[str, ...], Tuple[str, ...]]:
    unavailable_required = tuple(
        package
        for package in PYTORCH_ROCM_REQUIRED_PACKAGES
        if (probe := probes.get(package)) is None or probe.available is not True
    )
    unavailable_optional = tuple(
        package
        for package in PYTORCH_ROCM_OPTIONAL_PACKAGES
        if (probe := probes.get(package)) is None or probe.available is not True
    )
    return not unavailable_required, unavailable_required, unavailable_optional


def print_supported_architectures() -> None:
    lines = ["Supported architectures:"]
    groups = ("gfx10", "gfx11", "gfx12", "gfx9")
    for prefix in groups:
        architectures = tuple(
            architecture
            for architecture in ARCHITECTURES
            if architecture.startswith(prefix)
        )
        if architectures:
            lines.append(f"  {prefix}: " + ", ".join(architectures))
    print("\n".join(lines), flush=True)


def print_architecture_wheel_status(
    availability: Mapping[str, Mapping[Tuple[str, str], WheelProbe]]
) -> None:
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    reset = "\033[0m"
    colors = {
        "available": green,
        "missing": red,
        "unknown": yellow,
    }

    def print_channel_status(title: str, include_probe) -> None:
        print(title)
        for prefix in ("gfx10", "gfx11", "gfx12", "gfx9"):
            architectures = [
                architecture
                for architecture in ARCHITECTURES
                if architecture.startswith(prefix) and architecture in availability
            ]
            colored_architectures = []
            for architecture in architectures:
                probes = {
                    key: probe
                    for key, probe in availability[architecture].items()
                    if include_probe(key[0], key[1])
                }
                colored_architectures.append(
                    f"{colors[architecture_wheel_status(probes)]}{architecture}{reset}"
                )
            if colored_architectures:
                print(f"  {prefix}: " + ", ".join(colored_architectures))

    if IS_LINUX:
        print_channel_status(
            "AMD package availability (stable + nightly):",
            lambda channel, _package: not channel.startswith("pytorch-rocm"),
        )
        print_channel_status(
            "Official PyTorch availability (Python 3.12; torch + torchvision required):",
            lambda channel, package: (
                channel.startswith("pytorch-rocm")
                and package in PYTORCH_ROCM_REQUIRED_PACKAGES
            ),
        )
        print_channel_status(
            "Optional official PyTorch torchaudio availability:",
            lambda channel, package: (
                channel.startswith("pytorch-rocm")
                and package in PYTORCH_ROCM_OPTIONAL_PACKAGES
            ),
        )
    else:
        print_channel_status(
            "AMD package availability (stable + nightly):",
            lambda _channel, _package: True,
        )


def architecture_warning(
    architecture: str,
    probes: Mapping[Tuple[str, str], WheelProbe],
) -> Optional[str]:
    status = architecture_wheel_status(probes)
    if status == "available":
        return None

    unavailable = []
    uncertain = []
    for (channel, package), probe in probes.items():
        if probe.available is False:
            unavailable.append(f"{channel} {package}: {probe.detail}")
        elif probe.available is None:
            uncertain.append(f"{channel} {package}: {probe.detail}")
    if unavailable:
        detail = "; ".join(unavailable[:5])
        if len(unavailable) > 5:
            detail += f"; and {len(unavailable) - 5} more"
        return (
            f"Some required package artifacts for {architecture} are missing for this "
            f"Python/platform: {detail}. Installation may fail."
        )
    detail = "; ".join(uncertain[:5])
    if len(uncertain) > 5:
        detail += f"; and {len(uncertain) - 5} more"
    return (
        f"Package availability for {architecture} could not be fully checked: "
        f"{detail}. Continuing without disabling the architecture."
    )


def stable_runtime_constraints(architecture: str) -> Optional[str]:
    if architecture not in STABLE_ROCM_10_0_ARCHITECTURES:
        return None

    versions = {
        f"amd-torch-device-{architecture}": "2.13.0+rocm10.0.0",
        f"amd-torchvision-device-{architecture}": "0.28.0+rocm10.0.0",
        f"rocm-sdk-device-{architecture}": "10.0.0",
        "rocm": "10.0.0",
        "rocm-sdk-core": "10.0.0",
        "rocm-sdk-devel": "10.0.0",
        "rocm-sdk-libraries": "10.0.0",
        "torch": "2.13.0+rocm10.0.0",
        "torchaudio": "2.11.0.2+rocm10.0.0",
        "torchvision": "0.28.0+rocm10.0.0",
    }
    return "\n".join(f"{package}=={version}" for package, version in versions.items()) + "\n"


def requirements_for_architecture(architecture: str) -> tuple[str, str]:
    contents = REQUIREMENTS.read_text(encoding="utf-8")
    rewritten, replacements = DEVICE_REQUIREMENT.subn(
        lambda match: f"{match.group(1)}{architecture}", contents
    )
    if replacements == 0:
        raise ValueError(
            f"No architecture-specific '-device-gfx...' requirements found in {REQUIREMENTS.name}"
        )

    common_requirements = []
    amd_requirements = []
    for line in rewritten.splitlines():
        match = REQUIREMENT_NAME.match(line)
        if match is None:
            common_requirements.append(line)
            continue

        package = match.group(1).lower().replace("_", "-").replace(".", "-")
        is_amd_package = (
            package in AMD_PACKAGE_NAMES
            or package.startswith("amd-torch-device-")
            or package.startswith("amd-torchvision-device-")
            or package.startswith("rocm-sdk-device-")
        )
        if is_amd_package:
            # Each repository publishes its own compatible ROCm/PyTorch versions.
            amd_requirements.append(match.group(1))
        else:
            common_requirements.append(line)

    if not amd_requirements:
        raise ValueError(f"No AMD runtime packages found in {REQUIREMENTS.name}")
    return "\n".join(common_requirements) + "\n", "\n".join(amd_requirements) + "\n"


def venv_python(environment: Path) -> Path:
    executable = "python.exe" if sys.platform == "win32" else "python"
    scripts_directory = "Scripts" if sys.platform == "win32" else "bin"
    return environment / scripts_directory / executable


def install_command(
    python: Path,
    requirements: Path,
    index_url: str,
    extra_index_url: Optional[str] = None,
    constraints: Optional[Path] = None,
    allow_prerelease: bool = False,
) -> list[str]:
    if UV_EXECUTABLE is not None:
        command = [
            UV_EXECUTABLE,
            "pip",
            "install",
            "--link-mode=copy",
            "--python",
            str(python),
            "--index-url",
            index_url,
            "--prerelease",
            "allow",
        ]
        if extra_index_url is not None:
            command.extend(["--extra-index-url", extra_index_url])
        if constraints is not None:
            command.extend(["--constraints", str(constraints)])
        command.extend(["--requirements", str(requirements)])
        return command

    command = [
        str(python),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--index-url",
        index_url,
    ]
    if extra_index_url is not None:
        command.extend(["--extra-index-url", extra_index_url])
    if constraints is not None:
        command.extend(["--constraint", str(constraints)])
    if extra_index_url is not None or allow_prerelease:
        command.append("--pre")
    command.extend(["-r", str(requirements)])
    return command


def run_and_tee(command: list[str], output) -> None:
    with subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("Benchmark subprocess stdout was not captured.")
        for line in process.stdout:
            output.write(comparison_generator.strip_ansi(line))
            output.flush()
            sys.stdout.write(line)
            sys.stdout.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def recommended_install_command(
    environment: str,
    stable_results: Path,
    nightly_results: Path,
    torchvision_validated: bool,
) -> Optional[str]:
    if not torchvision_validated:
        return None

    results_file = stable_results if environment == "stable" else nightly_results
    packages, _ = comparison_generator.parse_file(results_file)
    architecture = comparison_generator.read_metadata(results_file, "Architecture")
    if architecture is None:
        return None
    required_packages = {
        "torch",
        "torchvision",
        f"amd-torch-device-{architecture}",
        f"amd-torchvision-device-{architecture}",
    }
    if not required_packages.issubset(packages) or any(
        not packages[package] or packages[package].lower() == "none"
        for package in required_packages
    ):
        return None
    if packages["torchvision"] != packages[f"amd-torchvision-device-{architecture}"]:
        return None

    architecture_packages = {
        f"amd-torch-device-{architecture}",
        f"amd-torchvision-device-{architecture}",
        f"rocm-sdk-device-{architecture}",
    }
    runtime_packages = {
        package: version
        for package, version in packages.items()
        if (
            package in AMD_PACKAGE_NAMES
            or package in architecture_packages
            or package in TRITON_PACKAGE_NAMES
        )
    }
    if not runtime_packages:
        raise ValueError(f"No ROCm or PyTorch packages found in {results_file}.")

    repositories = REPOSITORIES[environment]
    package_specs = " ".join(
        f"{package}=={version}" for package, version in sorted(runtime_packages.items())
    )
    return (
        "python -m pip install --upgrade --pre "
        f"--index-url {repositories['pytorch']} "
        f"--extra-index-url {repositories['core']} "
        f"{package_specs}"
    )


def recommended_official_pytorch_install_command(
    results_file: Path, validation_passed: bool
) -> Optional[str]:
    if not validation_passed:
        return None

    packages, _ = comparison_generator.parse_file(results_file)
    required_packages = ("torch", "torchvision")
    if any(
        package not in packages or not packages[package]
        or packages[package].lower() == "none"
        for package in required_packages
    ):
        return None

    repository = comparison_generator.read_metadata(results_file, "Repository")
    if repository not in PYTORCH_ROCM_REPOSITORIES.values():
        return None

    install_packages = list(required_packages)
    if packages.get("torchaudio") and packages["torchaudio"].lower() != "none":
        install_packages.append("torchaudio")
    install_packages.extend(
        package
        for package in sorted(TRITON_PACKAGE_NAMES)
        if packages.get(package) and packages[package].lower() != "none"
    )
    package_specs = " ".join(
        f"{package}=={packages[package]}" for package in install_packages
    )
    return (
        "python -m pip install --upgrade --pre "
        f"--index-url {repository} {package_specs}"
    )


def validate_torchvision_runtime(python: Path, architecture: str) -> Tuple[bool, str]:
    validation_script = """
from importlib.metadata import version
import sys

import torch
import torchvision

architecture = sys.argv[1]
torchvision_version = version("torchvision")
device_version = version(f"amd-torchvision-device-{architecture}")
if torchvision_version != device_version:
    raise RuntimeError(
        f"torchvision {torchvision_version} does not match "
        f"amd-torchvision-device-{architecture} {device_version}"
    )

def verify_nms(device):
    boxes = torch.tensor(
        [[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device=device
    )
    scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device=device)
    selected = torchvision.ops.nms(boxes, scores, 0.5)
    if selected.tolist() != [0]:
        raise RuntimeError(f"torchvision NMS returned an unexpected result on {device}")

verify_nms("cpu")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch reports no ROCm/CUDA device for torchvision GPU validation")
verify_nms("cuda")
print(
    f"torch={torch.__version__}, torchvision={torchvision_version}, "
    f"amd-torchvision-device={device_version}; CPU and GPU NMS passed"
)
"""
    try:
        result = subprocess.run(
            [str(python), "-c", validation_script, architecture],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "torchvision validation timed out"

    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if not detail:
            detail = f"validation exited with status {result.returncode}"
        return False, detail
    return True, result.stdout.strip()


def validate_pytorch_runtime(python: Path) -> Tuple[bool, str]:
    validation_script = """
import torch
import torchvision

if not torch.version.hip:
    raise RuntimeError("The installed PyTorch build does not report a ROCm runtime")
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch reports no ROCm device")

boxes = torch.tensor(
    [[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device="cpu"
)
scores = torch.tensor([0.9, 0.8], dtype=torch.float32, device="cpu")
if torchvision.ops.nms(boxes, scores, 0.5).tolist() != [0]:
    raise RuntimeError("torchvision CPU NMS returned an unexpected result")
if torchvision.ops.nms(boxes.cuda(), scores.cuda(), 0.5).tolist() != [0]:
    raise RuntimeError("torchvision GPU NMS returned an unexpected result")
print(
    f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
    f"HIP={torch.version.hip}; CPU and GPU NMS passed"
)
"""
    try:
        result = subprocess.run(
            [str(python), "-c", validation_script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "official PyTorch ROCm validation timed out"

    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if not detail:
            detail = f"validation exited with status {result.returncode}"
        return False, detail
    return True, result.stdout.strip()


def print_linux_validation_summary(
    validations: Mapping[str, Tuple[bool, str]],
) -> None:
    cyan = "\033[36m"
    green = "\033[92m"
    red = "\033[91m"
    reset = "\033[0m"
    print(f"\n{cyan}{'=' * 78}{reset}")
    print(f"{cyan}Torchvision validation{reset}")
    for name in ("stable", "nightly"):
        if name not in validations:
            continue
        passed, detail = validations[name]
        color = green if passed else red
        status = "PASS" if passed else "FAILED"
        print(f"{color}Torchvision validation ({name}): {status} - {detail}{reset}")

    for name, (passed, detail) in validations.items():
        if name in ("stable", "nightly"):
            continue
        color = green if passed else red
        status = "PASS" if passed else "FAILED"
        print(f"{cyan}{name} validation{reset}: {color}{status} - {detail}{reset}")
    print(f"{cyan}{'=' * 78}{reset}", flush=True)


def print_linux_completion_summary(
    comparison_file: Path,
    metrics: Mapping[str, Mapping[str, Union[float, Mapping[str, float]]]],
    configuration_count: int,
    preferred: str,
    pip_command: Optional[str],
) -> None:
    cyan = "\033[36m"
    green = "\033[92m"
    yellow = "\033[93m"
    dim = "\033[2m"
    reset = "\033[0m"
    print(f"\n{cyan}{'=' * 78}{reset}")
    title = (
        "Stable / nightly / official PyTorch benchmark summary"
        if len(metrics) == 3
        else "Stable / nightly benchmark summary (official PyTorch run skipped)"
    )
    print(f"{cyan}{title}{reset}")
    print(
        f"  {dim}Matched configurations{reset} "
        f"{yellow}{configuration_count}{reset}"
    )
    for label, run in metrics.items():
        print(
            f"  {dim}{label.removesuffix('-results') + ' total / average':<34}{reset} "
            f"{cyan}{run['total']:.3f} ms / {run['average']:.3f} ms{reset}"
        )
    print(
        f"  {dim}{'Preferred packages':<34}{reset} "
        f"{green}{preferred}{reset}"
    )
    print(f"  {dim}{'Comparison report':<34}{reset} {cyan}{comparison_file}{reset}")
    if pip_command is not None:
        print(f"\n{green}Recommended pip install command:{reset}")
        print(f"{yellow}{pip_command}{reset}")
    else:
        print(
            f"\n{yellow}No pip install recommendation: required runtime validation "
            f"must pass and package versions must be present in the result file.{reset}"
        )
    print(f"{cyan}{'=' * 78}{reset}")


def print_completion_summary(
    comparison_file: Path,
    metrics: Mapping[str, Union[int, float]],
    preferred: str,
    pip_command: Optional[str],
) -> None:
    cyan = "\033[36m"
    green = "\033[92m"
    yellow = "\033[93m"
    magenta = "\033[95m"
    dim = "\033[2m"
    reset = "\033[0m"

    print(f"\n{cyan}{'=' * 78}{reset}")
    print(f"{cyan}Stable / nightly benchmark summary{reset}")
    print(
        f"  {dim}Matched configurations{reset} "
        f"{yellow}{metrics['configuration_count']}{reset}"
    )
    print(
        f"  {dim}{'Stable total / average':<24}{reset} "
        f"{cyan}{metrics['stable_total']:.3f} ms / "
        f"{metrics['stable_average']:.3f} ms{reset}"
    )
    print(
        f"  {dim}{'Nightly total / average':<24}{reset} "
        f"{magenta}{metrics['nightly_total']:.3f} ms / "
        f"{metrics['nightly_average']:.3f} ms{reset}"
    )
    print(
        f"  {dim}{'Preferred packages':<24}{reset} "
        f"{green}{preferred}{reset}"
    )
    print(f"  {dim}{'Comparison report':<24}{reset} {cyan}{comparison_file}{reset}")
    if pip_command is not None:
        print(f"\n{green}Recommended pip install command:{reset}")
        print(f"{yellow}{pip_command}{reset}")
    else:
        print(
            f"\n{yellow}No pip install recommendation: torchvision and its "
            f"architecture package must pass CPU and GPU NMS validation in both "
            f"benchmark environments.{reset}"
        )
    print(f"{cyan}{'=' * 78}{reset}")


def prepare_environment(
    name: str,
    architecture: str,
    common_requirements: Path,
    amd_requirements: Path,
    constraints: Optional[Path] = None,
    repositories: Optional[Mapping[str, str]] = None,
    allow_prerelease: bool = False,
) -> Path:
    environment = ROOT / f"venv-{name}"
    if not venv_python(environment).is_file():
        print(f"Creating {name} virtual environment at {environment}", flush=True)
        if IS_LINUX and UV_EXECUTABLE is not None:
            create_command = [
                UV_EXECUTABLE,
                "venv",
                "--python",
                "3.12",
                "--seed",
                str(environment),
            ]
        elif IS_LINUX:
            python312 = shutil.which("python3.12")
            if python312 is None:
                raise SystemExit("Linux benchmark environments require uv or python3.12.")
            create_command = [python312, "-m", "venv", str(environment)]
        else:
            create_command = [sys.executable, "-m", "venv", str(environment)]
        subprocess.run(
            create_command,
            cwd=ROOT,
            check=True,
        )
    if IS_LINUX:
        version = subprocess.run(
            [str(venv_python(environment)), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if version != "3.12":
            raise SystemExit(
                f"{environment.name} uses Python {version}; Linux benchmark environments require Python 3.12."
            )

    repositories = repositories or REPOSITORIES[name]
    print(f"Installing {name} requirements for {architecture}", flush=True)
    environment_python = venv_python(environment)
    commands = (
        install_command(
            environment_python, common_requirements, "https://pypi.org/simple"
        ),
        install_command(
            environment_python,
            amd_requirements,
            repositories["pytorch"],
            repositories.get("core"),
            constraints,
            allow_prerelease,
        ),
    )
    for command in commands:
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            raise SystemExit(
                f"{name} dependency installation failed (exit code {error.returncode}). "
                f"Check repository availability and wheel support for {architecture}."
            ) from error
    return venv_python(environment)


def run_analysis(
    name: str,
    architecture: str,
    python: Path,
    mode: str,
    repositories: Optional[Mapping[str, str]] = None,
) -> Path:
    results_file = ROOT / f"{name}-results.md"
    check_script = ROOT / f"miopen-check-{mode}.py"
    repositories = repositories or REPOSITORIES[name]
    print(f"\nRunning {name} {mode} benchmark; recording {results_file.name}", flush=True)
    with results_file.open("w", encoding="utf-8") as output:
        metadata = (
            f"# MIOpen benchmark results\n\n"
            f"Repository: {repositories['pytorch']}  \n"
        )
        if "core" in repositories:
            metadata += f"Core repository: {repositories['core']}  \n"
        metadata += (
            f"Architecture: {architecture}  \n"
            f"Benchmark: {check_script.name}\n\n"
            "## Installed packages\n\n"
            "```text\n"
        )
        output.write(metadata)
        subprocess.run(
            [str(python), "-m", "pip", "list", "--format=columns"],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
        output.write("\n```\n\n## Benchmark output\n\n```text\n")
        output.flush()
        print(f"\n=== {name.title()} benchmark output (also saved to {results_file.name}) ===")
        try:
            run_and_tee([str(python), "-u", str(check_script)], output)
        finally:
            output.write("\n```\n")
    return results_file


def linux_main(mode: str, parser: argparse.ArgumentParser) -> None:
    print(
        f"Package installer: {'uv' if UV_EXECUTABLE is not None else 'pip'}",
        flush=True,
    )
    print("Linux benchmark environments use Python 3.12.", flush=True)
    print_supported_architectures()
    wheel_availability = check_architecture_wheels()
    print_architecture_wheel_status(wheel_availability)
    try:
        architecture = parse_architecture(
            input("Enter the GFX architecture (for example, gfx1030): ")
        )
        architecture_probes = wheel_availability[architecture]
        amd_probes = {
            key: probe
            for key, probe in architecture_probes.items()
            if not key[0].startswith("pytorch-rocm")
        }
        warning = architecture_warning(architecture, amd_probes)
        if warning is not None:
            warning_color = "\033[91m" if architecture_wheel_status(
                amd_probes
            ) == "missing" else "\033[93m"
            print(f"{warning_color}Warning: {warning}\033[0m", flush=True)
        rocm_release = pytorch_rocm_release(architecture)
        pytorch_channel = f"pytorch-rocm{rocm_release}"
        pytorch_probes = {
            package: probe
            for (channel, package), probe in wheel_availability[architecture].items()
            if channel == pytorch_channel
        }
        pytorch_available, unavailable_required, unavailable_optional = (
            official_pytorch_package_status(pytorch_probes)
        )
        if unavailable_required:
            detail = "; ".join(
                f"{package}: {pytorch_probes[package].detail}"
                if package in pytorch_probes
                else f"{package}: not checked"
                for package in unavailable_required
            )
            print(
                f"\033[93mOfficial PyTorch ROCm {rocm_release} benchmark skipped "
                f"for {architecture}: Python 3.12 Linux wheels are unavailable "
                f"or unverified ({detail}). AMD benchmarks will continue.\033[0m",
                flush=True,
            )
        if unavailable_optional:
            print(
                f"\033[93mOptional PyTorch package(s) skipped: "
                f"{', '.join(unavailable_optional)}.\033[0m",
                flush=True,
            )
        pytorch_packages = list(PYTORCH_ROCM_REQUIRED_PACKAGES)
        pytorch_packages.extend(
            package
            for package in PYTORCH_ROCM_OPTIONAL_PACKAGES
            if package not in unavailable_optional
        )
        common, amd = requirements_for_architecture(architecture)
    except ValueError as error:
        parser.error(str(error))

    temporary_requirements = []
    try:
        requirement_sets = (
            ("miopen-common-requirements-", common),
            ("miopen-amd-requirements-", amd),
            ("miopen-pytorch-requirements-", "\n".join(pytorch_packages) + "\n"),
        )
        for prefix, contents in requirement_sets:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix=prefix,
                dir=ROOT,
                delete=False,
            ) as file:
                file.write(contents)
                temporary_requirements.append(Path(file.name))

        stable_constraints = stable_runtime_constraints(architecture)
        stable_constraints_path: Optional[Path] = None
        if stable_constraints is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix="miopen-stable-rocm-10.0-constraints-",
                dir=ROOT,
                delete=False,
            ) as file:
                file.write(stable_constraints)
                stable_constraints_path = Path(file.name)
                temporary_requirements.append(stable_constraints_path)
            print(f"Pinning stable AMD packages to ROCm 10.0.0 for {architecture}", flush=True)

        stable_python = prepare_environment(
            "stable",
            architecture,
            temporary_requirements[0],
            temporary_requirements[1],
            stable_constraints_path,
        )
        nightly_python = prepare_environment(
            "nightly", architecture, temporary_requirements[0], temporary_requirements[1]
        )

        pytorch_name = f"pytorch-rocm{rocm_release}"
        pytorch_repositories = {
            "pytorch": PYTORCH_ROCM_REPOSITORIES[rocm_release],
        }
        pytorch_python = None
        if pytorch_available:
            pytorch_python = prepare_environment(
                pytorch_name,
                architecture,
                temporary_requirements[0],
                temporary_requirements[2],
                repositories=pytorch_repositories,
                allow_prerelease=True,
            )

        stable_results = run_analysis("stable", architecture, stable_python, mode)
        nightly_results = run_analysis("nightly", architecture, nightly_python, mode)
        result_files = [stable_results, nightly_results]
        pytorch_results = None
        if pytorch_python is not None:
            pytorch_results = run_analysis(
                pytorch_name, architecture, pytorch_python, mode, pytorch_repositories
            )
            result_files.append(pytorch_results)

        comparison_file = ROOT / "comparison.md"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "comparison_generator.py"),
                *[str(result_file) for result_file in result_files],
                "--output",
                str(comparison_file),
            ],
            cwd=ROOT,
            check=True,
        )
        if pytorch_python is not None:
            metrics, common_configs = comparison_generator.aggregate_multi_metrics(result_files)
        else:
            two_way_metrics = comparison_generator.aggregate_metrics(
                stable_results, nightly_results
            )
            metrics = {
                "stable": {
                    "by_config": two_way_metrics["stable_by_config"],
                    "total": two_way_metrics["stable_total"],
                    "average": two_way_metrics["stable_average"],
                },
                "nightly": {
                    "by_config": two_way_metrics["nightly_by_config"],
                    "total": two_way_metrics["nightly_total"],
                    "average": two_way_metrics["nightly_average"],
                },
            }
            common_configs = two_way_metrics["common_configs"]
        validations = {
            "stable": validate_torchvision_runtime(stable_python, architecture),
            "nightly": validate_torchvision_runtime(nightly_python, architecture),
        }
        if pytorch_python is not None:
            validations[pytorch_name] = validate_pytorch_runtime(pytorch_python)
        print_linux_validation_summary(validations)
        preferred = min(metrics, key=lambda label: metrics[label]["average"])
        preferred_environment = preferred.rsplit("-results", 1)[0]
        all_validations_passed = all(passed for passed, _detail in validations.values())
        if preferred_environment.startswith("pytorch-rocm"):
            pip_command = recommended_official_pytorch_install_command(
                pytorch_results, all_validations_passed
            )
        else:
            pip_command = recommended_install_command(
                preferred_environment,
                stable_results,
                nightly_results,
                all_validations_passed,
            )
        print_linux_completion_summary(
            comparison_file,
            metrics,
            len(common_configs),
            preferred_environment,
            pip_command,
        )
    finally:
        for requirements_file in temporary_requirements:
            requirements_file.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("normal", "verbose"),
        default="normal",
        help="benchmark output mode (default: normal)",
    )
    args = parser.parse_args()

    if IS_LINUX:
        linux_main(args.mode, parser)
        return

    print(
        f"Package installer: {'uv' if UV_EXECUTABLE is not None else 'pip'}",
        flush=True,
    )
    print_supported_architectures()
    wheel_availability = check_architecture_wheels()
    print_architecture_wheel_status(wheel_availability)
    try:
        architecture = parse_architecture(
            input("Enter the GFX architecture (for example, gfx1030): ")
        )
        warning = architecture_warning(
            architecture, wheel_availability[architecture]
        )
        if warning is not None:
            warning_color = "\033[91m" if architecture_wheel_status(
                wheel_availability[architecture]
            ) == "missing" else "\033[93m"
            print(f"{warning_color}Warning: {warning}\033[0m", flush=True)
        common, amd = requirements_for_architecture(architecture)
    except ValueError as error:
        parser.error(str(error))

    temporary_requirements = []
    try:
        for prefix, contents in (
            ("miopen-common-requirements-", common),
            ("miopen-amd-requirements-", amd),
        ):
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix=prefix,
                dir=ROOT,
                delete=False,
            ) as file:
                file.write(contents)
                temporary_requirements.append(Path(file.name))

        stable_constraints = stable_runtime_constraints(architecture)
        stable_constraints_path: Optional[Path] = None
        if stable_constraints is not None:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix="miopen-stable-rocm-10.0-constraints-",
                dir=ROOT,
                delete=False,
            ) as file:
                file.write(stable_constraints)
                stable_constraints_path = Path(file.name)
                temporary_requirements.append(stable_constraints_path)
            print(
                f"Pinning stable ROCm packages to 10.0.0 for {architecture}",
                flush=True,
            )

        stable_python = prepare_environment(
            "stable",
            architecture,
            temporary_requirements[0],
            temporary_requirements[1],
            stable_constraints_path,
        )
        nightly_python = prepare_environment(
            "nightly", architecture, temporary_requirements[0], temporary_requirements[1]
        )
        stable_results = run_analysis("stable", architecture, stable_python, args.mode)
        nightly_results = run_analysis("nightly", architecture, nightly_python, args.mode)

        comparison_file = ROOT / "comparison.md"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "comparison_generator.py"),
                str(stable_results),
                str(nightly_results),
                "--output",
                str(comparison_file),
            ],
            cwd=ROOT,
            check=True,
        )
        metrics = comparison_generator.aggregate_metrics(stable_results, nightly_results)
        torchvision_validation = {}
        cyan = "\033[36m"
        reset = "\033[0m"
        print(f"\n{cyan}{'=' * 78}{reset}")
        print(f"{cyan}Torchvision validation{reset}")
        for name, python in (
            ("stable", stable_python),
            ("nightly", nightly_python),
        ):
            passed, detail = validate_torchvision_runtime(python, architecture)
            torchvision_validation[name] = passed
            status_color = "\033[92m" if passed else "\033[91m"
            print(
                f"{status_color}Torchvision validation ({name}): "
                f"{'PASS' if passed else 'FAILED'} - {detail}\033[0m",
                flush=True,
            )
        print(f"{cyan}{'=' * 78}{reset}", flush=True)
        preferred = (
            "stable"
            if metrics["stable_average"] <= metrics["nightly_average"]
            else "nightly"
        )
        print_completion_summary(
            comparison_file,
            metrics,
            preferred,
            recommended_install_command(
                preferred,
                stable_results,
                nightly_results,
                all(torchvision_validation.values()),
            ),
        )
    finally:
        for requirements_file in temporary_requirements:
            requirements_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
