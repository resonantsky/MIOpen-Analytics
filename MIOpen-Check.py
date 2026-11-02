#!/usr/bin/env python3
"""Install stable/nightly ROCm environments and compare their MIOpen benchmarks."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
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


def _wheel_matches_runtime(filename: str) -> Tuple[bool, Optional[str]]:
    try:
        prefix, python_tag, abi_tag, platform_tag = filename[:-4].rsplit("-", 3)
        _, version = prefix.rsplit("-", 1)
    except ValueError:
        return False, None

    major, minor = sys.version_info[:2]
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
    if "any" not in platform_tags and runtime_platform not in platform_tags:
        return False, None
    return True, unquote(version)


def _probe_package_artifact(
    index_url: str, package: str, required_version: Optional[str]
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
        if compatible and version is not None:
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

    results = {}
    print(
        f"Checking required AMD package artifacts for Python {sys.version_info.major}."
        f"{sys.version_info.minor} ({sysconfig.get_platform()})...",
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

    print("AMD package availability (stable + nightly):")
    for prefix in ("gfx10", "gfx11", "gfx12", "gfx9"):
        architectures = [
            architecture
            for architecture in ARCHITECTURES
            if architecture.startswith(prefix) and architecture in availability
        ]
        if not architectures:
            continue
        colored_architectures = (
            f"{colors[architecture_wheel_status(availability[architecture])]}"
            f"{architecture}{reset}"
            for architecture in architectures
        )
        print(f"  {prefix}: " + ", ".join(colored_architectures))


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
) -> list[str]:
    if UV_EXECUTABLE is not None:
        command = [
            UV_EXECUTABLE,
            "pip",
            "install",
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
    if extra_index_url is not None:
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
            output.write(line)
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
        if package in AMD_PACKAGE_NAMES or package in architecture_packages
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
) -> Path:
    environment = ROOT / f"venv-{name}"
    if not venv_python(environment).is_file():
        print(f"Creating {name} virtual environment at {environment}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", str(environment)],
            cwd=ROOT,
            check=True,
        )

    repositories = REPOSITORIES[name]
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
            repositories["core"],
            constraints,
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


def run_analysis(name: str, architecture: str, python: Path, mode: str) -> Path:
    results_file = ROOT / f"{name}-results.txt"
    check_script = ROOT / f"miopen-check-{mode}.py"
    print(f"Running {name} {mode} benchmark; recording {results_file.name}", flush=True)
    with results_file.open("w", encoding="utf-8") as output:
        output.write(
            f"MIOpen benchmark results\n"
            f"Repository: {REPOSITORIES[name]['pytorch']}\n"
            f"Core repository: {REPOSITORIES[name]['core']}\n"
            f"Architecture: {architecture}\n"
            f"Benchmark: {check_script.name}\n\n"
            "=== Installed packages ===\n"
        )
        subprocess.run(
            [str(python), "-m", "pip", "list", "--format=columns"],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
        output.write("\n=== Benchmark output ===\n")
        output.flush()
        print(f"\n=== {name.title()} benchmark output (also saved to {results_file.name}) ===")
        run_and_tee([str(python), "-u", str(check_script)], output)
    return results_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("normal", "verbose"),
        default="normal",
        help="benchmark output mode (default: normal)",
    )
    args = parser.parse_args()

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
