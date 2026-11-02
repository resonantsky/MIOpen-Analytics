#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_PACKAGE_LINE = re.compile(
    r"^([a-zA-Z0-9_.-]+)\s+([a-zA-Z0-9_.+!-]+)\s*$", re.MULTILINE
)
_LATENCY = re.compile(r"([\d.]+)\s*ms")
_BENCHMARK_SEPARATOR = re.compile(r"(?m)^\s*={10,}\s*$")
_CONFIGURATION = re.compile(r"^conv[123]d\b.*$", re.IGNORECASE)
_MIOPEN_HEADING = re.compile(r"^(?:native\s*:|miopen\s*:)", re.IGNORECASE)
_MEASURED_LATENCY = re.compile(r"^\s*Measured latency\s*:?\s*(.*)$", re.IGNORECASE)


def strip_ansi(text):
    return _ANSI_ESCAPE.sub("", text)


def parse_file(filepath):
    content = strip_ansi(Path(filepath).read_text(encoding="utf-8"))
    
    packages = {}
    for package, version in _PACKAGE_LINE.findall(content):
        if package.lower() != "package" and not set(package) <= {"-"}:
            packages[package.lower()] = version
        
    benchmarks = []
    blocks = _BENCHMARK_SEPARATOR.split(content)
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        config = next((line for line in lines if _CONFIGURATION.fullmatch(line)), None)
        if config is None:
            continue

        miopen_index = next(
            (index for index, line in enumerate(lines) if _MIOPEN_HEADING.match(line)),
            None,
        )
        if miopen_index is None:
            continue

        for line in lines[miopen_index + 1 :]:
            latency_line = _MEASURED_LATENCY.match(line)
            if latency_line is None:
                continue
            match = _LATENCY.search(latency_line.group(1))
            if match:
                benchmarks.append({
                    "config": config,
                    "native_latency": float(match.group(1)),
                })
            break
                
    return packages, benchmarks


def read_metadata(filepath, field):
    for line in Path(filepath).read_text(encoding="utf-8").splitlines():
        label, separator, value = line.partition(":")
        if separator and label.strip().lower() == field.lower():
            return value.strip()
    return None


def _average_by_config(benchmarks):
    samples = {}
    for benchmark in benchmarks:
        samples.setdefault(benchmark["config"], []).append(benchmark["native_latency"])
    return {
        config: sum(latencies) / len(latencies)
        for config, latencies in samples.items()
    }


def aggregate_metrics(file1, file2):
    _, stable_benchmarks = parse_file(file1)
    _, nightly_benchmarks = parse_file(file2)
    stable_by_config = _average_by_config(stable_benchmarks)
    nightly_by_config = _average_by_config(nightly_benchmarks)
    common_configs = sorted(set(stable_by_config) & set(nightly_by_config))
    if not common_configs:
        raise ValueError("The input files contain no matching native benchmark configurations.")

    stable_total = sum(stable_by_config[config] for config in common_configs)
    nightly_total = sum(nightly_by_config[config] for config in common_configs)
    return {
        "stable_by_config": stable_by_config,
        "nightly_by_config": nightly_by_config,
        "common_configs": common_configs,
        "configuration_count": len(common_configs),
        "stable_total": stable_total,
        "stable_average": stable_total / len(common_configs),
        "nightly_total": nightly_total,
        "nightly_average": nightly_total / len(common_configs),
    }


def generate_markdown(file1, file2):
    pkgs1, _ = parse_file(file1)
    pkgs2, _ = parse_file(file2)

    all_pkgs = sorted(set(pkgs1) | set(pkgs2))
    metrics = aggregate_metrics(file1, file2)
    benchmarks1 = metrics["stable_by_config"]
    benchmarks2 = metrics["nightly_by_config"]
    common_configs = metrics["common_configs"]
    total_lat1 = metrics["stable_total"]
    avg_lat1 = metrics["stable_average"]
    total_lat2 = metrics["nightly_total"]
    avg_lat2 = metrics["nightly_average"]
    pct_gain = ((avg_lat1 - avg_lat2) / avg_lat1 * 100) if avg_lat1 > 0 else 0

    md = []
    md.append(f"# Performance Comparison Report: {Path(file1).name} vs {Path(file2).name}\n")
    stable_architecture = read_metadata(file1, "Architecture")
    nightly_architecture = read_metadata(file2, "Architecture")
    if stable_architecture or nightly_architecture:
        md.append(
            f"**Architecture:** stable `{stable_architecture or 'not recorded'}`; "
            f"nightly `{nightly_architecture or 'not recorded'}`"
        )
    stable_repository = read_metadata(file1, "Repository")
    nightly_repository = read_metadata(file2, "Repository")
    if stable_repository or nightly_repository:
        md.append(
            f"**Repositories:** stable `{stable_repository or 'not recorded'}`; "
            f"nightly `{nightly_repository or 'not recorded'}`"
        )
    if stable_architecture or nightly_architecture or stable_repository or nightly_repository:
        md.append("")

    md.append("## 1. Version Comparison (`pip list`)")
    md.append(f"| Package | Stable (`{Path(file1).name}`) | Nightly (`{Path(file2).name}`) |")
    md.append("| :--- | :--- | :--- |")
    for pkg in all_pkgs:
        v1 = pkgs1.get(pkg, "*Not present*")
        v2 = pkgs2.get(pkg, "*Not present*")
        md.append(f"| **`{pkg}`** | `{v1}` | `{v2}` |")
    md.append("")

    md.append("## 2. Matched Native Solver Latencies")
    md.append("| Benchmark configuration | Stable (ms) | Nightly (ms) | Nightly - stable (ms) | Nightly latency reduction |")
    md.append("| :--- | ---: | ---: | ---: | ---: |")
    for config in sorted(set(benchmarks1) | set(benchmarks2)):
        stable_latency = benchmarks1.get(config)
        nightly_latency = benchmarks2.get(config)
        stable_value = f"{stable_latency:.3f}" if stable_latency is not None else "Not measured"
        nightly_value = f"{nightly_latency:.3f}" if nightly_latency is not None else "Not measured"
        if stable_latency is not None and nightly_latency is not None:
            difference = nightly_latency - stable_latency
            reduction = (
                f"{(stable_latency - nightly_latency) / stable_latency * 100:.2f}%"
                if stable_latency > 0
                else "n/a"
            )
            difference_value = f"{difference:+.3f}"
        else:
            difference_value = "n/a"
            reduction = "n/a"
        md.append(
            f"| `{config}` | {stable_value} | {nightly_value} | "
            f"{difference_value} | {reduction} |"
        )

    md.append("")
    md.append(f"Aggregate metrics use the {len(common_configs)} configurations measured in both files.")
    md.append(f"* **Stable total / average:** `{total_lat1:.3f} ms` / `{avg_lat1:.3f} ms`")
    md.append(f"* **Nightly total / average:** `{total_lat2:.3f} ms` / `{avg_lat2:.3f} ms`")
    if set(benchmarks1) != set(benchmarks2):
        md.append("* Configurations missing from one run are shown as `Not measured` and excluded from aggregate metrics.")
    winner = "Nightly" if avg_lat2 < avg_lat1 else "Stable"
    if avg_lat1 == avg_lat2:
        winner = "Tie"
    md.append(f"* **Lower average latency:** {winner}")
    md.append(f"* **Nightly latency reduction on matched configurations:** `{pct_gain:.2f}%`")

    return "\n".join(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare stable and nightly MIOpen benchmark results.")
    parser.add_argument("stable_file")
    parser.add_argument("nightly_file")
    parser.add_argument("--output", help="write the report to this Markdown file")
    args = parser.parse_args()

    report = generate_markdown(args.stable_file, args.nightly_file)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")
    else:
        print(report)