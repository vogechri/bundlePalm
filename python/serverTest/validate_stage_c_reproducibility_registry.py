#!/usr/bin/env python3
"""Validate the Stage-C reproducibility registry and artifact boundary."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = ROOT / "benchmark_results/stage_c_reproducibility_registry.json"
EXPECTED_MODES = {
    "plain", "c1", "c5", "c1_c5", "k4", "k16", "product_so3", "startup_bootstrap"
}
TABLE_SOURCES = (
    ROOT / "stage_c_long_horizon_results.tex",
    ROOT / "benchmark_results/stage_c_publication_comparison/final_complete_cohort_table.tex",
    ROOT / "benchmark_results/stage_c_publication_comparison/k1_carryover_factorial_table.tex",
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*arguments):
    return subprocess.check_output(("git", *arguments), cwd=ROOT, text=True).strip()


def validate(registry_path):
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported registry schema")
    checkpoint = registry.get("validated_commit")
    subprocess.check_call(
        ("git", "merge-base", "--is-ancestor", checkpoint, "HEAD"),
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
    )

    worker = ROOT / registry["worker"]["path"]
    if not worker.is_file() or sha256(worker) != registry["worker"]["sha256"]:
        raise ValueError("worker hash mismatch")

    modes = registry.get("modes", {})
    if set(modes) != EXPECTED_MODES:
        raise ValueError("canonical mode coverage mismatch")
    for mode, record in modes.items():
        for field in ("role", "report", "artifact", "command"):
            if not record.get(field):
                raise ValueError(f"missing {field} for {mode}")
        for field in ("report", "artifact"):
            if not (ROOT / record[field]).exists():
                raise FileNotFoundError(f"missing {mode} {field}: {record[field]}")
        if "OVERWRITE=0" not in record["command"]:
            raise ValueError(f"canonical command is not resumable for {mode}")

    labels = set()
    for path in TABLE_SOURCES:
        labels.update(re.findall(r"\\label\{(tab:[^}]+)\}", path.read_text(encoding="utf-8")))
    tables = registry.get("tables", {})
    if set(tables) != labels:
        raise ValueError(
            f"manuscript table coverage mismatch: registry={sorted(tables)} source={sorted(labels)}"
        )
    for label, record in tables.items():
        for field in ("report", "artifact"):
            if not (ROOT / record[field]).exists():
                raise FileNotFoundError(f"missing {label} {field}: {record[field]}")

    tracked = git_output("ls-files", "benchmark_results").splitlines()
    forbidden = [
        path for path in tracked
        if any(component in {"logs", "states", "memory"} for component in Path(path).parts)
    ]
    if forbidden:
        raise ValueError(f"raw run trees are tracked: {forbidden[:5]}")
    return {
        "modes": len(modes),
        "tables": len(tables),
        "maintained_tests": registry["environment"]["maintained_tests"],
        "worker_sha256": registry["worker"]["sha256"],
        "tracked_benchmark_files": len(tracked),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    arguments = parser.parse_args()
    print(json.dumps(validate(arguments.registry), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
