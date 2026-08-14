#!/usr/bin/env python3
"""Analyze no-correction, reset, and product-transport mid-Schur arms."""

import argparse
import json
import math
import re
from pathlib import Path


EXPECTED = {
    "1dsfm": ("roman_forum", "trafalgar"),
    "bal": ("bal1490", "bal3068"),
}


def scene_key(row):
    dataset = Path(row["dataset"])
    if "sfm_init_1dsfm" in row["dataset"]:
        return dataset.parent.name
    match = re.search(r"problem-(\d+)-", dataset.name)
    if not match:
        raise ValueError(f"cannot identify dataset: {dataset}")
    return f"bal{match.group(1)}"


def load(directory):
    paths = sorted(directory.glob("*.jsonl"))
    if len(paths) != 1:
        raise ValueError(f"expected one JSONL in {directory}, found {len(paths)}")
    rows = {}
    for line in paths[0].read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            scene = scene_key(row)
            if scene in rows:
                raise ValueError(f"duplicate scene {scene} in {paths[0]}")
            rows[scene] = row
    return rows


def geometric_mean(values):
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def sse(row):
    return row["qualityMetrics"]["sumSquaredError"]


def validate(row, arm, context):
    correction_iteration = 0 if arm == "control" else 10
    transport = arm == "transport"
    expected = {
        "clusters": 24,
        "iterations": 30,
        "localSolver": "schur_pcg",
        "outerAcceleration": "none",
        "persistentTrustRegion": True,
        "sharedOnlyCameraProximal": True,
        "midSharedSchurCorrectionIteration": correction_iteration,
        "midSharedSchurTransportProductState": transport,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(
                f"configuration mismatch {context}: {key}={row.get(key)!r}, "
                f"expected={value!r}"
            )
    if row.get("completedIterations") != 30:
        raise ValueError(f"incomplete {context}: {row.get('terminationReason')}")
    if arm != "control":
        if not row.get("midSharedSchurAttempted"):
            raise ValueError(f"missing correction attempt {context}")
        expected_transport = transport and row.get("midSharedSchurAccepted")
        if row.get("midSharedSchurProductStateTransported") != expected_transport:
            raise ValueError(f"transport outcome mismatch {context}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=("control", "reset", "transport"))
    parser.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()

    rows = {}
    for arm in arguments.arms:
        rows[arm] = {}
        for family, scenes in EXPECTED.items():
            family_rows = load(arguments.root / arm / family)
            if arguments.require_complete and set(family_rows) != set(scenes):
                raise ValueError(f"coverage mismatch {arm}/{family}: {sorted(family_rows)}")
            for scene, row in family_rows.items():
                validate(row, arm, f"{arm}/{family}/{scene}")
            rows[arm][family] = family_rows

    summary = {}
    for arm in arguments.arms:
        summary[arm] = {}
        for family, scenes in EXPECTED.items():
            control = rows["control"][family]
            reset = rows["reset"][family]
            candidate = rows[arm][family]
            versus_control = [sse(candidate[x]) / sse(control[x]) for x in scenes]
            versus_reset = [sse(candidate[x]) / sse(reset[x]) for x in scenes]
            post_accepts = []
            post_rejections = []
            ties = [math.isclose(value, 1.0, rel_tol=1e-12) for value in versus_control]
            for scene in scenes:
                post = [
                    item for item in candidate[scene]["trajectory"]
                    if 10 <= item["iteration"] < 15
                ]
                post_accepts.append(sum(not item["rejected"] for item in post))
                post_rejections.append(sum(item["rejected"] for item in post))
            summary[arm][family] = {
                "geometricSSEVsControl": geometric_mean(versus_control),
                "geometricSSEVsReset": geometric_mean(versus_reset),
                "summedSSEVsControl": (
                    math.fsum(sse(candidate[x]) for x in scenes)
                    / math.fsum(sse(control[x]) for x in scenes)
                ),
                "winsTiesLossesVsControl": [
                    sum(value < 1.0 and not ties[index] for index, value in enumerate(versus_control)),
                    sum(ties),
                    sum(value > 1.0 and not ties[index] for index, value in enumerate(versus_control)),
                ],
                "optimizationSeconds": math.fsum(
                    candidate[x]["optimizationSeconds"] for x in scenes
                ),
                "postI10Accepts": sum(post_accepts),
                "postI10Rejections": sum(post_rejections),
                "scenes": {
                    scene: {
                        "sse": sse(candidate[scene]),
                        "sseVsControl": versus_control[index],
                        "sseVsReset": versus_reset[index],
                        "postI10Accepts": post_accepts[index],
                        "postI10Rejections": post_rejections[index],
                        "correctionAccepted": candidate[scene].get("midSharedSchurAccepted", False),
                        "offsetError": candidate[scene].get("midSharedSchurTransportOffsetError"),
                    }
                    for index, scene in enumerate(scenes)
                },
            }

    arguments.root.mkdir(parents=True, exist_ok=True)
    (arguments.root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = arguments.root / "report.md"
    with report.open("w", encoding="utf-8") as output:
        output.write("# Mid-Run Schur Product-State Transport Gate\n\n")
        output.write(
            "Fresh K24/I30 arms use no correction, the existing I10 reset, or "
            "the same accepted correction transported through local camera "
            "copies and centers with exact dual-offset and trust preservation.\n\n"
        )
        output.write(
            "| Family | Arm | SSE/control | SSE/reset | Summed/control | "
            "W/T/L vs control | Opt. s | I10--I14 accepts/rejects |\n"
        )
        output.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for family in ("1dsfm", "bal"):
            for arm in arguments.arms:
                values = summary[arm][family]
                wtl = "/".join(str(value) for value in values["winsTiesLossesVsControl"])
                output.write(
                    f"| {family} | {arm} | {values['geometricSSEVsControl']:.9f} | "
                    f"{values['geometricSSEVsReset']:.9f} | "
                    f"{values['summedSSEVsControl']:.9f} | {wtl} | "
                    f"{values['optimizationSeconds']:.3f} | "
                    f"{values['postI10Accepts']}/{values['postI10Rejections']} |\n"
                )
        output.write("\nAll settings are global; no scene selects an arm.\n")
        output.write(
            "\n## Decision\n\n"
            "Reject dual-preserving mid-Schur transport as a quality "
            "mechanism. It partially repairs the continuation failure: "
            "I10--I14 accepts/rejects improve from `3/7` to `6/4` on the "
            "1DSfM pair and from `6/4` to `9/1` on BAL. Trafalgar and BAL3068 "
            "continue more smoothly. This does not transfer to endpoint "
            "quality. Transport/reset geometric SSE is `1.020074x` on "
            "1DSfM and `1.002651x` on BAL. Roman remains at `1/4` "
            "post-correction accepts/rejects and ends `1.103501x` reset; "
            "BAL3068 ends `1.005310x` reset. BAL1490 rejects the global "
            "correction itself and therefore remains an exact reset/transport "
            "tie.\n\n"
            "Product-space collapse explains part of the post-correction "
            "stall, but preserving dual offsets and trust state does not "
            "produce a safer common basin. Retain the default-off transport "
            "path as state-transition diagnostic infrastructure; do not "
            "broaden or tune the I10 trigger.\n"
        )
    print(report)


if __name__ == "__main__":
    main()
