#!/usr/bin/env python3
"""Run isolated K1-to-DRS bridge arms on the mature DRS backbone."""

import argparse
import os
import sys
from pathlib import Path


BASELINE_ARGUMENTS = (
    "--local-solver", "nesterov",
    "--nesterov-max-iterations", "300",
    "--enhanced-inner-max-iterations", "300",
    "--nesterov-min-iterations", "1",
    "--nesterov-stop-tolerance", "1e-2",
    "--enhanced-inner-until", "30",
    "--trust-region-policy", "drs",
    "--persistent-trust-region",
    "--camera-scaling", "jacobi_initial",
    "--scene-normalization", "points_p95",
    "--camera-diagonal-relative-floor", "1e-48",
    "--camera-diagonal-metric-scale", "75",
    "--outer-acceleration", "nesterov",
    "--line-search-grid", "0,1",
    "--acceleration-restart-after", "3",
    "--proximal-metric", "block",
    "--consensus-metric", "full",
    "--block-regularization", "1e-4",
    "--block-curvature-multiplier", "0.4",
    "--block-recovery-mode", "curvature",
    "--maximum-block-curvature-multiplier", "64",
    "--curvature-decay-after", "5",
    "--curvature-decay-ratio", "0.5",
    "--metric-proposal-disagreement-scale", "0.5",
    "--safeguard-mode", "relative",
    "--dre-relative-increase", "0.01",
    "--minimum-primal-ratio", "1.001",
    "--recovery-penalty-ratio", "2",
    "--worker-owned-landmarks",
    "--packed-request-buffers",
)


def build_invocation(arguments, environment=None):
    child_environment = dict(os.environ if environment is None else environment)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--bridge-arm",
        choices=(
            "legacy", "direct", "direct_shared", "direct_shared_diag",
            "direct_shared_l2",
        ),
        default=child_environment.get("BUNDLE_PALM_K1_BRIDGE_ARM", "legacy"),
    )
    bridge, forwarded = parser.parse_known_args(arguments)
    base_client = child_environment.get(
        "BUNDLE_PALM_BASE_CLIENT_DRS",
        str(Path(__file__).with_name("client_drs.py")),
    )
    command = [
        sys.executable,
        base_client,
        *BASELINE_ARGUMENTS,
    ]
    if bridge.bridge_arm in (
        "direct_shared", "direct_shared_diag", "direct_shared_l2"
    ):
        command.append("--shared-only-camera-proximal")
    if bridge.bridge_arm == "direct_shared_diag":
        command.append("--interior-defect-diagnostic")
    if bridge.bridge_arm == "direct_shared_l2":
        command.extend(("--local-steps", "2"))
    command.extend(forwarded)

    child_environment.update({
        "BUNDLE_PALM_CAMERA_UPDATE": "se3_left",
        "BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS": (
            "0" if bridge.bridge_arm == "legacy" else "1"
        ),
        "BUNDLE_PALM_DIAGONAL_TRUST_DAMPING": "0",
        "BUNDLE_PALM_DISABLE_LANDMARK_PRECONDITIONING": "0",
        "BUNDLE_PALM_BAE_TRUST_SCHEDULE": "0",
        "BUNDLE_PALM_CUMULATIVE_DIAGONAL_DAMPING": "0",
        "BUNDLE_PALM_DABA_INITIAL_TRUST_REGION_CAP": "100",
    })
    return bridge.bridge_arm, command, child_environment


def main():
    bridge_arm, command, environment = build_invocation(sys.argv[1:])
    environment["BUNDLE_PALM_K1_BRIDGE_ARM"] = bridge_arm
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    main()