import unittest

from client_drs_k1_bridge import BASELINE_ARGUMENTS, build_invocation


class K1BridgeInvocationTest(unittest.TestCase):
    def test_arms_change_only_k1_factors(self):
        invocations = {}
        for arm in (
            "legacy", "direct", "direct_shared", "direct_shared_diag",
            "direct_shared_l2",
        ):
            selected, command, environment = build_invocation(
                ["--bridge-arm", arm, "scene.txt", "--iterations", "30"],
                {"BUNDLE_PALM_BASE_CLIENT_DRS": "/clean/client_drs.py"},
            )
            self.assertEqual(selected, arm)
            self.assertEqual(command[1], "/clean/client_drs.py")
            self.assertEqual(
                environment["BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS"],
                "0" if arm == "legacy" else "1",
            )
            self.assertEqual(
                "--shared-only-camera-proximal" in command,
                arm in ("direct_shared", "direct_shared_diag", "direct_shared_l2"),
            )
            self.assertEqual(
                "--interior-defect-diagnostic" in command,
                arm == "direct_shared_diag",
            )
            self.assertEqual(
                command[command.index("--local-steps") + 1]
                if "--local-steps" in command else "1",
                "2" if arm == "direct_shared_l2" else "1",
            )
            invocations[arm] = command

        legacy = invocations["legacy"]
        direct = invocations["direct"]
        self.assertEqual(legacy, direct)
        self.assertEqual(tuple(legacy[2:2 + len(BASELINE_ARGUMENTS)]), BASELINE_ARGUMENTS)

    def test_user_arguments_follow_defaults(self):
        _, command, _ = build_invocation(
            ["scene.txt", "--camera-diagonal-metric-scale", "50"], {}
        )
        self.assertEqual(command[-3:], [
            "scene.txt", "--camera-diagonal-metric-scale", "50"
        ])

    def test_environment_selects_default_arm(self):
        selected, command, environment = build_invocation(
            ["scene.txt"],
            {
                "BUNDLE_PALM_BASE_CLIENT_DRS": "/clean/client_drs.py",
                "BUNDLE_PALM_K1_BRIDGE_ARM": "direct",
            },
        )
        self.assertEqual(selected, "direct")
        self.assertEqual(
            environment["BUNDLE_PALM_DIRECT_TANGENT_NORMAL_EQUATIONS"], "1"
        )
        self.assertNotIn("--shared-only-camera-proximal", command)


if __name__ == "__main__":
    unittest.main()