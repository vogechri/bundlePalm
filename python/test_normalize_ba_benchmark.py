import json
import tempfile
import unittest
from pathlib import Path

from normalize_ba_benchmark import normalize_record, read_jsonl


class NormalizeBenchmarkTest(unittest.TestCase):
    def test_reader_ignores_legacy_comment_headers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                '# base\npython old_runner.py\n{"file_name":"problem.txt"}\n'
            )

            records = list(read_jsonl(path, allow_legacy_metadata=True))

        self.assertEqual(records, [{"file_name": "problem.txt"}])

    def test_reader_rejects_metadata_in_strict_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text("python old_runner.py\n")

            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                list(read_jsonl(path))

    def test_drs_preserves_native_result_without_inventing_metrics(self):
        native = {
            "base_url": "https://example.test/bal/",
            "file_name": "problem-49-7776-pre.txt.bz2",
            "iterations": 10,
            "bestCost": 1234,
            "bestIt": 7,
            "kClusters": 4,
            "accelerator": "nesterov",
        }

        normalized = normalize_record(native, "drs", Path("native.jsonl"))

        self.assertEqual(normalized["method"]["name"], "variable_metric_drs")
        self.assertEqual(normalized["configuration"]["partitions"], 4)
        self.assertEqual(normalized["outcome"]["objective"]["best"], 1234)
        self.assertFalse(normalized["outcome"]["objective"]["independentlyVerified"])
        self.assertIsNone(normalized["outcome"]["elapsedSeconds"])
        self.assertIsNone(normalized["distributedMetrics"]["bytesSent"])

    def test_native_timing_and_partition_fields_are_normalized(self):
        drs = normalize_record(
            {
                "file_name": "problem-49-7776-pre.txt.bz2",
                "iterations": 10,
                "kClusters": 4,
                "overallSeconds": 12.5,
                "partitionSeconds": 0.75,
            },
            "drs",
            Path("drs.jsonl"),
        )
        schur = normalize_record(
            {
                "algorithm": "schur_megba_edge_nesterov_sim",
                "file_name": "problem-49-7776-pre.txt.bz2",
                "iterations": 10,
                "edgePartitions": 8,
                "elapsedSeconds": 3.25,
            },
            "schur",
            Path("schur.jsonl"),
        )

        self.assertEqual(drs["configuration"]["partitions"], 4)
        self.assertEqual(drs["outcome"]["elapsedSeconds"], 12.5)
        self.assertEqual(drs["outcome"]["partitionSeconds"], 0.75)
        self.assertEqual(schur["configuration"]["partitions"], 8)
        self.assertEqual(schur["outcome"]["elapsedSeconds"], 3.25)

    def test_schur_derives_elapsed_time_and_count_from_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "trajectory.jsonl"
            trajectory.write_text(
                json.dumps({"iteration": 0, "cost": 8.0, "elapsedSeconds": 0.5})
                + "\n"
                + json.dumps({"iteration": 1, "cost": 4.0, "elapsedSeconds": 1.25})
                + "\n"
            )
            native = {
                "algorithm": "schur_pcg_cg",
                "base_url": "https://example.test/bal/",
                "file_name": "problem-49-7776-pre.txt.bz2",
                "iterations": 2,
                "bestCost": 4.0,
                "bestIt": 1,
                "status": "completed",
                "trajectory": str(trajectory),
            }

            normalized = normalize_record(native, "schur", Path("native.jsonl"))

        self.assertEqual(normalized["configuration"]["outerIterationsCompleted"], 2)
        self.assertEqual(normalized["outcome"]["elapsedSeconds"], 1.25)


if __name__ == "__main__":
    unittest.main()