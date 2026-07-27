import json
import tempfile
import unittest
from pathlib import Path

from analyze_ba_competitiveness import (
    compare_records,
    read_labeled_section,
    render_markdown,
)


class AnalyzeBaCompetitivenessTest(unittest.TestCase):
    def test_reads_only_requested_legacy_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                '# old\n{"file_name":"old"}\n'
                '# pilot\n{"file_name":"superseded"}\n'
                '# next\n{"file_name":"excluded"}\n'
                '# pilot\n{"file_name":"kept"}\n'
            )

            records = read_labeled_section(path, "# pilot")

        self.assertEqual(records, [{"file_name": "kept"}])

    def test_comparison_reports_tradeoff_and_fairness_blockers(self):
        drs = [{
            "file_name": "problem-52-64053-pre.txt.bz2",
            "iterations": 90,
            "bestCost": 474033,
            "overallSeconds": 15.0,
            "partitionSeconds": 3.0,
            "kClusters": 30,
        }]
        schur = [{
            "algorithm": "schur_sim",
            "file_name": "problem-52-64053-pre.txt.bz2",
            "iterations": 20,
            "bestCost": 479793,
            "elapsedSeconds": 10.0,
            "edgePartitions": 30,
            "crossTermsVerified": False,
            "status": "completed",
        }]

        comparisons = compare_records(drs, schur)
        report = render_markdown(comparisons, "# pilot")

        self.assertEqual(comparisons[0]["pareto"], "tradeoff")
        self.assertAlmostEqual(comparisons[0]["drsTimeRatio"], 1.5)
        self.assertAlmostEqual(comparisons[0]["drsSolveTimeRatio"], 1.2)
        self.assertIn("Schur cross terms unverified", comparisons[0]["warnings"])
        self.assertIn("iteration budgets differ", report)


if __name__ == "__main__":
    unittest.main()