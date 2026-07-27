from analyze_admm_innovation_matrix import (
    best_pixel_metrics_through,
    best_sse_through,
    format_integer,
    render_markdown,
    summarize,
    time_to_target,
)


def row(variant, dataset, clusters, final_sse, trajectory, initial_sse=None):
    result = {
        "variant": variant,
        "dataset": dataset,
        "clusters": clusters,
        "qualityMetrics": {
            "sumSquaredError": final_sse,
            "meanReprojectionError": 1.0,
        },
        "overallSeconds": trajectory[-1]["overallSeconds"],
        "localSteps": 1,
        "alpha": 1.0,
        "trajectory": trajectory,
    }
    if initial_sse is not None:
        result["initialQualityMetrics"] = {
            "sumSquaredError": initial_sse,
        }
    return result


def test_time_to_target_returns_first_crossing_or_infinity():
    candidate = row("x", "a.txt", 5, 8.0, [
        {"sumSquaredError": 20.0, "overallSeconds": 1.0},
        {"sumSquaredError": 9.0, "overallSeconds": 2.0},
        {"sumSquaredError": 8.0, "overallSeconds": 3.0},
    ])
    assert time_to_target(candidate, 10.0) == 2.0
    assert time_to_target(candidate, 7.0) == float("inf")


def test_time_to_target_includes_initial_state_at_time_zero():
    candidate = row("x", "a.txt", 5, 10.0, [
        {"sumSquaredError": 20.0, "overallSeconds": 1.0},
    ], initial_sse=10.0)

    assert time_to_target(candidate, 10.0) == 0.0


def test_best_sse_through_uses_initial_and_requested_prefix():
    candidate = row("x", "a.txt", 5, 7.0, [
        {"sumSquaredError": 12.0, "meanReprojectionError": 1.2, "overallSeconds": 1.0},
        {"sumSquaredError": 8.4, "meanReprojectionError": 0.84, "overallSeconds": 2.0},
        {"sumSquaredError": 9.0, "meanReprojectionError": 0.9, "overallSeconds": 3.0},
        {"sumSquaredError": 7.0, "meanReprojectionError": 0.7, "overallSeconds": 4.0},
    ], initial_sse=10.0)

    assert best_sse_through(candidate, 3) == 8.4
    assert best_pixel_metrics_through(candidate, 3) == (8.4, 0.84)
    assert best_sse_through(candidate, 4) == 7.0
    assert format_integer(1234567.6) == "1,234,568"


def test_report_shows_best_iteration_and_integer_sse_at_checkpoint():
    trajectory = [
        {
            "sumSquaredError": 100.0 - index,
            "meanReprojectionError": (100.0 - index) / 100.0,
            "overallSeconds": index + 1.0,
        }
        for index in range(60)
    ]
    candidate = row(
        "baseline", "a.txt", 10, 41.0, trajectory, initial_sse=110.0)
    summaries, by_variant = summarize([candidate])

    report = render_markdown(summaries, by_variant, checkpoint=30)

    assert "independently evaluated on the global camera" in report
    assert "worker-local surrogate costs are not" in report
    assert "| Local LM steps | Alpha | Best iteration <=30 |" in report
    assert "| Time at 30 s | Communication at 30 MiB |" in report
    assert "| baseline | a.txt | 10 | 1 | 1.0 | 29 | 71 | 0.7100 | 71 | 1.00 |" in report


def test_greedy_eligibility_requires_all_paired_targets():
    rows = [
        row("baseline", "a.txt", 5, 10.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 10.0, "overallSeconds": 4.0},
        ]),
        row("baseline", "b.txt", 5, 10.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 10.0, "overallSeconds": 4.0},
        ]),
        row("good", "a.txt", 5, 8.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 8.0, "overallSeconds": 2.0},
        ]),
        row("good", "b.txt", 5, 8.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 8.0, "overallSeconds": 2.0},
        ]),
        row("unstable", "a.txt", 5, 12.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 12.0, "overallSeconds": 2.0},
        ]),
        row("unstable", "b.txt", 5, 8.0, [
            {"sumSquaredError": 20.0, "overallSeconds": 1.0},
            {"sumSquaredError": 8.0, "overallSeconds": 2.0},
        ]),
    ]

    summaries, _ = summarize(rows)
    summaries = {summary["variant"]: summary for summary in summaries}

    assert summaries["good"]["eligibleForGreedy"] is True
    assert summaries["good"]["targetsSolved"] == 2
    assert summaries["good"]["geomeanSpeedupToBaselineTarget"] == 2.0
    assert summaries["unstable"]["eligibleForGreedy"] is False
    assert summaries["unstable"]["targetsSolved"] == 1
