from build_ceres_se3_left_unified_benchmark import (
    DEFAULT_CARRYOVER_ROOT,
    build_summary,
)


def test_unified_benchmark_matches_authoritative_quality_and_proposals():
    summary = build_summary(DEFAULT_CARRYOVER_ROOT)
    one_d_sfm = summary["families"]["1dsfm"]
    bal = summary["families"]["bal"]

    assert one_d_sfm["candidate_over_ceres"]["geometric"] == 1.1696996553730585
    assert one_d_sfm["candidate_over_ceres"]["summed"] == 1.0465179061522758
    assert one_d_sfm["candidate_over_direct"]["geometric"] == 0.7629020121823338
    assert one_d_sfm["proposal"]["60"] == {
        "attempted": 15,
        "selected": 15,
        "no_op": 0,
    }
    assert one_d_sfm["proposal"]["90"] == {
        "attempted": 15,
        "selected": 8,
        "no_op": 7,
    }

    assert bal["candidate_over_ceres"]["geometric"] == 1.0011805476457836
    assert bal["candidate_over_ceres"]["summed"] == 0.9976962981047527
    assert bal["candidate_over_direct"]["geometric"] == 1.0001306589048602
    assert bal["proposal"]["60"] == {
        "attempted": 29,
        "selected": 1,
        "no_op": 28,
    }
    assert bal["proposal"]["90"] == {
        "attempted": 29,
        "selected": 0,
        "no_op": 29,
    }
    assert one_d_sfm["recovery_exhausted"]["candidate"] == []
    assert bal["recovery_exhausted"]["candidate"] == []
