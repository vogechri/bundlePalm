import numpy as np
import pytest

from drs_safeguards import (
    increase_recovery_parameter,
    relative_safeguard_ratios,
    should_reject_trial,
)


def test_relative_ratios_match_client_acc_schedule():
    dre_zero, primal_zero = relative_safeguard_ratios(0, 30)
    dre_five, primal_five = relative_safeguard_ratios(5, 30)
    dre_last, primal_last = relative_safeguard_ratios(29, 30)

    assert dre_five == pytest.approx(1.01)
    assert primal_five == pytest.approx(np.sqrt(1.01))
    assert dre_zero > dre_five > dre_last > 1.0
    assert primal_last == pytest.approx(1.001)


def test_relative_rejection_requires_both_merit_and_primal_worsening():
    assert should_reject_trial(0, 1, 102.0, 101.0, 100.0, 100.0, 1.01, 1.005)
    assert not should_reject_trial(0, 1, 100.5, 101.0, 100.0, 100.0, 1.01, 1.005)
    assert not should_reject_trial(0, 1, 102.0, 100.1, 100.0, 100.0, 1.01, 1.005)


def test_nonfinite_relative_candidate_is_always_rejected():
    assert should_reject_trial(0, 1, np.inf, 1.0, 1.0, 1.0, 1.01, 1.005)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_final_trial_is_rejected(value):
    assert should_reject_trial(1, 2, value, 10.0, 5.0, 5.0, 1.01, 1.001)
    assert should_reject_trial(1, 2, 10.0, value, 5.0, 5.0, 1.01, 1.001)


def test_catastrophic_final_trial_is_rejected_without_metric_ceiling_gate():
    assert should_reject_trial(1, 2, 1000.0, 1000.0, 10.0, 10.0, 1.01, 1.001)


def test_intermediate_or_single_metric_failure_is_not_rejected():
    assert not should_reject_trial(0, 2, 1000.0, 1000.0, 10.0, 10.0, 1.01, 1.001)
    assert not should_reject_trial(1, 2, 1000.0, 10.0, 10.0, 10.0, 1.01, 1.001)


def test_recovery_parameter_increases_up_to_ceiling():
    assert increase_recovery_parameter(0.25, 1.0, 2.0) == (0.5, True)
    assert increase_recovery_parameter(0.75, 1.0, 2.0) == (1.0, True)


def test_recovery_parameter_reports_exhausted_ceiling():
    assert increase_recovery_parameter(1.0, 1.0, 2.0) == (1.0, False)
