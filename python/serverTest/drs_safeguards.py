"""Numerical safeguards for accepting or rejecting DRS trial states."""

import numpy as np


def relative_safeguard_ratios(
    iteration,
    total_iterations,
    dre_increase_at_reference=0.01,
    reference_iteration=5,
    annealing_exponent=4.0,
    minimum_primal_ratio=1.001,
):
    """Return the annealed DRE and primal ratios used by client_acc.py."""
    if total_iterations <= 0 or not 0 <= iteration < total_iterations:
        raise ValueError("iteration must be inside the run")
    if (
        dre_increase_at_reference < 0.0
        or minimum_primal_ratio < 1.0
        or not np.isfinite(annealing_exponent)
        or annealing_exponent <= 0.0
        or reference_iteration < 0
    ):
        raise ValueError("relative safeguard tolerances are invalid")
    reference_iteration = min(reference_iteration, total_iterations - 1)
    iteration_factor = (
        1.0 - iteration / total_iterations
    ) ** annealing_exponent
    reference_factor = (
        1.0 - reference_iteration / total_iterations
    ) ** annealing_exponent
    dre_ratio = (
        1.0
        + dre_increase_at_reference * iteration_factor / reference_factor
    )
    primal_ratio = max(minimum_primal_ratio, np.sqrt(dre_ratio))
    return float(dre_ratio), float(primal_ratio)


def should_reject_trial(
    line_search_iteration,
    line_search_iterations,
    dre,
    primal_cost,
    reference_dre,
    reference_primal_cost,
    max_dre_ratio,
    max_primal_ratio,
    relative_deadband=0.0,
):
    """Reject a final trial that is non-finite or worse than both references."""
    if line_search_iteration != line_search_iterations - 1:
        return False
    if not np.isfinite(dre) or not np.isfinite(primal_cost):
        return True
    if relative_deadband < 0.0 or not np.isfinite(relative_deadband):
        raise ValueError("relative_deadband must be finite and nonnegative")
    return (
        exceeds_with_relative_deadband(
            dre, max_dre_ratio * reference_dre, relative_deadband
        )
        and exceeds_with_relative_deadband(
            primal_cost,
            max_primal_ratio * reference_primal_cost,
            relative_deadband,
        )
    )


def bootstrap_basin_guard_decision(candidate_sse, ceiling_sse, active):
    """Return rejection and release decisions for an accepted bootstrap basin."""
    if not active:
        return False, False
    if not np.isfinite(ceiling_sse) or ceiling_sse < 0.0:
        raise ValueError("bootstrap basin ceiling must be finite and nonnegative")
    if not np.isfinite(candidate_sse):
        return True, False
    if candidate_sse < ceiling_sse:
        return False, True
    return True, False


def exceeds_with_relative_deadband(value, threshold, relative_deadband):
    """Return whether a finite value exceeds a threshold beyond roundoff slack."""
    if relative_deadband < 0.0 or not np.isfinite(relative_deadband):
        raise ValueError("relative_deadband must be finite and nonnegative")
    margin = relative_deadband * max(1.0, abs(value), abs(threshold))
    return bool(value > threshold + margin)


def increase_recovery_parameter(current_value, maximum_value, recovery_ratio):
    """Increase a proximal control and report whether recovery changed it."""
    if current_value <= 0.0 or maximum_value < current_value:
        raise ValueError("recovery parameter bounds are invalid")
    if recovery_ratio <= 1.0:
        raise ValueError("recovery ratio must exceed one")
    next_value = min(maximum_value, current_value * recovery_ratio)
    return float(next_value), bool(next_value > current_value)
