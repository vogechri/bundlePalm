"""Safeguarded accelerators for the outer DRS fixed-point iteration.

Configuration environment variables:

* BUNDLE_PALM_ACCEL_MAX_STEP_RATIO (default 10)
* BUNDLE_PALM_LBFGS_MEMORY (default 6)
* BUNDLE_PALM_LBFGS_CURVATURE_TOL (default 1e-10)
* BUNDLE_PALM_LBFGS_SCALE_MIN / SCALE_MAX (defaults 0.1 / 10)
* BUNDLE_PALM_ANDERSON_MEMORY (default 6)
* BUNDLE_PALM_ANDERSON_REGULARIZATION (default 1e-4)
* BUNDLE_PALM_ANDERSON_DAMPING (default 1)
* BUNDLE_PALM_NESTEROV_MAX_BETA (default 0.9)
"""

from collections import deque
import os

import numpy as np


def _env_float(name, default):
    value = float(os.environ.get(name, default))
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _env_int(name, default, minimum=1):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def interpolate_line_search_center(nominal, accelerated, weight):
    """Interpolate from the nominal DRS image to an accelerated center."""
    if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("line-search weight must be in [0, 1]")
    nominal = np.asarray(nominal, dtype=float)
    accelerated = np.asarray(accelerated, dtype=float)
    if nominal.shape != accelerated.shape:
        raise ValueError("line-search centers must have equal shape")
    return nominal + weight * (accelerated - nominal)


class FixedPointAccelerator:
    """Base class for proposals around the fixed-point map ``mapped = T(x)``."""

    name = "none"

    def __init__(self):
        self.max_step_ratio = _env_float(
            "BUNDLE_PALM_ACCEL_MAX_STEP_RATIO", "10.0")
        self.last_proposal_accelerated = False
        self.current_residual = None
        self.proposal_direction = None
        if self.max_step_ratio < 1.0:
            raise ValueError(
                "BUNDLE_PALM_ACCEL_MAX_STEP_RATIO must be at least 1")

    def reset(self):
        """Discard history after a rejected or externally restored state."""
        self.last_proposal_accelerated = False
        self.current_residual = None
        self.proposal_direction = None

    def accepted(self, accelerated):
        """Observe whether the accelerated proposal won the line search."""

    def observe_first_trial(self, trial_residual):
        """Update history from the paper's full-step trial residual."""

    def _candidate(self, current, mapped, iteration):
        return mapped

    def propose(self, current, mapped, iteration):
        current = np.asarray(current, dtype=float)
        mapped = np.asarray(mapped, dtype=float)
        if current.shape != mapped.shape:
            raise ValueError("current and mapped states must have equal shape")

        plain_step = mapped - current
        candidate = np.asarray(
            self._candidate(current, mapped, iteration), dtype=float)
        if candidate.shape != current.shape or not np.all(np.isfinite(candidate)):
            return mapped.copy(), False

        candidate_step = candidate - current
        plain_norm = np.linalg.norm(plain_step)
        candidate_norm = np.linalg.norm(candidate_step)
        step_limit = self.max_step_ratio * max(plain_norm, np.finfo(float).eps)
        if candidate_norm > step_limit:
            candidate = current + candidate_step * (step_limit / candidate_norm)

        self.current_residual = current - mapped
        self.proposal_direction = candidate - current
        self.last_proposal_accelerated = not np.array_equal(candidate, mapped)
        return candidate, self.last_proposal_accelerated


class NoAcceleration(FixedPointAccelerator):
    name = "none"


class LegacyNesterov(FixedPointAccelerator):
    """The momentum recurrence used by the original client_acc.py."""

    name = "nesterov"

    def __init__(self):
        super().__init__()
        self.previous_direction = None
        self.restart_iteration = 0

    def reset(self):
        super().reset()
        self.previous_direction = None

    def _candidate(self, current, mapped, iteration):
        plain_step = mapped - current
        if self.previous_direction is None:
            direction = plain_step
            self.restart_iteration = iteration
        else:
            relative_iteration = iteration - self.restart_iteration
            beta = (relative_iteration - 1.0) / (relative_iteration + 2.0)
            direction = plain_step + beta * self.previous_direction
        self.previous_direction = direction.copy()
        return current + direction


class InertialNesterov(FixedPointAccelerator):
    """FISTA-style inertia applied to consecutive plain fixed-point images."""

    name = "fista"

    def __init__(self, adaptive=False):
        super().__init__()
        self.adaptive = adaptive
        self.previous_mapped = None
        self.fista_t = 1.0

    def reset(self):
        super().reset()
        self.previous_mapped = None
        self.fista_t = 1.0

    def accepted(self, accelerated):
        if self.adaptive and self.last_proposal_accelerated and not accelerated:
            self.reset()

    def _candidate(self, current, mapped, iteration):
        if self.previous_mapped is None:
            self.previous_mapped = mapped.copy()
            return mapped

        next_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * self.fista_t**2))
        beta = (self.fista_t - 1.0) / next_t
        if self.adaptive:
            beta = min(beta, _env_float(
                "BUNDLE_PALM_NESTEROV_MAX_BETA", "0.9"))
        candidate = mapped + beta * (mapped - self.previous_mapped)
        self.previous_mapped = mapped.copy()
        self.fista_t = next_t
        return candidate


class ThemelisNesterov(FixedPointAccelerator):
    """Fast-DRS direction from Section 3.3.1 of the Themelis paper."""

    name = "themelis_nesterov"

    def __init__(self):
        super().__init__()
        self.previous_mapped = None
        self.restart_iteration = 0

    def reset(self):
        super().reset()
        self.previous_mapped = None

    def _candidate(self, current, mapped, iteration):
        if self.previous_mapped is None:
            self.previous_mapped = mapped.copy()
            self.restart_iteration = iteration
            return mapped

        relative_iteration = iteration - self.restart_iteration
        beta = (relative_iteration - 1.0) / (relative_iteration + 2.0)
        candidate = mapped + beta * (mapped - self.previous_mapped)
        self.previous_mapped = mapped.copy()
        return candidate


class LBFGSAcceleration(FixedPointAccelerator):
    """Paper-style L-BFGS using p=d and first-trial residual differences."""

    name = "lbfgs"

    def __init__(self):
        super().__init__()
        self.memory = _env_int("BUNDLE_PALM_LBFGS_MEMORY", "6")
        self.curvature_tolerance = _env_float(
            "BUNDLE_PALM_LBFGS_CURVATURE_TOL", "1e-10")
        self.initial_scale_min = _env_float(
            "BUNDLE_PALM_LBFGS_SCALE_MIN", "0.1")
        self.initial_scale_max = _env_float(
            "BUNDLE_PALM_LBFGS_SCALE_MAX", "10.0")
        self.steps = deque(maxlen=self.memory)
        self.residual_differences = deque(maxlen=self.memory)

    def reset(self):
        super().reset()
        self.steps.clear()
        self.residual_differences.clear()

    def _candidate(self, current, mapped, iteration):
        residual = current - mapped
        if not self.steps:
            return mapped

        vector = residual.copy()
        coefficients = []
        for step, difference in reversed(
                list(zip(self.steps, self.residual_differences))):
            inverse_curvature = 1.0 / np.dot(step, difference)
            coefficient = inverse_curvature * np.dot(step, vector)
            coefficients.append((coefficient, inverse_curvature))
            vector -= coefficient * difference

        last_step = self.steps[-1]
        last_difference = self.residual_differences[-1]
        scale = np.dot(last_step, last_difference) / np.dot(
            last_difference, last_difference)
        scale = np.clip(scale, self.initial_scale_min, self.initial_scale_max)
        vector *= scale

        pairs = list(zip(self.steps, self.residual_differences))
        for (step, difference), (coefficient, inverse_curvature) in zip(
                pairs, reversed(coefficients)):
            beta = inverse_curvature * np.dot(difference, vector)
            vector += step * (coefficient - beta)

        return current - vector

    def observe_first_trial(self, trial_residual):
        if self.current_residual is None or self.proposal_direction is None:
            return
        difference = np.asarray(trial_residual) - self.current_residual
        curvature = float(np.dot(self.proposal_direction, difference))
        threshold = self.curvature_tolerance * max(
            np.linalg.norm(self.proposal_direction) * np.linalg.norm(difference),
            np.finfo(float).eps,
        )
        if curvature > threshold:
            self.steps.append(self.proposal_direction.copy())
            self.residual_differences.append(difference.copy())


class AndersonAcceleration(FixedPointAccelerator):
    """Paper-style inverse multisecant Anderson acceleration."""

    name = "anderson"

    def __init__(self):
        super().__init__()
        self.memory = _env_int("BUNDLE_PALM_ANDERSON_MEMORY", "6")
        self.regularization = _env_float(
            "BUNDLE_PALM_ANDERSON_REGULARIZATION", "1e-4")
        self.damping = _env_float("BUNDLE_PALM_ANDERSON_DAMPING", "1.0")
        if self.regularization < 0.0:
            raise ValueError(
                "BUNDLE_PALM_ANDERSON_REGULARIZATION must be nonnegative")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError(
                "BUNDLE_PALM_ANDERSON_DAMPING must be in (0, 1]")
        self.steps = deque(maxlen=self.memory)
        self.residual_differences = deque(maxlen=self.memory)

    def reset(self):
        super().reset()
        self.steps.clear()
        self.residual_differences.clear()

    def _candidate(self, current, mapped, iteration):
        residual = current - mapped
        if not self.steps:
            return mapped

        step_matrix = np.column_stack(self.steps)
        difference_matrix = np.column_stack(self.residual_differences)
        gram = difference_matrix.T @ difference_matrix
        history_size = len(self.steps)
        scale = max(float(np.trace(gram)) / history_size, np.finfo(float).eps)
        gram += self.regularization * scale * np.eye(history_size)
        try:
            coefficients = np.linalg.solve(
                gram, difference_matrix.T @ residual)
        except np.linalg.LinAlgError:
            return mapped

        inverse_jacobian_residual = (
            residual + (step_matrix - difference_matrix) @ coefficients)
        return current - self.damping * inverse_jacobian_residual

    def observe_first_trial(self, trial_residual):
        if self.current_residual is None or self.proposal_direction is None:
            return
        difference = np.asarray(trial_residual) - self.current_residual
        if np.linalg.norm(difference) > np.finfo(float).eps:
            self.steps.append(self.proposal_direction.copy())
            self.residual_differences.append(difference.copy())


def create_accelerator(name=None):
    """Create an accelerator selected by name or BUNDLE_PALM_ACCELERATOR."""
    selected = (name or os.environ.get(
        "BUNDLE_PALM_ACCELERATOR", "nesterov")).strip().lower()
    factories = {
        "none": NoAcceleration,
        "drs": NoAcceleration,
        "nesterov": LegacyNesterov,
        "legacy_nesterov": LegacyNesterov,
        "themelis_nesterov": ThemelisNesterov,
        "fista": InertialNesterov,
        "adaptive_nesterov": lambda: InertialNesterov(adaptive=True),
        "lbfgs": LBFGSAcceleration,
        "bfgs": LBFGSAcceleration,
        "anderson": AndersonAcceleration,
    }
    if selected not in factories:
        choices = ", ".join(sorted(factories))
        raise ValueError(
            f"unknown BUNDLE_PALM_ACCELERATOR={selected!r}; choose {choices}")
    return factories[selected]()