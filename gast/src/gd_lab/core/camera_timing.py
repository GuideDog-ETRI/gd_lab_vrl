"""Global camera cadence, with an immediate snapshot after a row reset."""

import math


def camera_period_steps(policy_dt: float, contract_policy_dt: float, contract_period_steps: int) -> int:
    """Keep the contract's physical capture interval at a different control rate."""
    period = contract_policy_dt * contract_period_steps / policy_dt
    steps = round(period)
    if steps <= 0 or not math.isclose(period, steps, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("VRL camera interval must be an integer number of policy steps")
    return steps


def camera_refresh_mask(last_step, step: int, period_steps: int):
    """Never shift the regular sampling phase when one environment resets.

    Negative timestamps request a reset snapshot. The caller must render on
    reset, and otherwise render on the same global period used here.
    """
    if period_steps <= 0:
        raise ValueError("camera period_steps must be positive")
    return (last_step < 0) | ((step % period_steps == 0) & (last_step != step))
