"""Global camera cadence, with an immediate snapshot after a row reset."""


def camera_refresh_mask(last_step, step: int, period_steps: int):
    """Never shift the regular sampling phase when one environment resets.

    Negative timestamps request a reset snapshot. The caller must render on
    reset, and otherwise render on the same global period used here.
    """
    if period_steps <= 0:
        raise ValueError("camera period_steps must be positive")
    return (last_step < 0) | ((step % period_steps == 0) & (last_step != step))
