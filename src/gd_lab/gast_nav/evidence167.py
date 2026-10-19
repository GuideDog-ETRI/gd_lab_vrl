"""Fail-closed integration seam pending 90.111 measured free evidence model."""
import torch


def certified_cells(state, support_known, measured_free=None):
    # Existing occupied evidence remains observed; no missing free measurement
    # is converted to FREE. The independent occupied clearance veto is retained.
    if measured_free is None:measured_free=torch.zeros_like(state,dtype=torch.bool)
    if measured_free.shape!=state.shape:raise ValueError('free evidence shape')
    return support_known & ((state[:,:2]==2)|measured_free[:,:2]).all(1)


def entered_unknown_cells(current_footprint, previous_footprint, previous_known, valid):
    """Measured newly occupied raster cells in the previous capture frame.

    This is a reward diagnostic, NOT a continuous-time collision certificate.
    The supervisor still requires the full candidate-through-stop envelope.
    """
    return ((current_footprint & ~previous_footprint & ~previous_known).flatten(1).any(1) & valid)
