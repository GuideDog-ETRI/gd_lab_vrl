"""Left-right mirror augmentation bound to the DreamWaQ observation spec."""

from __future__ import annotations

from gd_lab.mdp.symmetry import make_augmentation

from .spec import DREAMWAQ_SPEC

compute_symmetric_states = make_augmentation(DREAMWAQ_SPEC)
