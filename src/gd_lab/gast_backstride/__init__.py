"""Backward gap stride: an A' (gait v3 G5) continuation that fixes the backward gap crossing only.

MuJoCo a1993 / a1973 (11-03): after the leading hind feet cross a backward slot they stay planted and the front feet
shuffle up to them -- same-side front-to-hind stance spacing 0.6 m on flat ground, -0.05..0.31 m after the slots
(6566 0.58 m), and a front foot stepping onto the hind foot is one of the falls. A' turned R9 (stance overextension)
off on backward gaps (R19) and R17 (3 feet in contact near a slot) pays for keeping the hind feet down. The wanted
crossing mirrors the forward one: the hind feet cross and carry the body, the front feet lift clear and step over
with a full stride. Terms (gap tiles while moving backward only; forward gaps / stairs / flat are A' and are held by
the retention anchor toward the start model): stance spacing floor, front foot lift over the slot, R9 at 30 cm.
"""
