"""Stage C' (GAST-Narrow v2.6): the cliff stop on top of the stage B' model.

Design: docs/design/GAST_NARROW_V26_CLIFF_DESIGN.md. B' (gast_bprime) + the v2.5 cliff decks (4 columns, front / back
commands only, 0.3..0.8 m/s) + one actor input, ``support_end`` (gast_cprime.support), computed from the terrain input
along the command direction. The v2.x / B' task code is unchanged; this package composes it.
"""
