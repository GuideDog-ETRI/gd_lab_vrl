"""Boundary types shared across the framework. This module imports nothing from gd_lab."""

from __future__ import annotations

from dataclasses import dataclass

# Canonical foot order. Observation terms that index feet positionally and the
# left-right mirror permutations all assume this order.
FOOT_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")

# Mirror-rule tags understood by gd_lab.mdp.symmetry. Every observation term
# must declare one; an unknown tag fails at spec-resolution time, not silently.
MIRROR_TAGS = frozenset(
    {
        "ang_vel",  # pseudo-vector: (-1, 1, -1)
        "lin_vec",  # true vector: (1, -1, 1)
        "commands",  # (vx, vy, wz): (1, -1, -1)
        "joint",  # runtime joint L<->R permutation, HIP sign flip
        "scan",  # lateral flip of the height-scanner grid (runtime dims)
        "feet4",  # [FL, FR, RL, RR] scalar blocks: swap L<->R
        "feet16",  # 4 feet x (2x2 rays): block swap + in-block y flip
        "feet_vec3",  # 3-vector per foot: block swap + y sign flip
        "gain24",  # [kp x NJ, kd x NJ]: joint perm per block, no sign
        "identity",  # copy unchanged
    }
)


@dataclass(frozen=True)
class ObsTermSpec:
    """Structural declaration of one observation term.

    ``dim`` is the per-history-step width. ``dim=None`` marks a term whose width
    is only known at runtime (e.g. the height-scanner ray count); such terms must
    come after every term whose slice needs a static offset.
    """

    name: str
    dim: int | None
    mirror: str

    def __post_init__(self) -> None:
        if self.mirror not in MIRROR_TAGS:
            raise ValueError(f"Unknown mirror tag '{self.mirror}' for obs term '{self.name}'")


@dataclass(frozen=True)
class ObsSpec:
    """Layout of one observation group.

    The flattened group layout follows the IsaacLab observation manager:
    terms concatenated in declaration order; a term with group history H
    contributes a contiguous ``H * dim`` block ordered oldest -> newest.
    """

    terms: tuple[ObsTermSpec, ...]
    history: int = 1

    def __post_init__(self) -> None:
        names = [t.name for t in self.terms]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate obs term names: {names}")

    @property
    def per_step(self) -> int:
        dims = [t.dim for t in self.terms]
        if any(d is None for d in dims):
            raise ValueError("Spec has unresolved dims; call resolve() first")
        return sum(dims)  # type: ignore[arg-type]

    @property
    def total(self) -> int:
        return self.per_step * self.history

    def resolve(self, **dims: int) -> ObsSpec:
        """Return a copy with runtime-derived dims filled in (by term name)."""
        new_terms = []
        for t in self.terms:
            if t.name in dims:
                new_terms.append(ObsTermSpec(t.name, int(dims[t.name]), t.mirror))
            else:
                new_terms.append(t)
        unresolved = [t.name for t in new_terms if t.dim is None]
        if unresolved:
            raise ValueError(f"Unresolved obs term dims: {unresolved}")
        return ObsSpec(tuple(new_terms), self.history)

    def term(self, name: str) -> ObsTermSpec:
        for t in self.terms:
            if t.name == name:
                return t
        raise KeyError(f"Obs term '{name}' not in spec ({[t.name for t in self.terms]})")

    def offset(self, name: str) -> int:
        """Flat offset of the term's block start (includes the history factor)."""
        off = 0
        for t in self.terms:
            if t.name == name:
                return off
            if t.dim is None:
                raise ValueError(f"Cannot compute offset of '{name}': '{t.name}' is unresolved")
            off += t.dim * self.history
        raise KeyError(f"Obs term '{name}' not in spec")

    def slice(self, name: str) -> slice:
        """Flat slice of the term's full block (history included)."""
        t = self.term(name)
        if t.dim is None:
            raise ValueError(f"Obs term '{name}' has unresolved dim")
        start = self.offset(name)
        return slice(start, start + t.dim * self.history)

    def latest_slices(self) -> list[slice]:
        """Per-term slices of the newest history step, in term order.

        Concatenating them yields the current one-step observation (``per_step``
        wide) out of the flattened history layout.
        """
        out = []
        off = 0
        for t in self.terms:
            if t.dim is None:
                raise ValueError(f"Obs term '{t.name}' has unresolved dim")
            start = off + (self.history - 1) * t.dim
            out.append(slice(start, start + t.dim))
            off += t.dim * self.history
        return out

    def latest_indices(self) -> list[int]:
        """Flat indices of the newest history step (concatenation of latest_slices)."""
        idx: list[int] = []
        for sl in self.latest_slices():
            idx.extend(range(sl.start, sl.stop))
        return idx


@dataclass(frozen=True)
class ObsSpecSet:
    """The observation contract of one method: named groups, single source of truth."""

    policy: ObsSpec
    critic: ObsSpec

    @property
    def groups(self) -> dict[str, ObsSpec]:
        return {"policy": self.policy, "critic": self.critic}
