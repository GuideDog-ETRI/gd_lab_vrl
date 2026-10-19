"""Content-addressed effective offline sensor profile; no live ROS assumptions."""
import hashlib
import json
from pathlib import Path

from .gdm_settings import GdmCoverage
from .memory import validate_settings


def effective_manifest(settings, *, screen_door=False, overrides=None):
    """Caller must supply effective overrides; this does not discover live ROS state.

    The returned profile hash belongs in each capture's sidecar alongside its
    original timestamp/epoch. Merely emitting a manifest never refreshes evidence.
    """
    effective = json.loads(json.dumps(settings))
    if overrides:
        effective.update(overrides)
    validate_settings(effective)
    coverage = GdmCoverage(effective)
    paths = {k: effective[k] for k in
             ('extrinsics_file', 'mid360_pattern_file', 'ouster_emitter_file')}
    paths.update({f'gdm.{k}': v for k, v in effective['gdm'].items()})
    files = {}
    for name, path in paths.items():
        p = Path(path).resolve(strict=True)
        files[name] = dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    body = dict(schema=1, settings=effective, files=files,
                override_scope='explicit offline overrides; not live ROS discovery',
                screen_door=bool(screen_door),
                effective_filters=dict(mid_rear_deg=coverage.mid_rear_deg,
                    mid_screen_door_x_positive=bool(screen_door),
                    ouster_exclude_cw_deg=coverage.ouster_exclude,
                    ouster_rings=coverage.rings,
                    ouster_near_m=coverage.ouster_min_range,
                    mid_near_m=coverage.mid_min_range,
                    mid_far_m=coverage.mid_max_range,mid_vertical_deg=coverage.mid_vertical,
                    self_hits=coverage.self_hits))
    canonical = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return dict(profile=body, sha256=hashlib.sha256(canonical.encode()).hexdigest())
