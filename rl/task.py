"""Grasp-checkpoint predicate and potential, computed from the privileged vector so the
environment and replay relabelling share one definition.

priv = [(mid - tool)/0.05 (3), cos 2a, sin 2a, h/0.05, (tool_z - surface)/0.1,
        sin 2(yaw - a), cos 2(yaw - a), jaw_gap/0.012]
"""
import numpy as np

HOLD_STEPS = 3
LIFT_H = 0.04
MAX_ALONG, MAX_PERP = 0.0015, 0.004  # ACT only resumes from a grasp within ~1.5 mm of the demo grasp
LAMBDA, COST = 0.1, 0.01


def _geom(priv):
    rel = np.asarray(priv[..., 0:3]) * 0.05
    ang = 0.5 * np.arctan2(priv[..., 4], priv[..., 3])
    ax = np.stack([np.cos(ang), np.sin(ang)], -1)
    along = np.abs((rel[..., :2] * ax).sum(-1))
    perp = np.abs(rel[..., 0] * -ax[..., 1] + rel[..., 1] * ax[..., 0])
    return rel, along, perp, priv[..., 5] * 0.05


def phi(priv):
    rel, along, _, h = _geom(priv)
    d = np.linalg.norm(rel, axis=-1)
    align = np.sqrt(np.clip((1 - priv[..., 8]) / 2, 0, 1))  # |sin(yaw - a)|
    lift = np.where(d < 0.03, np.clip(h - 0.01, 0, 0.05) / 0.05, 0.0)
    centre = np.where(d < 0.03, np.minimum(along, 0.01) / 0.005, 0.0)  # mm-level centring
    return -d / 0.05 - 0.5 * align + 2.0 * lift - centre


def grasp_ok(priv, closed):
    _, along, perp, h = _geom(priv)
    return np.asarray(closed, bool) & (h > LIFT_H) & (along < MAX_ALONG) & (perp < MAX_PERP)


def relabel(priv, proprio, unsafe_terminal):
    """Rewards/terminals for an episode of T transitions (priv/proprio have T+1 rows).
    Unsafe terminations are kept; success is re-derived with the current predicate."""
    T = len(priv) - 1
    p = phi(priv)
    ok = grasp_ok(priv, proprio[:, 6] > 0.5)
    r = (LAMBDA * (p[1:] - p[:-1]) - COST).astype(np.float32)
    term = np.zeros(T, bool)
    hold = 0
    for t in range(T):
        hold = hold + 1 if ok[t + 1] else 0
        if hold >= HOLD_STEPS:
            r[t], term[t] = 1.0, True
            return r[: t + 1], term[: t + 1]
        if unsafe_terminal[t]:
            r[t], term[t] = -1.0, True
            return r[: t + 1], term[: t + 1]
    return r, term
