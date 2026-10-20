"""Detect -> restore -> resume components that use deployment-visible inputs only.

- Detector: nearest-neighbour novelty of (state, executed command) against the
  v4 demonstrations, threshold = 99th percentile of leave-one-episode-out scores.
- Restore: locate the dropped screw in the calibrated table camera by differencing
  against the episode's first frame, back-project to the table plane, re-grasp with
  IK, confirm with the camera that the screw left the table, then move to the demos'
  post-lift checkpoint pose.

Nothing here reads simulator object poses.
"""
from pathlib import Path

import json

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "repos/dataset/data/chunk-000/file-000.parquet"
REF = HERE / "demo_state_action.npz"  # written by export_reference(); Isaac's venv lacks pyarrow


def export_reference():
    import pyarrow.parquet as pq
    rows = pq.read_table(DATA, columns=["episode_index", "observation.state", "action"]).to_pylist()
    np.savez(REF, episode=np.array([r["episode_index"] for r in rows]),
             x=np.array([np.concatenate([r["observation.state"], r["action"]]) for r in rows]))


def trace_features(trace_path):
    """(state, executed command) rows from an eval_v05 trace."""
    out = []
    for r in json.load(open(trace_path)):
        a = list(r["action"][:6]) + [1.0 if r["action"][6] >= 0.5 else 0.0]
        out.append(np.concatenate([r["state"], a]))
    return np.asarray(out, float)


ARM = [0, 1, 2, 3, 4, 5]  # measured joints; commanded joints jump at ACT chunk boundaries


class NoveltyDetector:
    """Gripper-phase-conditioned novelty: each frame is compared only with demo frames
    whose (state gripper, commanded gripper) pair matches, on measured arm joints.

    The pair acts as a coarse phase label: an opening command while holding
    (1 -> 0) appears in the demos only above the bin, so it scores high elsewhere.
    """

    def __init__(self, quantile=0.99, calibration_traces=()):
        d = np.load(REF)
        x = d["x"]
        self.scale = np.maximum(x.std(axis=0), 1.0)[ARM]
        self.ref = {}
        for sg in (0, 1):
            for cg in (0, 1):
                m = ((x[:, 6] > 0.5) == sg) & ((x[:, 13] > 0.5) == cg)
                self.ref[(sg, cg)] = self._arm(x[m])
        eps = d["episode"]
        if calibration_traces:
            clean = np.concatenate([self.score_rows(trace_features(p)) for p in calibration_traces])
        else:
            clean = []
            for k in np.unique(eps):
                held_out = NoveltyDetector.__new__(NoveltyDetector)
                held_out.scale = self.scale
                held_out.ref = {pair: self._arm(x[(eps != k) & ((x[:, 6] > 0.5) == pair[0])
                                                  & ((x[:, 13] > 0.5) == pair[1])])
                                for pair in self.ref}
                clean.append(held_out.score_rows(x[eps == k]))
            clean = np.concatenate(clean)
        clean = clean[np.isfinite(clean)]
        self.threshold = float(np.quantile(clean, quantile))
        self.calibration_frames = int(len(clean))

    def _arm(self, rows):
        return rows[:, ARM] / self.scale

    def score_rows(self, rows):
        out = np.full(len(rows), np.inf)
        for pair, ref in self.ref.items():
            m = ((rows[:, 6] > 0.5) == pair[0]) & ((rows[:, 13] > 0.5) == pair[1])
            if m.any() and len(ref):
                q = self._arm(rows[m])
                d2 = (q * q).sum(1)[:, None] + (ref * ref).sum(1)[None, :] - 2 * q @ ref.T
                out[m] = np.sqrt(np.maximum(d2.min(1), 0.0))
        return out

    def score(self, state, command):
        return float(self.score_rows(np.concatenate([state, command])[None, :])[0])


class TableCamera:
    """Pinhole model from the calibration config (eye/look_at/up/focal/aperture)."""

    def __init__(self, cfg, table_z):
        eye, tgt, up = (np.asarray(cfg[k], float) for k in ("eye", "look_at", "up"))
        z = eye - tgt
        z /= np.linalg.norm(z)
        x = np.cross(up, z)
        x /= np.linalg.norm(x)
        self.R = np.stack([x, np.cross(z, x), z])  # rows: camera axes in world
        self.eye = eye
        self.W, self.H = cfg["resolution"]
        self.tan_h = cfg["horiz_aperture"] / 2 / cfg["focal_length"]
        self.tan_v = self.tan_h * self.H / self.W
        self.table_z = table_z

    def to_plane(self, u, v, z):
        d = np.array([(u / self.W * 2 - 1) * self.tan_h, (1 - v / self.H * 2) * self.tan_v, -1.0]) @ self.R
        t = (z - self.eye[2]) / d[2]
        return self.eye + t * d

    def project(self, p):
        c = self.R @ (np.asarray(p, float) - self.eye)
        return np.array([(c[0] / -c[2] / self.tan_h + 1) * 0.5 * self.W,
                         (1 - c[1] / -c[2] / self.tan_v) * 0.5 * self.H])


def find_screw(cam, ref_rgb, rgb, roi_xy=((0.25, 0.62), (-0.30, 0.25)), min_px=40):
    """Dark pixels that are new relative to the reference frame, on the work area.

    Returns (grasp_xy, axis_yaw_deg, n_pixels) in the robot base frame, or None.
    The reference is the episode's first table frame with the arm at its start pose,
    so the call must be made with the arm back at that pose. A 5x5 blur suppresses
    renderer grain; the largest connected blob is taken as the screw.
    """
    from scipy import ndimage

    ref = ndimage.uniform_filter(ref_rgb.astype(np.float32).mean(2), 5)
    cur = ndimage.uniform_filter(rgb.astype(np.float32).mean(2), 5)
    mask = (cur < 90) & (ref - cur > 40)
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    z = cam.table_z + 0.004  # screw top surface, roughly half its head diameter up
    (x0, x1), (y0, y1) = roi_xy
    for k in np.argsort(-sizes):
        if sizes[k] < min_px:
            return None
        vs, us = np.nonzero(labels == k + 1)
        pts = np.array([cam.to_plane(u + 0.5, v + 0.5, z)[:2] for u, v in zip(us, vs)])
        c = pts.mean(0)
        if not (x0 < c[0] < x1 and y0 < c[1] < y1):
            continue
        w, V = np.linalg.eigh(np.cov((pts - c).T))
        axis = V[:, -1]
        proj = (pts - c) @ axis
        mid = c + axis * (proj.min() + proj.max()) / 2  # mid-length, not the head-heavy centroid
        return mid, float(np.degrees(np.arctan2(axis[1], axis[0]))), int(len(pts))
    return None
