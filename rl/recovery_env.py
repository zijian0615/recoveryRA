"""Grasp-checkpoint recovery environment in the matched v0.5 screw cell (Isaac Sim).

Episodes start from post-attack simulator snapshots (runs/rl_snapshots), i.e. the
real state after ACT ran, the gripper-open attack fired and the detector alarmed.
The actor observes only deployment signals; simulator object poses feed the reward,
termination, the privileged critic input and evaluation.

Must be imported after SimulationApp has been created.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import omni.usd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fanuc_er4ia_sim"))
sys.path.insert(0, os.path.join(HERE, "..", "matched_sim"))
import cell  # noqa: E402
import fk  # noqa: E402
import ik  # noqa: E402
import recovery  # noqa: E402
import task  # noqa: E402
import obs_utils as O  # noqa: E402
import render_settings  # noqa: E402
import scene_config as C  # noqa: E402

C.SCREW_LENGTH, C.SCREW_HEAD_D, C.SCREW_HEAD_H = 0.030, 0.012, 0.0033
C.SCREW_SHANK_D, C.SCREW_MASS = 0.006, 0.0076
BIN_XY, BIN_OUTER, BIN_WALL_H = (0.511, 0.108), (0.060, 0.060), 0.060

SURF = C.SURFACE_DZ            # work surface height in the robot base frame
PICK_Z = -0.0021               # demo grasp tip height (base frame)
WS_LO = np.array([0.25, -0.30, -0.003])
WS_HI = np.array([0.60, 0.25, 0.25])
STEP_XYZ, STEP_YAW = 0.015, 0.15
HORIZON = 60
IMG = 96
GOAL_GRASP = 1.0  # one-hot over target checkpoints; only "grasp" exists so far
LAMBDA, COST = 0.1, 0.01


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def wrap_pi_half(a):
    """Angle difference modulo pi, in (-pi/2, pi/2]: a parallel jaw is pi-symmetric."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


class RecoveryEnv:
    def __init__(self, snapshots, seed=0, render="pathtrace1"):
        self.rng = np.random.default_rng(seed)
        self.snaps = snapshots
        if render == "realtime":
            render_settings.realtime()
        elif render == "pathtrace1":
            # 1 spp + OptiX denoiser: clean images at ~0.08 s/step. The realtime mode is
            # grainy here because DLSS (NGX) is unavailable with this driver.
            import carb
            st = carb.settings.get_settings()
            st.set("/rtx/rendermode", "PathTracing")
            st.set("/rtx/pathtracing/spp", 1)
            st.set("/rtx/pathtracing/totalSpp", 1)
            st.set("/rtx/pathtracing/maxBounces", 4)
            st.set("/rtx/pathtracing/optixDenoiser/enabled", True)
        import carb
        for k, v in json.loads(os.environ.get("RL_RTX_SETTINGS", "{}")).items():
            carb.settings.get_settings().set(k, v)
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        C.SCREWS = [tuple(snapshots[0]["screw_layout"])]
        self.world = World(stage_units_in_meters=1.0)
        stage = omni.usd.get_context().get_stage()
        cell.build(stage, with_objects=False)
        cell.add_screws(stage)
        cell._make_bin(stage, "/World/Bin", xy=BIN_XY, outer=BIN_OUTER, pocket_size=0.036,
                       pocket_r=0.008, wall_h=BIN_WALL_H, floor_t=0.015, mat_name="bin_yellow")
        # Low-resolution render products of the deployment cameras (same pose/FOV).
        self.wrist_cam = Camera(prim_path=cell.WRIST_CAM_PATH, resolution=O.WRIST_RENDER)
        self.table_cam = Camera(prim_path=cell.TABLE_CAM_PATH, resolution=O.TABLE_RENDER)
        cell.enable_ccd(self.world)
        self.world.reset()
        self.wrist_cam.initialize()
        self.table_cam.initialize()
        for _ in range(40):  # cameras return None until their render products have frames
            self.world.step(render=True)
        self.arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        self.arm.initialize()
        names = list(self.arm.dof_names)
        self.arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
        self.idx = {n: i for i, n in enumerate(names)}
        self.lim = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
        self.tool_off = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        self.screw = SingleRigidPrim("/World/Screw0")
        self.sub = max(1, int(round(0.1 / self.world.get_physics_dt())))
        self.table_model = recovery.TableCamera(C.TABLE_CAM, C.TABLE_TOP_Z + SURF)

    # ------------------------------------------------------------------ sim helpers
    def _q(self):
        jp = self.arm.get_joint_positions()
        return np.array([jp[i] for i in self.arm_idx], float)

    def _command(self, q6, closed):
        full = self.arm.get_joint_positions().copy()
        for k, i in enumerate(self.arm_idx):
            full[i] = q6[k]
        for jn, v in cell.jaw_targets(C.JAW_CLOSED if closed else C.JAW_OPEN).items():
            full[self.idx[jn]] = v
        self.arm.apply_action(ArticulationAction(joint_positions=full))

    def _advance(self):
        for k in range(self.sub):
            self.world.step(render=k >= self.sub - 2)

    def _tool(self, q=None):
        R, p = fk.tool_pose(self._q() if q is None else q, self.tool_off)
        sp = R @ np.array(C.FINGER_SPREAD_AXIS, float)
        return p, float(np.arctan2(-sp[0], sp[1]))  # spread = (-sin y, cos y)

    def screw_state(self):
        """Privileged: shank mid-point (base frame), axis angle, height above surface."""
        pos, quat = self.screw.get_world_pose()
        R = quat_to_R(np.asarray(quat, float))
        axis = R @ np.array([0.0, 0.0, -1.0])  # head -> tip
        mid = np.asarray(pos, float) + axis * C.SCREW_LENGTH / 2
        mid[2] -= C.TABLE_TOP_Z
        return mid, float(np.arctan2(axis[1], axis[0])), float(mid[2] - SURF)

    # ------------------------------------------------------------------ observations
    def _images(self):
        w = self.wrist_cam.get_rgba()[:, :, :3].astype(np.uint8)
        t = self.table_cam.get_rgba()[:, :, :3].astype(np.uint8)
        p, _ = self._tool()
        return O.images(w, t, O.fovea_box(self.table_model, p + np.array([0, 0, C.TABLE_TOP_Z])))

    def _proprio(self):
        p, yaw = self._tool()
        return O.proprio(self._q(), self.closed, p, yaw)

    def _priv(self):
        p, yaw = self._tool()
        mid, a, h = self.screw_state()
        jp = self.arm.get_joint_positions()
        gap = abs(float(jp[self.idx["jaw_l"]])) + abs(float(jp[self.idx["jaw_r"]]))
        return np.concatenate([(mid - p) / 0.05, [np.cos(2 * a), np.sin(2 * a)], [h / 0.05],
                               [(p[2] - SURF) / 0.1], [np.sin(2 * (yaw - a)), np.cos(2 * (yaw - a))],
                               [gap / 0.012]]).astype(np.float32)

    def obs(self):
        w, t = self._images()
        return {"wrist": w, "table": t, "proprio": self._proprio(), "priv": self._priv()}

    # ------------------------------------------------------------------ task predicates
    def _phi(self):
        return float(task.phi(self._priv()))

    def _grasp_ok(self):
        """Stable-grasp checkpoint predicate (task.py): lifted, centred pinch."""
        return bool(task.grasp_ok(self._priv(), self.closed))

    def _unsafe(self):
        p, _ = self._tool()
        mid, _, h = self.screw_state()
        r = np.hypot(mid[0], mid[1])
        off_table = h < -0.03 or not (0.18 < r < 0.62)
        near_bin = (abs(p[0] - BIN_XY[0]) < BIN_OUTER[0] / 2 + 0.01
                    and abs(p[1] - BIN_XY[1]) < BIN_OUTER[1] / 2 + 0.01
                    and p[2] - SURF < BIN_WALL_H + 0.01)
        return bool(off_table or near_bin)

    # ------------------------------------------------------------------ episode API
    def reset(self, snap=None, augment=True):
        s = snap if snap is not None else self.snaps[self.rng.integers(len(self.snaps))]
        self.snap = s
        full = np.array(s["joints_rest"], float)
        for jn, v in cell.jaw_targets(C.JAW_OPEN).items():
            full[self.idx[jn]] = v
        self.arm.set_joint_positions(full)
        self.arm.set_joint_velocities(np.zeros_like(full))
        pos = np.array(s["screw_rest_pos"], float)
        quat = np.array(s["screw_rest_quat"], float)
        if augment and self.rng.random() < 0.5:
            # Perturb the dropped screw: small shift and a spin about the vertical.
            pos[:2] += self.rng.uniform(-0.02, 0.02, 2)
            th = self.rng.uniform(-np.pi, np.pi)
            qz = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
            w1, x1, y1, z1 = qz
            w2, x2, y2, z2 = quat
            quat = np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                             w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])
            pos[2] += 0.002
        self.screw.set_world_pose(pos, quat)
        self.screw.set_linear_velocity(np.zeros(3))
        self.screw.set_angular_velocity(np.zeros(3))
        self.closed, self.closed_t = False, 0
        q = np.array([full[i] for i in self.arm_idx])
        for k in range(30):
            self._command(q, False)
            self.world.step(render=k >= 26)
        self.target, self.yaw = self._tool()
        self.target = np.clip(self.target, WS_LO, WS_HI)
        self.t, self.hold = 0, 0
        self.prev_phi = self._phi()
        return self.obs()

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        self.target = np.clip(self.target + a[:3] * STEP_XYZ, WS_LO, WS_HI)
        self.yaw += a[3] * STEP_YAW
        self.closed = bool(a[4] > 0)
        self.closed_t = self.closed_t + 1 if self.closed else 0
        sp = (-np.sin(self.yaw), np.cos(self.yaw), 0.0)
        q, _ = ik.solve(self._q(), self.target, (0.0, 0.0, -1.0), self.lim,
                        tool_offset=self.tool_off, spread_dir=sp)
        self._command(q, self.closed)
        self._advance()
        self.t += 1
        phi = self._phi()
        self.hold = self.hold + 1 if self._grasp_ok() else 0
        info = {"success": False, "unsafe": False, "timeout": False}
        if self.hold >= task.HOLD_STEPS:
            r, term = 1.0, True
            info["success"] = True
        elif self._unsafe():
            r, term = -1.0, True
            info["unsafe"] = True
        else:
            r, term = task.LAMBDA * (phi - self.prev_phi) - task.COST, False
        self.prev_phi = phi
        trunc = (not term) and self.t >= HORIZON
        info["timeout"] = trunc
        return self.obs(), r, term, trunc, info

    # ------------------------------------------------------------------ teacher
    def expert_action(self, noise=0.0):
        """Privileged scripted teacher, used only to seed the replay buffer."""
        p, yaw = self._tool()
        mid, a, h = self.screw_state()
        dyaw = wrap_pi_half(a - yaw)
        aligned = abs(dyaw) < 0.08
        dxy = mid[:2] - p[:2]
        if self.closed and self.closed_t < 3:
            goal, g = self.target.copy(), 1.0
        elif self.closed:
            goal = np.array([p[0], p[1], 0.08])
            g = 1.0
        elif np.linalg.norm(dxy) > 0.0015 or not aligned:
            goal = np.array([mid[0], mid[1], max(0.05, p[2] - 0.01)])
            g = -1.0
        elif p[2] > PICK_Z + 0.003:
            goal = np.array([mid[0], mid[1], PICK_Z])
            g = -1.0
        else:
            goal, g = np.array([mid[0], mid[1], PICK_Z]), 1.0
        act = np.concatenate([np.clip((goal - self.target) / STEP_XYZ, -1, 1),
                              [np.clip(dyaw / STEP_YAW, -1, 1)], [g]])
        act[:4] += noise * self.rng.standard_normal(4)
        return np.clip(act, -1, 1)


def load_snapshots(root):
    out = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "report_snapshot.json")
        if os.path.exists(p):
            out.append(json.load(open(p)))
    return out
