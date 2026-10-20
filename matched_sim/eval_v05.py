"""Roll the screw ACT policy in a v0.5 cell that matches the v4 recordings.

The v4 demonstrations were recorded in the v0.5 cell (clean white table, enclosure,
v0.5 cameras and 293 mm tool). Their kinematics give one screw grasped near
(0.43, -0.13) and a release above (0.511, 0.108); see LAYOUT.md.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim-venv/bin/python eval_v05.py --seed 0 --port 8765
"""
import argparse
import json
import os
import pickle
import socket
import sys

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--port", type=int, default=8765)
parser.add_argument("--steps", type=int, default=450)
parser.add_argument("--jitter", type=float, default=0.006,
                    help="uniform screw XY jitter (m); demos span about +-6 mm")
parser.add_argument("--yaw-range", nargs=2, type=float, default=(33.0, 53.0),
                    help="screw yaw range (deg); demos span 33..53")
parser.add_argument("--flip-screw", action="store_true", help="add 180 deg to the screw yaw")
parser.add_argument("--replan-interval", type=int, default=10)
parser.add_argument("--attack", choices=("none", "open_when_held"), default="none")
parser.add_argument("--attack-delay", type=int, default=0,
                    help="steps after the hold trigger before the attack starts")
parser.add_argument("--attack-duration", type=int, default=5)
parser.add_argument("--response", choices=("none", "stop", "continue", "inverse_replay", "ours",
                                           "snapshot", "ours_rl"),
                    default="none", help="what happens once the detector alarms")
parser.add_argument("--calibration-traces", nargs="*", default=(),
                    help="clean-run traces that set the detector threshold")
parser.add_argument("--report", required=True)
parser.add_argument("--trace", action="store_true")
parser.add_argument("--render", choices=("default", "realtime", "pathtrace", "pathtrace1"),
                    default="default")
parser.add_argument("--rl-actor", default=None, help="learned recovery actor for --response ours_rl")
parser.add_argument("--rl-steps", type=int, default=60)
parser.add_argument("--rl-obs", choices=("v1", "v2"), default="v1",
                    help="v1: fixed work-area crop (actor_040000 / actor_ft030000); v2: tool fovea")
args = parser.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation, SingleXFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
import omni.usd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "fanuc_er4ia_sim"))
import cell
import ik
import recovery
import render_settings
import scene_config as C

# Screw and bin values from the screw cell (screw_sim/scene_config.py).
C.SCREW_LENGTH, C.SCREW_HEAD_D, C.SCREW_HEAD_H = 0.030, 0.012, 0.0033
C.SCREW_SHANK_D, C.SCREW_MASS = 0.006, 0.0076
BIN_XY = (0.511, 0.108)
BIN_OUTER, BIN_POCKET, BIN_POCKET_R = (0.060, 0.060), 0.036, 0.008
BIN_WALL_H, BIN_FLOOR_T = 0.060, 0.015

rng = np.random.default_rng(args.seed)
yaw = float(rng.uniform(*args.yaw_range)) + (180.0 if args.flip_screw else 0.0)
grasp_xy = np.array([0.4291, -0.1301]) + rng.uniform(-args.jitter, args.jitter, 2)
axis = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])
head_xy = grasp_xy - axis * C.SCREW_LENGTH / 2  # grasp at mid-length
C.SCREWS = [(float(head_xy[0]), float(head_xy[1]), yaw)]

if args.render == "realtime":
    render_settings.realtime()
elif args.render == "pathtrace":
    render_settings.enable_path_tracing(spp=16)
elif args.render == "pathtrace1":  # what the RL recovery actor was trained with
    import carb
    st = carb.settings.get_settings()
    st.set("/rtx/rendermode", "PathTracing")
    st.set("/rtx/pathtracing/spp", 1)
    st.set("/rtx/pathtracing/totalSpp", 1)
    st.set("/rtx/pathtracing/maxBounces", 4)
    st.set("/rtx/pathtracing/optixDenoiser/enabled", True)

ik.set_tool_axis(C.TOOL_AXIS)
ik.set_spread_axis(C.FINGER_SPREAD_AXIS)

world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
cell.build(stage, with_objects=False)
cell.add_screws(stage)
cell._make_bin(stage, "/World/Bin", xy=BIN_XY, outer=BIN_OUTER, pocket_size=BIN_POCKET,
               pocket_r=BIN_POCKET_R, wall_h=BIN_WALL_H, floor_t=BIN_FLOOR_T,
               mat_name="bin_yellow")
cams = []
for name, path, cfg in (("table", cell.TABLE_CAM_PATH, C.TABLE_CAM),
                        ("wrist", cell.WRIST_CAM_PATH, C.WRIST_CAM),
                        ("side", cell.SIDE_CAM_PATH, C.SIDE_CAM)):
    cams.append((name, Camera(prim_path=path, resolution=cfg["resolution"])))
cell.enable_ccd(world)
world.reset()
for _, cam in cams:
    cam.initialize()
for _ in range(40):
    world.step(render=True)

arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
arm.initialize()
names = list(arm.dof_names)
ARM_IDX = [names.index(f"J{i}") for i in range(1, 7)]
IDX = {n: names.index(n) for n in names}
START_DEG = np.array([0.0, 3.3, -11.9, 0.0, -74.8, 0.0])  # every v4 episode starts here


LIM = np.deg2rad(np.array([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)]))
TOOL_OFF = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
DOWN = (0.0, 0.0, -1.0)
# Demo grasp-tip height and the episode-0 pose 30 steps after the grasp closed, with the
# screw lifted: the grasp checkpoint we hand back to ACT from.
PICK_Z, HOVER_Z = -0.0021, 0.125
CHECKPOINT_DEG = np.array([-16.9, 43.1, 17.1, 0.0, -64.1, -125.2])


def command(q6, jaw):
    full = arm.get_joint_positions().copy()
    for k, i in enumerate(ARM_IDX):
        full[i] = q6[k]
    for jn, v in cell.jaw_targets(jaw).items():
        full[IDX[jn]] = v
    arm.apply_action(ArticulationAction(joint_positions=full))


for k in range(60):
    command(np.deg2rad(START_DEG), C.JAW_OPEN)
    world.step(render=k >= 50)  # cameras must show the start pose before the first observation

sock = socket.create_connection(("127.0.0.1", args.port))


def rpc(msg):
    payload = pickle.dumps(msg)
    sock.sendall(len(payload).to_bytes(8, "big") + payload)
    n = int.from_bytes(sock.recv(8), "big")
    buf = b""
    while len(buf) < n:
        buf += sock.recv(n - len(buf))
    return pickle.loads(buf)


SURFACE_Z = C.TABLE_TOP_Z + C.SURFACE_DZ


def screw_pos():
    return np.asarray(SingleXFormPrim("/World/Screw0").get_world_pose()[0], float)


def held():
    """Evaluation-only checkpoint: screw lifted clear of the table."""
    return bool(screw_pos()[2] > SURFACE_Z + 0.02)


def observe():
    images = {}
    for name, cam in cams:
        rgb = cam.get_rgba()[:, :, :3].astype(np.uint8)
        images[name] = np.asarray(Image.fromarray(rgb).resize((640, 480), Image.BILINEAR))
    jp = arm.get_joint_positions()
    return images, np.asarray([float(np.degrees(jp[i])) for i in ARM_IDX], np.float32)


sub = max(1, int(round(0.1 / world.get_physics_dt())))
table_cam = dict(cams)["table"]
cam_model = recovery.TableCamera(C.TABLE_CAM, SURFACE_Z)
detector = recovery.NoveltyDetector(calibration_traces=args.calibration_traces)
print(f"[eval] tip_offset {cell.tip_offset():.4f} detector threshold {detector.threshold:.3f}", flush=True)
step = 0
trace, frame_dir = [], os.path.splitext(args.report)[0] + "_frames"
restore_log = []


def table_rgb():
    return table_cam.get_rgba()[:, :, :3].astype(np.uint8)


def tick(q6, jaw_closed, tag):
    """One 10 Hz control step outside the policy (restore or replay motion)."""
    global step
    command(q6, C.JAW_CLOSED if jaw_closed else C.JAW_OPEN)
    for k in range(sub):
        world.step(render=k >= sub - 2)
    if args.trace:
        trace.append({"step": step, "mode": tag, "q": np.degrees(q6).tolist(),
                      "gripper": float(jaw_closed), "screw": screw_pos().tolist()})
    step += 1


def current_q():
    jp = arm.get_joint_positions()
    return np.array([jp[i] for i in ARM_IDX], float)


def move_joints(q_goal, jaw_closed, n, tag):
    q0 = current_q()
    for i in range(1, n + 1):
        t = i / n
        tick(q0 + (q_goal - q0) * t * t * (3 - 2 * t), jaw_closed, tag)  # eased: a held screw slips under jerk


def move_line(p_goal, spread, jaw_closed, n, tag):
    q = current_q()
    import fk as _fk
    p0 = _fk.tool_pose(q, TOOL_OFF)[1]
    for i in range(1, n + 1):
        t = i / n
        t = t * t * (3 - 2 * t)
        q, _ = ik.solve(q, p0 + (np.asarray(p_goal) - p0) * t, DOWN, LIM,
                        tool_offset=TOOL_OFF, spread_dir=spread)
        tick(q, jaw_closed, tag)


def restore(ref_rgb, max_tries=2):
    """Re-establish the grasp checkpoint from camera evidence. Returns an outcome tag."""
    import fk as _fk
    start_q = np.deg2rad(START_DEG)
    p = _fk.tool_pose(current_q(), TOOL_OFF)[1]
    move_line([p[0], p[1], max(p[2], HOVER_Z)], None, False, 6, "restore")
    move_joints(start_q, False, 12, "restore")
    for _ in range(5):
        tick(start_q, False, "restore")
    for attempt in range(max_tries):
        cur = table_rgb()
        if args.trace:
            os.makedirs(frame_dir, exist_ok=True)
            Image.fromarray(ref_rgb).save(os.path.join(frame_dir, "restore_ref.png"))
            Image.fromarray(cur).save(os.path.join(frame_dir, f"restore_cur{attempt}.png"))
        found = recovery.find_screw(cam_model, ref_rgb, cur)
        restore_log.append({"attempt": attempt, "found": None if found is None else
                            {"xy": found[0].tolist(), "yaw": found[1], "px": found[2]}})
        if found is None:
            return "no_displaced_screw" if attempt == 0 else "restored"
        xy, yaw, _ = found
        a = np.radians(yaw)
        spread = np.array([-np.sin(a), np.cos(a), 0.0])
        if spread @ np.array([0.61, -0.79, 0.0]) < 0:  # demo wrist orientation
            spread = -spread
        move_line([xy[0], xy[1], HOVER_Z], spread, False, 15, "restore")
        move_line([xy[0], xy[1], PICK_Z], spread, False, 8, "restore")
        for _ in range(5):
            tick(current_q(), True, "restore")
        move_line([xy[0], xy[1], HOVER_Z], spread, True, 12, "restore")
        move_joints(start_q, True, 30, "restore")
        for _ in range(3):
            tick(start_q, True, "restore")
    return "restored" if recovery.find_screw(cam_model, ref_rgb, table_rgb()) is None else "failed"


def rl_restore():
    """Learned goal-conditioned recovery actor (rl/), deployment observations only.
    Hands back once the gripper has been closed and lifted for 3 steps."""
    import torch
    sys.path.insert(0, os.path.join(HERE, "..", "rl"))
    import fk as _fk
    import networks as RN
    import obs_utils as O
    actor = RN.Actor()
    actor.load_state_dict(torch.load(args.rl_actor, map_location="cpu"))
    actor.eval()
    cams_d = dict(cams)
    lo, hi = np.array([0.25, -0.30, -0.003]), np.array([0.60, 0.25, 0.25])
    q = current_q()
    for _ in range(3):  # let the dropped screw land (training starts from rested drops)
        tick(q, False, "rl")

    def tool():
        R, p = _fk.tool_pose(current_q(), TOOL_OFF)
        sp = R @ np.array(C.FINGER_SPREAD_AXIS, float)
        return p, float(np.arctan2(-sp[0], sp[1]))

    target, yaw = tool()
    target = np.clip(target, lo, hi)
    closed, lifted = False, 0
    for t in range(args.rl_steps):
        w = cams_d["wrist"].get_rgba()[:, :, :3].astype(np.uint8)
        tb = cams_d["table"].get_rgba()[:, :, :3].astype(np.uint8)
        p, y = tool()
        if args.rl_obs == "v1":
            import obs_utils_v1 as O1
            wi, ti = O1.images(w, tb, O1.table_crop(cam_model, C.TABLE_TOP_Z + C.SURFACE_DZ))
        else:
            wi, ti = O.images(w, tb, O.fovea_box(cam_model, p + np.array([0, 0, C.TABLE_TOP_Z])))
        pr = O.proprio(current_q(), closed, p, y)
        a = actor.act(torch.from_numpy(wi)[None], torch.from_numpy(ti)[None],
                      torch.from_numpy(pr)[None])[0].numpy()
        target = np.clip(target + a[:3] * 0.015, lo, hi)
        yaw += a[3] * 0.15
        closed = bool(a[4] > 0)
        sp = (-np.sin(yaw), np.cos(yaw), 0.0)
        qn, _ = ik.solve(current_q(), target, DOWN, LIM, tool_offset=TOOL_OFF, spread_dir=sp)
        tick(qn, closed, "rl")
        p, _ = tool()
        lifted = lifted + 1 if closed and p[2] - C.SURFACE_DZ > 0.045 else 0
        if lifted >= 3:
            restore_log.append({"rl_steps": t + 1})
            return "restored"
    restore_log.append({"rl_steps": args.rl_steps})
    return "failed"


ref_rgb = table_rgb()
rpc({"op": "reset"})
gripper = 0.0
first_hold = attack_start = None
checkpoint_before = checkpoint_after = None
stopped_at = alarm_step = response_step = None
alarm_steps, restore_outcome, checkpoint_restored = [], None, None
commanded = []
policy_steps = 0
while step < args.steps:
    if args.replan_interval and policy_steps and policy_steps % args.replan_interval == 0:
        rpc({"op": "reset"})
    policy_steps += 1
    images, joints = observe()
    state = np.append(joints, gripper).astype(np.float32)
    action = np.asarray(rpc({"op": "act", "images": images, "state": state,
                             "task": "pick one countersunk screw and drop it in the yellow bin"})
                        ["action"], np.float32).reshape(-1)
    q, gripper = np.deg2rad(action[:6]), (1.0 if action[6] >= 0.5 else 0.0)
    if args.trace:
        trace.append({"step": step, "mode": "policy", "state": state.tolist(),
                      "action": action.tolist(), "screw": screw_pos().tolist()})
        if step % 25 == 0:
            os.makedirs(frame_dir, exist_ok=True)
            for name, img in images.items():
                Image.fromarray(img).save(os.path.join(frame_dir, f"{step:04d}_{name}.png"))
    if first_hold is None and gripper and held():
        first_hold = step
        if args.attack != "none":
            attack_start = step + args.attack_delay
    if attack_start is not None and step == attack_start:
        checkpoint_before = held()
    attacking = attack_start is not None and attack_start <= step < attack_start + args.attack_duration
    if attacking:
        gripper = 0.0
    # The detector sees what the controller actually received.
    score = detector.score(state, np.append(action[:6], gripper))
    if score > detector.threshold:
        alarm_steps.append(step)
        if alarm_step is None:
            alarm_step = step
    command(q, C.JAW_CLOSED if gripper else C.JAW_OPEN)
    commanded.append((q.copy(), gripper))
    for k in range(sub):
        world.step(render=k >= sub - 2)  # only the frames the next observation reads
    attack_end = None if attack_start is None else attack_start + args.attack_duration - 1
    if attack_end is not None and step == attack_end:
        checkpoint_after = held()
    step += 1
    # Respond once the alarm has fired and the attacker no longer holds the channel.
    if (response_step is None and alarm_step is not None and args.response != "none"
            and (attack_end is None or step > attack_end)):
        response_step = step
        rpc({"op": "reset"})
        if args.response == "stop":
            stopped_at = step
            break
        if args.response == "snapshot":
            # Post-attack simulator state, used only to reset RL recovery episodes.
            from isaacsim.core.prims import SingleRigidPrim
            screw = SingleRigidPrim("/World/Screw0")
            pos, quat = screw.get_world_pose()
            snap = {"seed": args.seed, "step": step, "screw_layout": C.SCREWS[0],
                    "dof_names": names, "joints": arm.get_joint_positions().tolist(),
                    "screw_pos": np.asarray(pos).tolist(), "screw_quat": np.asarray(quat).tolist(),
                    "attack_delay": args.attack_delay, "attack_duration": args.attack_duration}
            for _ in range(20):  # let the screw land, then record where it rests
                world.step(render=False)
            pos, quat = screw.get_world_pose()
            snap["screw_rest_pos"], snap["screw_rest_quat"] = np.asarray(pos).tolist(), np.asarray(quat).tolist()
            snap["joints_rest"] = arm.get_joint_positions().tolist()
            with open(os.path.splitext(args.report)[0] + "_snapshot.json", "w") as f:
                json.dump(snap, f)
            stopped_at = step
            break
        if args.response == "inverse_replay":
            back_to = attack_start if attack_start is not None else alarm_step
            for qq, gg in reversed(commanded[back_to:]):
                tick(qq, gg, "replay")
        if args.response in ("ours", "ours_rl"):
            restore_outcome = restore(ref_rgb) if args.response == "ours" else rl_restore()
            if restore_outcome == "failed":
                stopped_at = step
                break
            if restore_outcome == "restored":
                move_joints(np.deg2rad(CHECKPOINT_DEG), True, 40, "restore")
                checkpoint_restored = held()  # evaluation only
                gripper = 1.0
        rpc({"op": "reset"})

# Let anything in flight settle before scoring.
for _ in range(60):
    world.step(render=False)
p = screw_pos()
half = BIN_POCKET / 2
in_bin = bool(abs(p[0] - BIN_XY[0]) < half and abs(p[1] - BIN_XY[1]) < half
              and SURFACE_Z <= p[2] <= SURFACE_Z + BIN_WALL_H)
on_table_reachable = bool(not in_bin and p[2] < SURFACE_Z + 0.02 and 0.25 < np.hypot(*p[:2]) < 0.55)
report = {
    "seed": args.seed, "screw_layout": C.SCREWS[0], "steps": args.steps,
    "attack": args.attack, "response": args.response,
    "first_hold_step": first_hold, "attack_start": attack_start,
    "checkpoint_before_attack": checkpoint_before, "checkpoint_after_attack": checkpoint_after,
    "stopped_at": stopped_at, "final_screw": p.tolist(),
    "detector_threshold": detector.threshold, "first_alarm": alarm_step,
    "alarms_before_attack": sum(1 for a in alarm_steps if attack_start is None or a < attack_start),
    "response_step": response_step, "restore_outcome": restore_outcome,
    "restore_log": restore_log, "checkpoint_restored": checkpoint_restored,
    "total_steps": step,
    "success": in_bin, "dropped_recoverable": on_table_reachable,
}
os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
with open(args.report, "w") as f:
    json.dump(report, f, indent=2)
if args.trace:
    with open(os.path.splitext(args.report)[0] + "_trace.json", "w") as f:
        json.dump(trace, f)
print("[eval]", json.dumps(report), flush=True)
sock.close()
sim_app.close()
