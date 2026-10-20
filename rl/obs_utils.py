"""Actor observation construction shared by training (recovery_env) and deployment
(matched_sim/eval_v05.py), so both feed the actor identical inputs.

Images: the wrist camera (box-filtered to 96x96) and a tool-centred fovea of the
full-resolution table camera: a 128 px window around the projection of the fingertip
(FK + camera calibration, both available on the robot), ~1.3 mm per output pixel.
"""
import numpy as np
from PIL import Image

IMG = 96
GOAL_GRASP = 1.0
WRIST_RENDER = (384, 216)
TABLE_RENDER = (1920, 1080)
FOVEA = 128


def fovea_box(table_camera_model, tool_p_world):
    """Pixel box of the fovea around the fingertip's projection, clamped to the image."""
    u, v = table_camera_model.project(tool_p_world)
    W, H = TABLE_RENDER
    u0 = int(np.clip(round(u) - FOVEA // 2, 0, W - FOVEA))
    v0 = int(np.clip(round(v) - FOVEA // 2, 0, H - FOVEA))
    return u0, v0, u0 + FOVEA, v0 + FOVEA


def images(wrist_rgb, table_rgb, box):
    """uint8 CHW 96x96 images; table_rgb must be TABLE_RENDER sized."""
    w = Image.fromarray(wrist_rgb)
    if w.size != WRIST_RENDER:
        w = w.resize(WRIST_RENDER, Image.BOX)
    w = np.asarray(w.resize((IMG, IMG), Image.BOX))
    t = np.asarray(Image.fromarray(table_rgb).crop(box).resize((IMG, IMG), Image.BOX))
    return w.transpose(2, 0, 1).copy(), t.transpose(2, 0, 1).copy()


def proprio(q_rad, closed, tool_p, yaw):
    return np.concatenate([np.degrees(q_rad) / 100.0, [float(closed)],
                           (np.asarray(tool_p) - np.array([0.42, 0.0, 0.05])) / 0.1,
                           [np.sin(2 * yaw), np.cos(2 * yaw)], [GOAL_GRASP]]).astype(np.float32)
