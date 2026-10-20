"""v1 actor observations (fixed work-area crop of the table camera). Used by the
actors in evals/actor_040000.pt and evals/actor_ft030000.pt; v2 is obs_utils.py."""
import numpy as np
from PIL import Image

from obs_utils import IMG, WRIST_RENDER, proprio  # noqa: F401  (unchanged in v2)

TABLE_RENDER = (960, 540)
CROP_XY = ((0.30, 0.64), (-0.26, 0.06))


def table_crop(table_camera_model, surface_z):
    (x0, x1), (y0, y1) = CROP_XY
    corners = [table_camera_model.project([x, y, surface_z]) for x in (x0, x1) for y in (y0, y1)]
    u, v = np.array(corners).T / (table_camera_model.W / TABLE_RENDER[0])
    W, H = TABLE_RENDER
    return int(max(0, u.min())), int(max(0, v.min())), int(min(W, u.max())), int(min(H, v.max()))


def images(wrist_rgb, table_rgb, crop):
    w = Image.fromarray(wrist_rgb)
    if w.size != WRIST_RENDER:
        w = w.resize(WRIST_RENDER, Image.BOX)
    t = Image.fromarray(table_rgb)
    if t.size != TABLE_RENDER:
        t = t.resize(TABLE_RENDER, Image.BOX)
    w = np.asarray(w.resize((IMG, IMG), Image.BOX))
    t = np.asarray(t.crop(crop).resize((IMG, IMG), Image.BOX))
    return w.transpose(2, 0, 1).copy(), t.transpose(2, 0, 1).copy()
