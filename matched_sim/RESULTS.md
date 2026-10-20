# Ours (detect → restore → resume) — first simulation validation, 2026-10-13

Scene: matched v0.5 cell (see LAYOUT.md), one M6 screw at the demo grasp location
(v1: fixed pick/place; yaw 33–53°), ACT `act_fanuc_er4ia_screw_single_v4` with
full-chunk execution (`n_action_steps=100`, `--replan-interval 0`), RTX realtime
renderer, 600 control steps (10 Hz). Data: `runs/v05/matrix/`, `summary.json`.

Attack: once the screw is lifted >2 cm (simulator truth, experiment trigger only),
wait 10 steps, then force the gripper command open for 5 steps. Every response
waits for the detector alarm *and* the end of the attack window.

| arm | valid n | task success | 95% CI (Wilson) | paired vs ours (exact McNemar) |
| --- | --- | --- | --- | --- |
| clean (no attack) | 10 | 8 | 0.49–0.94 | — |
| detect + stop | 9 | 0 | 0.00–0.30 | p = 0.004 |
| detect + continue (clear ACT queue) | 9 | 1 | 0.02–0.44 | p = 0.008 |
| detect + inverse replay | 10 | 4 | 0.17–0.69 | p = 0.06 (5 vs 0 discordant) |
| **ours** | 9 | **9** | **0.70–1.00** | — |

Valid = grasp checkpoint held at attack onset and broken after it. Seed 4 is
excluded where ACT never lifted the screw (no attack fired).

Ours details: detection delay 0 steps in 9/9; grasp checkpoint restored 9/9
(simulator truth at hand-back, evaluation only); 8/9 on the first re-grasp, seed 8
needed the camera-verified retry. Recovery latency 136 steps (13.6 s), 209 with retry.

Detector false alarms: 0 before the attack in 9/9 ours trials; 3/10 clean runs
alarm once, all at step 300 (an ACT chunk boundary). 2 of those 3 had already failed
from a natural slip.

## What "ours" is here (and is not)

- Detector: kNN novelty on measured joints, conditioned on the (state gripper,
  commanded gripper) pair, threshold = 99th percentile of 3 clean calibration runs
  (seeds 100–102, disjoint from evaluation). Not the image-based D_t of the memo.
- Restore: **scripted**, not the memo's goal-conditioned SAC actor. Return to the
  start pose → table-camera difference against the episode's first frame →
  calibrated back-projection → IK re-grasp → camera check that the screw left the
  table (one retry) → eased move to a demo post-lift pose → clear ACT queue → resume.
- Inputs: camera images, joint states and calibration only. Simulator object poses
  are used only to trigger the attack and to score.

## Limitations

- One attack type, one onset (hold + 10), one duration, v1 layout only, n ≤ 10.
- Realtime renderer; dataset frames use a different photographic background and
  robot appearance. Path-traced runs were too slow for the matrix (1/1 success clean).
- Sim grasps slip in transport without any attack (clean failures 2/10); this caps
  every arm's end-to-end success, which is why checkpoint recovery is reported apart.
- Screws that land upright (seen once after a false alarm) are not handled.
- The reference-difference localiser needs the arm back at the start pose and the
  rest of the table unchanged.

## Update: learned (RL) restore, 1-spp path tracing (`runs/v05/matrix_pt1`)

| arm | success | checkpoint restored |
| --- | --- | --- |
| clean | 9/10 | — |
| stop / continue / inverse replay | 0/10, 2/10, 1/10 | — |
| scripted restore | 7/10 | 9/10 |
| RL restore (goal-conditioned SAC, `rl/`) | 5/10 | 9/10 |

Most failures after a restored checkpoint come from ACT itself: it opens the gripper
right after hand-back unless the grasp is within ~1.5 mm of the demonstrated one.
Details in `rl/README.md`.
