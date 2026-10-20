# Matched v0.5 screw cell for `zijian2022/fanuc_er4ia_screw_single_v4`

`screw_sim/eval_act_sim.py` does not reproduce the recording conditions of the v4
demonstrations, so its failures say nothing about the ACT policy:

| | v4 recordings | `screw_sim` eval |
| --- | --- | --- |
| Table | plain white/grey, enclosure | printed mat, no enclosure |
| Screws visible | **one** | eight, randomly scattered |
| Grasp location | fixed, (0.43, -0.13) +- 6 mm | random over 0.32..0.50 x -0.20..0.08 |
| Bin | (0.511, 0.108) | (0.370, 0.161) |
| Tool length | 293 mm (v0.5) | screw_sim gripper |
| Cameras | v0.5 calibration | older calibration |

## How the layout was derived (kinematics only)

Forward kinematics on every episode, with the v0.5 293 mm J6-to-fingertip offset:

- Start state `[0, 3.3, -11.9, 0, -74.8, 0]` puts the tip exactly at v0.5 HOME
  (0.30, 0, 0.24) pointing down, which confirms the v0.5 tool.
- Gripper-close frames: tip at (0.420..0.435, -0.122..-0.139, -2.1 mm), i.e.
  3 mm above the v0.5 work surface (-5.1 mm). Jaw axis gives screw yaw 33..53 deg.
- Release frames: identical in all 10 episodes, tip at (0.511, 0.108, +82 mm).

The dataset's rendered appearance (photographic background, different robot
material) is not reproduced; path tracing removes the default renderer's grain.

## Reference images

`runs/diag_seed20/compare.png` (dataset vs screw_sim) and
`runs/v05/compare.png` (dataset vs this cell).
