# Learned recovery actor (goal-conditioned SAC)

Implements the memo's restore step with RL: from the real post-attack state, learn
\(a_t \sim \pi_R(a_t \mid o_t, \mathcal C^{\text{target}})\) that re-establishes the
grasp checkpoint, then hand back to the original ACT policy.

## Components

- **Post-attack states** (`runs/rl_snapshots/`, 59): ACT runs in the matched cell, a
  gripper-open attack fires after the hold (onset +0..40 steps, 3..8 steps long),
  the detector alarms, and the simulator state is saved once the screw has landed.
  Split by seed: 48 train, 11 test (seed % 5 == 0).
- **Environment** (`recovery_env.py`): resets the robot and screw to a snapshot
  (training: 50 % perturbed by ±2 cm and a random spin).
  - Action: tool Δx, Δy, Δz (≤15 mm), Δyaw (≤0.15 rad), gripper; executed with IK, 10 Hz.
  - Rendering: 1-spp path tracing + OptiX denoiser (realtime mode is grainy here
    because DLSS/NGX is unavailable).
- **Actor observation**: wrist RGB + table RGB (96x96), joints, gripper command,
  FK tool pose, target-checkpoint one-hot.
  - v1 (`obs_utils_v1.py`): fixed crop of the work area, ~3.5 mm/px.
  - v2 (`obs_utils.py`): 128 px fovea around the projected fingertip, ~1.3 mm/px.
- **Reward** (`task.py`, the memo's form): +1 when the grasp predicate holds for 3
  steps, −1 unsafe (screw off the table or out of reach, tool at the bin), otherwise
  λ[Φ(s') − Φ(s)] − c. Timeout (60 steps) is a failure, not unsafe.
- **Prior data**: episodes from a privileged scripted teacher (simulator screw pose),
  used only as replay data and for the BC initialisation.
- **Learner** (`learner.py`):
  - Asymmetric SAC: the critic sees privileged state + proprioception + action; the
    actor has its own CNN encoder with DrQ shifts.
  - RLPD-style 50/50 demo/online batches.
  - BC initialisation and 5k critic warm-up.
  - TD3+BC-style actor loss: Q/mean|Q| plus BC on demo samples.
  - Collectors: `collect.py`, Isaac processes rolling out the latest stochastic actor.

## Runs

| run | obs | grasp predicate | result |
| --- | --- | --- | --- |
| A `ckpt` → `evals/actor_040000.pt` | v1 | lifted 4 cm, tool within 3 cm | online 0.06→0.90 in 40k updates; held-out test 30/33 (BC init 5/33) |
| B `ckpt2` → `evals/actor_ft030000.pt` | v1 | + grasp within 6 mm of shank mid | fine-tune of A, 30k; test 29/33; median grasp offset 5.8→1.8 mm |
| C `ckpt3` | v1 | + within 1.5 mm | fine-tune of B: peaked 0.42 online, then collapsed to 0 |
| D `ckpt4` | v2 | within 1.5 mm | from BC init: Q overestimated, no learning in 30k updates; stopped |

## End-to-end (matched cell, 10 paired seeds, `runs/v05/matrix_pt1`)

| arm | success | checkpoint restored |
| --- | --- | --- |
| scripted restore | 7/10 | 9/10 |
| RL A (`ours_rl_v1`) | 4/10 | 8/10 |
| RL B (`ours_rl`) | 5/10 | 9/10 |

The RL actor restores the checkpoint as often as the scripted one, in ~15 steps
instead of ~130. Most end-to-end failures happen **after** hand-back: ACT immediately
commands the gripper open (≈0) and drops the screw. Across all runs ACT keeps the grip
when the head is 10–16 mm from the jaws (demos 14.5–14.9 mm) and releases at
16.7–20.6 mm. ACT only resumes from a grasp within ~1.5 mm of the demonstrated one,
which is the target of runs C and D. No RL actor reaches that precision yet.

## Reproduce

    ~/lerobot/.venv/bin/python learner.py --demos data/demos --online data/online --out ckpt --init-actor ckpt_bc_init.pt
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim-venv/bin/python collect.py --mode policy --out data/online/w0 --ckpt ckpt/actor.pt
    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim-venv/bin/python collect.py --mode eval --eval-augment --snapshot-split test --seed 7 --out evals/test --ckpt evals/actor_040000.pt --episodes 33
    # end-to-end with the v1 actors:
    matched_sim/eval_v05.py --response ours_rl --rl-actor rl/evals/actor_ft030000.pt --rl-obs v1 --render pathtrace1 ...

The v1 data and actors were produced before `task.py` and `recovery_env.py` moved to
the 1.5 mm predicate and v2 observations. Their raw data is in `data_v1/`.
