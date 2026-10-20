# recoveryRA: task-state recovery for attacked robot-arm manipulation

Detect → Restore → Resume on a FANUC ER-4iA screw pick-and-place task in Isaac Sim 6.0.
An attack forces the gripper open during transport. A detector flags it, and a recovery
actor re-establishes the grasp checkpoint. The original ACT policy then finishes the task.

## Layout

| path | content |
| --- | --- |
| `matched_sim/` | Scene matching the `fanuc_er4ia_screw_single_v4` recordings (`LAYOUT.md`) and the attack/eval loop (`eval_v05.py`). Also the detector and scripted restore (`recovery.py`), paired-seed runner (`run_matrix.sh`), summaries (`summarize.py`), results (`RESULTS.md`). |
| `rl/` | Learned restore, goal-conditioned asymmetric SAC: env, networks, learner, collectors. `README.md` has the runs and results; `evals/actor_*.pt` holds the evaluated actors. |
| `fanuc_er4ia_sim/` | v0.5 calibrated cell (hackathon starter) that the matched scene builds on. |
| `screw_sim/` | Earlier screw cell and ACT policy server (`act_policy_server.py`). |
| `runs/` | Kept: detector calibration traces, the 59 post-attack snapshots, matrix summaries. |
| `RESEARCH_STATUS.md` | Status notes. |

## Headline results (10 paired seeds, 1-spp path tracing)

| response after detection | task success | grasp checkpoint restored |
| --- | --- | --- |
| no attack | 9/10 | — |
| stop / continue / inverse replay | 0/10, 2/10, 1/10 | — |
| scripted restore | 7/10 | 9/10 |
| RL restore | 5/10 | 9/10 |

- The detector fired on the first attacked step in 97/98 attack runs.
- Most remaining failures happen after hand-back. ACT reopens the gripper unless the
  restored grasp is within ~1.5 mm of the demonstrated one.
- Limitations are listed in `matched_sim/RESULTS.md` and `rl/README.md`.

## Setup

- Isaac Sim 6.0 venv: `~/isaacsim-venv`.
- LeRobot venv with torch + CUDA: `~/lerobot/.venv`.
- Download into `repos/` (not tracked):
  - `repos/dataset`: `zijian2022/fanuc_er4ia_screw_single_v4`
  - `repos/policy`: `zijian2022/act_fanuc_er4ia_screw_single_v4`
- Some scripts contain absolute paths for these venvs; adjust them for your machine.

```bash
cd matched_sim
./run.sh ../runs/v05/example -- --seed 0 --jitter 0 --replan-interval 0 --steps 600 \
  --attack open_when_held --attack-delay 10 --response ours --render pathtrace1
```
