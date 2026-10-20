# Screw pick-and-place attack recovery pilot

## Task and assets

The task is to grasp one M6 countersunk screw while other screws remain visible,
then deposit it in the yellow bin. The project uses the existing screw Isaac Sim
cell, `zijian2022/fanuc_er4ia_screw_single_v4` (LeRobot v3, 10 episodes,
3482 frames, 10 Hz), and the supplied ACT checkpoint. The policy receives three
RGB views plus six joint angles and the gripper bit. The dataset has no recorded
`task.phase` or `task.checkpoint` fields.

## Checkpoints to validate in simulation

- **Grasp restored:** the selected screw is stably elevated with closed jaws for
  a dwell interval; screw position and contacts are evaluation signals only.
- **Place complete:** a screw is inside the bin pocket, below the rim, and remains
  there after settling. The current pilot uses a final pose check; support and
  settling checks still need to be added before claiming task success.
- A screw dropped onto reachable table space is recoverable, not unsafe. A screw
  outside the reachable area or dangerous collision is unsafe. A timeout without
  such an event is a separate failure.

## Current experiments

`offline_detector.py` trains an empirical state/action novelty diagnostic on
8 episodes and tests on 2 held-out episodes. It injects a five-frame gripper-open
command into a held-object portion of each test trace. This only checks whether
the command is unusual in the demonstration data. It does not simulate a drop,
learn from images, identify phases, or estimate the memo's likelihood score.

`run_pilot.sh` prepares paired normal and attack/continue Isaac runs with the
same seed. The attack starts only after a screw has been lifted, using privileged
simulator state as the *experiment trigger*. The controller does not receive that
state. After attack termination, ACT's old action queue is cleared and inference
continues from current observations. This is the switch-and-continue comparison,
not restore-and-resume. A valid comparison requires a successful normal run and
an actual attack trigger. `summary.json` records both gates.

## Required work before an Ours claim

1. Establish a successful no-attack ACT rollout on the matching screw scene.
2. Add a detector with observation-only inputs and phase-aware thresholds.
3. Train or implement an observation-conditioned recovery actor that reconstructs
   a physical grasp checkpoint after the screw is dropped. Use simulator poses,
   contacts, and collision data only for rewards and validation.
4. Verify the original ACT policy can continue after checkpoint restoration;
   clear its 100-step action queue and measure the final bin outcome.
5. Compare detect-and-stop, continue, inverse replay, and restore-and-resume on
   paired layouts and attack strengths. Report recovery rate, unsafe rate,
   end-to-end success, and latency with confidence intervals.

The previous single-screw baseline returned `success: false`, but that scene
removed the other visible screws used during demonstration recording. It is not
valid evidence about the corrected multi-screw setup. The corrected simulation
has not yet been physically run from this Codex environment, which has no GPU
device nodes.

## Update 2026-10-13: matched scene and first Ours validation

The v4 demonstrations show **one** screw on a plain table in the v0.5 cell, not the
eight-screw printed-mat `screw_sim` scene; the corrected multi-screw baseline is
therefore also mismatched. `matched_sim/` rebuilds the recording layout from the
demo kinematics (LAYOUT.md) and evaluates detect → restore → resume against stop,
continue and inverse replay on 10 paired seeds (RESULTS.md). Ours: 9/9 task success
after a gripper-open attack vs 0/9, 1/9, 4/10 for the baselines; clean ACT 8/10.
The restore actor is scripted (camera + IK), not yet the learned SAC actor.
