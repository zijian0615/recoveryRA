"""Isaac-side worker: roll out the recovery env and write episodes for the learner.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacsim-venv/bin/python collect.py --mode expert --out data/demos --episodes 40
    ... --mode policy --out data/online/w0 --ckpt ckpt/actor.pt   (reloads weights as they change)
    ... --mode eval   --out data/eval --ckpt ckpt/actor.pt --episodes 30
"""
import argparse
import json
import os
import time

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("expert", "policy", "eval", "random"), required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--episodes", type=int, default=10**9)
parser.add_argument("--ckpt", default=None)
parser.add_argument("--snapshots", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "..", "runs", "rl_snapshots"))
parser.add_argument("--snapshot-split", choices=("train", "test", "all"), default="train")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--noise", type=float, default=0.3, help="expert action noise")
parser.add_argument("--save-frames", action="store_true")
parser.add_argument("--eval-augment", action="store_true",
                    help="after one unperturbed pass, keep cycling snapshots with perturbations")
parser.add_argument("--sequential", action="store_true",
                    help="visit snapshots in order without augmentation (implied by --mode eval)")
args = parser.parse_args()

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True})

import torch  # noqa: E402
import recovery_env as E  # noqa: E402
import networks as N  # noqa: E402

snaps = E.load_snapshots(args.snapshots)
# Snapshot split by seed parity keeps evaluation drops unseen during training.
if args.snapshot_split != "all":
    snaps = [s for s in snaps if (s["seed"] % 5 == 0) == (args.snapshot_split == "test")]
print(f"[collect] {len(snaps)} snapshots ({args.snapshot_split})", flush=True)
env = E.RecoveryEnv(snaps, seed=args.seed)
os.makedirs(args.out, exist_ok=True)
actor, ckpt_mtime = None, None


def maybe_reload():
    global actor, ckpt_mtime
    if not args.ckpt or not os.path.exists(args.ckpt):
        return
    m = os.path.getmtime(args.ckpt)
    if m != ckpt_mtime:
        a = N.DeployedActor()
        a.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        a.eval()
        actor, ckpt_mtime = a, m


def to_t(o):
    return (torch.from_numpy(o["wrist"])[None], torch.from_numpy(o["table"])[None],
            torch.from_numpy(o["proprio"])[None])


ep = len([f for f in os.listdir(args.out) if f.endswith(".npz")])
results = []
while ep < args.episodes:
    maybe_reload()
    seq = args.mode == "eval" or args.sequential
    # Eval visits each snapshot once unperturbed, then perturbed copies (fixed --seed).
    aug = (not seq) or (args.eval_augment and ep >= len(snaps))
    o = env.reset(snaps[ep % len(snaps)] if seq else None, augment=aug)
    buf = {k: [o[k]] for k in o}
    acts, rews, terms = [], [], []
    t0 = time.time()
    while True:
        if args.mode == "expert":
            a = env.expert_action(noise=args.noise)
        elif args.mode == "random" or actor is None:
            a = np.random.uniform(-1, 1, 5)
        else:
            a = actor.act(*to_t(o), deterministic=args.mode == "eval")[0].numpy()
        o, r, term, trunc, info = env.step(a)
        for k in o:
            buf[k].append(o[k])
        acts.append(a)
        rews.append(r)
        terms.append(term)
        if args.save_frames and len(acts) % 5 == 0:
            from PIL import Image
            fd = os.path.join(args.out, f"frames_{ep:05d}")
            os.makedirs(fd, exist_ok=True)
            Image.fromarray(o["wrist"].transpose(1, 2, 0)).save(os.path.join(fd, f"{len(acts):03d}_wrist.png"))
            Image.fromarray(o["table"].transpose(1, 2, 0)).save(os.path.join(fd, f"{len(acts):03d}_table.png"))
        if term or trunc:
            break
    outcome = "success" if info["success"] else "unsafe" if info["unsafe"] else "timeout"
    np.savez_compressed(os.path.join(args.out, f"ep_{ep:05d}.npz"),
                        **{k: np.stack(v) for k, v in buf.items()},
                        action=np.asarray(acts, np.float32), reward=np.asarray(rews, np.float32),
                        terminal=np.asarray(terms, bool), outcome=outcome,
                        snapshot_seed=env.snap["seed"], ckpt_mtime=ckpt_mtime or 0.0)
    results.append({"ep": ep, "snapshot_seed": env.snap["seed"], "outcome": outcome,
                    "steps": len(acts), "return": float(sum(rews))})
    print(f"[collect] ep {ep} {outcome} steps {len(acts)} return {sum(rews):.2f} "
          f"({(time.time() - t0) / len(acts):.3f}s/step)", flush=True)
    ep += 1
with open(os.path.join(args.out, "results.json"), "w") as f:
    json.dump(results, f, indent=1)
sim_app.close()
