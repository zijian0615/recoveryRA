"""Goal-conditioned SAC learner for the grasp-checkpoint recovery actor.

Reads episodes written by collect.py (demo dir + online dirs, polled), trains, and
publishes ckpt/actor.pt (DeployedActor: encoder + policy head) for the collectors.

- RLPD-style replay: each batch is half teacher demonstrations, half online data.
- Asymmetric actor-critic: the critic sees privileged simulator state (no images);
  the actor learns its own image encoder through the actor loss (DrQ random shifts).
- Actor initialised by behaviour cloning (--init-actor); critic pretrained for
  --critic-warmup TD updates with the actor frozen.
- TD3+BC-style actor loss: Q term normalised by mean |Q|, plus BC on demo samples.

    ~/lerobot/.venv/bin/python learner.py --demos data/demos --online data/online --out ckpt
"""
import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

import networks as N
import task

p = argparse.ArgumentParser()
p.add_argument("--demos", required=True)
p.add_argument("--online", required=True)
p.add_argument("--out", required=True)
p.add_argument("--updates", type=int, default=200_000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--gamma", type=float, default=0.97)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--init-actor", default=None)
p.add_argument("--critic-warmup", type=int, default=5000)
p.add_argument("--q-weight", type=float, default=2.5)
p.add_argument("--bc-weight", type=float, default=1.0)
p.add_argument("--target-entropy", type=float, default=-7.5)
p.add_argument("--publish-every", type=int, default=1000)
p.add_argument("--max-utd", type=float, default=4.0, help="updates per online transition, cap")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
torch.manual_seed(args.seed)
dev = torch.device("cuda")
os.makedirs(args.out, exist_ok=True)


class Replay:
    def __init__(self):
        self.files, self.eps, self.index = set(), [], []

    def poll(self, pattern):
        new = 0
        for f in sorted(glob.glob(pattern)):
            if f in self.files:
                continue
            try:
                d = np.load(f)
                ep = {k: d[k] for k in ("wrist", "table", "proprio", "priv", "action", "reward", "terminal")}
                ep["outcome"] = str(d["outcome"])
                # Relabel with the current checkpoint predicate (task.py).
                unsafe = ep["terminal"] & (ep["reward"] <= -0.99)
                r, term = task.relabel(ep["priv"], ep["proprio"], unsafe)
                T = len(r)
                ep.update(reward=r, terminal=term, action=ep["action"][:T])
                for k in ("wrist", "table", "proprio", "priv"):
                    ep[k] = ep[k][:T + 1]
                ep["outcome"] = "success" if term[-1] and r[-1] > 0.5 else (
                    "unsafe" if term[-1] else "timeout")
            except Exception:
                continue  # partially written; retry next poll
            self.files.add(f)
            i = len(self.eps)
            self.eps.append(ep)
            self.index += [(i, t) for t in range(len(ep["action"]))]
            new += len(ep["action"])
        return new

    def __len__(self):
        return len(self.index)

    def sample(self, n):
        picks = [self.index[j] for j in np.random.randint(len(self.index), size=n)]
        out = {k: [] for k in ("wrist", "table", "proprio", "priv", "nwrist", "ntable", "nproprio",
                               "npriv", "action", "reward", "done")}
        for i, t in picks:
            e = self.eps[i]
            for k in ("wrist", "table", "proprio", "priv"):
                out[k].append(e[k][t])
                out["n" + k].append(e[k][t + 1])
            out["action"].append(e["action"][t])
            out["reward"].append(e["reward"][t])
            out["done"].append(float(e["terminal"][t]))
        return {k: torch.as_tensor(np.stack(v).astype(np.float32) if k in ("reward", "done")
                                   else np.stack(v)).to(dev) for k, v in out.items()}


def cat(a, b):
    return {k: torch.cat([a[k], b[k]]) for k in a}


critic = N.Critic().to(dev)
critic_t = copy.deepcopy(critic)
actor = N.Actor().to(dev)
if args.init_actor:
    actor.load_state_dict(torch.load(args.init_actor))
log_alpha = torch.tensor(np.log(0.01), device=dev, requires_grad=True)
target_ent = args.target_entropy
opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr)
opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
opt_al = torch.optim.Adam([log_alpha], lr=args.lr)


def publish(step):
    sd = {k: v.cpu() for k, v in actor.state_dict().items()}
    tmp = os.path.join(args.out, "actor.pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, os.path.join(args.out, "actor.pt"))
    torch.save(sd, os.path.join(args.out, f"actor_{step:06d}.pt"))


demo, online = Replay(), Replay()
demo.poll(os.path.join(args.demos, "*", "*.npz"))
print(f"[learner] demos: {len(demo.eps)} episodes, {len(demo)} transitions", flush=True)
publish(0)
log = open(os.path.join(args.out, "log.jsonl"), "a")
t0, last_poll, n_online = time.time(), 0.0, 0
step = 0
while step < args.updates:
    if time.time() - last_poll > 10:
        n_online += online.poll(os.path.join(args.online, "*", "*.npz"))
        last_poll = time.time()
    # Do not outrun data collection by more than max_utd (demos alone allow a warm start).
    if step > args.critic_warmup and step > args.max_utd * max(n_online, 1) + args.critic_warmup:
        time.sleep(1.0)
        continue
    half = args.batch // 2
    if len(online) >= 1000:
        b = cat(demo.sample(half), online.sample(args.batch - half))
        is_demo = torch.cat([torch.ones(half), torch.zeros(args.batch - half)]).to(dev)
    else:
        b = demo.sample(args.batch)
        is_demo = torch.ones(args.batch, device=dev)
    w, tb = N.random_shift(b["wrist"]), N.random_shift(b["table"])
    nw, ntb = N.random_shift(b["nwrist"]), N.random_shift(b["ntable"])
    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        na, nlogp, _ = actor.sample(nw, ntb, b["nproprio"])
        tq1, tq2 = critic_t(b["nproprio"], b["npriv"], na)
        target = b["reward"] + args.gamma * (1 - b["done"]) * (torch.min(tq1, tq2) - alpha * nlogp)
    q1, q2 = critic(b["proprio"], b["priv"], b["action"])
    loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    opt_c.zero_grad()
    loss_c.backward()
    opt_c.step()

    bc_w = args.bc_weight
    loss_a = torch.zeros((), device=dev)
    if step >= args.critic_warmup:
        a, logp, mean = actor.sample(w, tb, b["proprio"])
        aq = torch.min(*critic(b["proprio"], b["priv"], a))
        lam = args.q_weight / aq.abs().mean().detach().clamp(min=1e-3)
        bc = ((mean - b["action"].clamp(-0.97, 0.97)) ** 2).sum(1)
        loss_a = (alpha * logp).mean() - lam * aq.mean() + bc_w * (bc * is_demo).sum() / is_demo.sum()
        opt_a.zero_grad()
        loss_a.backward()
        opt_a.step()
        loss_al = -(log_alpha * (logp.detach() + target_ent)).mean()
        opt_al.zero_grad()
        loss_al.backward()
        opt_al.step()

    with torch.no_grad():
        for pt, pp in zip(critic_t.parameters(), critic.parameters()):
            pt.mul_(0.99).add_(0.01 * pp)
    step += 1
    if step % args.publish_every == 0:
        publish(step)
        recent = [e["outcome"] for e in online.eps[-50:]]
        rec = {"step": step, "time": round(time.time() - t0), "online_eps": len(online.eps),
               "online_transitions": n_online, "loss_c": loss_c.item(), "loss_a": loss_a.item(),
               "alpha": alpha.item(), "q": q1.mean().item(), "bc_w": bc_w,
               "recent_success": recent.count("success") / max(1, len(recent)),
               "recent_unsafe": recent.count("unsafe") / max(1, len(recent))}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print("[learner]", rec, flush=True)
publish(step)
