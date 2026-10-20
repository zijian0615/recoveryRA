"""Diagnostic: behaviour cloning only, same actor and augmentation as the learner."""
import glob
import sys

import numpy as np
import torch

import networks as N

out, steps = sys.argv[1], int(sys.argv[2])
dev = torch.device("cuda")
W, T, P, A = [], [], [], []
for f in glob.glob("data/demos/*/*.npz"):
    d = np.load(f)
    n = len(d["action"])
    W.append(d["wrist"][:n]); T.append(d["table"][:n]); P.append(d["proprio"][:n]); A.append(d["action"])
W, T, P, A = (torch.as_tensor(np.concatenate(x)).to(dev) for x in (W, T, P, A))
actor = N.Actor().to(dev)
opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
for s in range(steps):
    i = torch.randint(len(A), (256,), device=dev)
    mu, _ = actor(N.random_shift(W[i]), N.random_shift(T[i]), P[i])
    loss = ((torch.tanh(mu) - A[i].clamp(-0.97, 0.97)) ** 2).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 2000 == 0:
        print(s, round(loss.item(), 4), flush=True)
torch.save({k: v.cpu() for k, v in actor.state_dict().items()}, out)
