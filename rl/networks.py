"""Goal-conditioned SAC networks for the grasp-checkpoint recovery actor.

Actor input  : wrist + table RGB (deployment cameras, 96x96) and proprioception
               (joints, gripper command, tool pose from FK, target-checkpoint one-hot).
Critic input : privileged simulator state (screw pose relative to the tool) plus
               proprioception and action. Training-only; the deployed actor never sees it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 96
PROPRIO_DIM = 13
PRIV_DIM = 10
ACT_DIM = 5  # dx, dy, dz, dyaw, gripper
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Encoder(nn.Module):
    """Two camera stems, shared trunk (DrQ-style)."""

    def __init__(self, feat=64):
        super().__init__()

        def stem():
            return nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(), nn.Flatten())

        self.wrist, self.table = stem(), stem()
        with torch.no_grad():
            n = self.wrist(torch.zeros(1, 3, IMG, IMG)).shape[1]
        self.proj = nn.Sequential(nn.Linear(2 * n, feat), nn.LayerNorm(feat), nn.Tanh())
        self.out_dim = feat

    def forward(self, wrist, table):
        w = self.wrist(wrist.float() / 255.0 - 0.5)
        t = self.table(table.float() / 255.0 - 0.5)
        return self.proj(torch.cat([w, t], 1))


def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Actor(nn.Module):
    """Deployed policy: its own image encoder + head, deployment observations only."""

    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.head = mlp(self.enc.out_dim + PROPRIO_DIM, 2 * ACT_DIM)

    def forward(self, wrist, table, proprio):
        mu, log_std = self.head(torch.cat([self.enc(wrist, table), proprio], 1)).chunk(2, 1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std

    def sample(self, wrist, table, proprio):
        mu, log_std = self(wrist, table, proprio)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        a = torch.tanh(mu + std * eps)
        logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(1)
        logp = logp - torch.log(1 - a.pow(2) + 1e-6).sum(1)
        return a, logp, torch.tanh(mu)

    @torch.no_grad()
    def act(self, wrist, table, proprio, deterministic=True):
        a, _, mean = self.sample(wrist, table, proprio)
        return mean if deterministic else a


DeployedActor = Actor


class Critic(nn.Module):
    """Twin Q on privileged simulator state + proprioception (training only, no images)."""

    def __init__(self):
        super().__init__()
        d = PROPRIO_DIM + PRIV_DIM + ACT_DIM
        self.q1, self.q2 = mlp(d, 1), mlp(d, 1)

    def forward(self, proprio, priv, act):
        x = torch.cat([proprio, priv, act], 1)
        return self.q1(x).squeeze(1), self.q2(x).squeeze(1)


def random_shift(x, pad=4):
    """DrQ augmentation: pad and randomly crop back, per sample."""
    n, c, h, w = x.shape
    x = F.pad(x.float(), (pad,) * 4, mode="replicate")
    eps = 1.0 / (h + 2 * pad)
    base = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * pad, device=x.device)[:h]
    base = base.unsqueeze(0).repeat(h, 1).unsqueeze(2)
    grid = torch.cat([base, base.transpose(1, 0)], 2).unsqueeze(0).repeat(n, 1, 1, 1)
    shift = torch.randint(0, 2 * pad + 1, (n, 1, 1, 2), device=x.device).float() * 2.0 / (h + 2 * pad)
    return F.grid_sample(x, grid + shift, padding_mode="zeros", align_corners=False)
