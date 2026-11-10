"""SAC networks on frozen ResNet-18 features (rl/sac).

The trunk (rl/ppo/nets.ResNetEncoder, ImageNet ResNet-18 conv1..layer3, frozen) runs once per collected frame; what the
replay buffer stores is its pooled output: per camera spatial-softmax keypoints (256 x 2, fixed temperature) + channel
means (256) = 768, two cameras = 1536 (fp16). The trainable "encoder" is a projection Linear(1536 -> 128) -> LayerNorm
-> tanh shared by actor and critics and trained by the critic loss only (the actor sees it detached), as in
ManiSkill3 sac_rgbd.py / DrQ. Proprio (150, RunningNorm) is concatenated after the projection.

  actor:  [z 128 | prop 150] -> MLP 256, 256 -> mean (act_dim), log_std (act_dim); tanh-squashed Gaussian
  critic: [z 128 | prop 150 | action] -> MLP 256, 256 (LayerNorm) -> 1, an ensemble of n_critics (min over a random
          subset of n_min_q for the target, RLPD style; n_critics=2, n_min_q=2 is plain SAC)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
FEAT_DIM = 2 * (2 * 256 + 256)          # two cameras x (keypoints 512 + channel means 256)


@torch.no_grad()
def pool_features(h):
    """Frozen trunk maps [B, n_cam, C, h, w] -> [B, n_cam * 3C] (spatial softmax, temperature 1, + channel means)."""
    B, n_cam, C, hh, ww = h.shape
    x = h.reshape(B * n_cam, C, hh * ww).float()
    p = F.softmax(x, dim=-1)
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, hh, device=h.device), torch.linspace(-1, 1, ww, device=h.device), indexing="ij")
    kx = (p * xs.reshape(1, 1, -1)).sum(-1); ky = (p * ys.reshape(1, 1, -1)).sum(-1)
    f = torch.cat([kx, ky, x.mean(-1)], -1)                    # [B * n_cam, 3C]
    return f.reshape(B, -1)


def mlp(inp, hidden, out, layer_norm=False):
    layers, d = [], inp
    for hdim in hidden:
        layers.append(nn.Linear(d, hdim))
        if layer_norm:
            layers.append(nn.LayerNorm(hdim))
        layers.append(nn.ReLU(inplace=True))
        d = hdim
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class PlainConv(nn.Module):
    """ManiSkill3 sac_rgbd.py encoder, verbatim: 5 conv layers (zero biases) with max-pooling down to 4x4x64, then
    Linear(1024 -> 256) + ReLU. Input uint8 [B, H, W, C] (cameras concatenated along channels), scaled by 1/255 only.
    Supports 64x64 and 128x128 inputs."""

    def __init__(self, in_ch=6, res=64, out=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(4 if res == 128 else 2),
            nn.Conv2d(16, 32, 3, padding=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1, bias=True), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU(inplace=True))
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.bias)
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, out), nn.ReLU(inplace=True))
        self.out_dim = out

    def forward(self, img):
        x = img.permute(0, 3, 1, 2).float() / 255.0
        return self.fc(self.cnn(x).flatten(1))


class Projection(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, out=128):
        super().__init__()
        self.proj = nn.Linear(feat_dim, out)
        self.ln = nn.LayerNorm(out)

    def forward(self, f):
        return torch.tanh(self.ln(self.proj(f.float())))


class Actor(nn.Module):
    def __init__(self, z_dim, prop_dim, act_dim, hidden=(256, 256), last_act=False):
        super().__init__()
        if last_act:                      # ManiSkill: make_mlp(..., last_act=True) -> separate mean / log_std heads
            self.net = nn.Sequential(mlp(z_dim + prop_dim, hidden[:-1], hidden[-1]), nn.ReLU(inplace=True), nn.Linear(hidden[-1], 2 * act_dim))
        else:
            self.net = mlp(z_dim + prop_dim, hidden, 2 * act_dim)
        self.act_dim = act_dim

    def forward(self, z, prop):
        mu, log_std = self.net(torch.cat([z, prop], -1)).chunk(2, -1)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std

    def sample(self, z, prop, deterministic=False):
        """tanh-squashed Gaussian in [-1, 1]; returns (action, log_prob, mean_action)."""
        mu, log_std = self(z, prop)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = mu if deterministic else dist.rsample()
        a = torch.tanh(x)
        logp = (dist.log_prob(x) - torch.log(1 - a.pow(2) + 1e-6)).sum(-1)
        return a, logp, torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, z_dim, prop_dim, act_dim, n_critics=2, hidden=(256, 256), layer_norm=True):
        super().__init__()
        self.qs = nn.ModuleList([mlp(z_dim + prop_dim + act_dim, hidden, 1, layer_norm) for _ in range(n_critics)])

    def forward(self, z, prop, a):
        x = torch.cat([z, prop, a], -1)
        return torch.stack([q(x).squeeze(-1) for q in self.qs], 0)       # [n_critics, B]


class SACModel(nn.Module):
    """Projection (shared, critic-trained) + actor + critic ensemble + target critic + log_alpha + proprio norm."""

    def __init__(self, prop_dim, act_dim, z_dim=128, n_critics=2, layer_norm=True, init_alpha=1.0,
                 encoder="resnet", image=128, n_cam=2, hidden=(256, 256), prop_norm=True, last_act=False):
        super().__init__()
        from nets import RunningNorm                                    # rl/ppo/nets.py
        self.norm = RunningNorm(prop_dim) if prop_norm else nn.Identity()
        if encoder == "plainconv":
            self.encoder = PlainConv(3 * n_cam, image, 256); z_dim = 256
        else:
            self.encoder = Projection(FEAT_DIM, z_dim)
        self.actor = Actor(z_dim, prop_dim, act_dim, hidden, last_act)
        self.critic = Critic(z_dim, prop_dim, act_dim, n_critics, hidden, layer_norm=layer_norm)
        self.critic_target = Critic(z_dim, prop_dim, act_dim, n_critics, hidden, layer_norm=layer_norm)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(init_alpha)))
        self.act_dim = act_dim

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, feat, prop, deterministic=False):
        z = self.encoder(feat)
        a, _, mean = self.actor.sample(z, self.norm(prop), deterministic)
        return mean if deterministic else a

    @torch.no_grad()
    def soft_update(self, tau):
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.lerp_(p, tau)
