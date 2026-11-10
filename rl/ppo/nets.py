"""Symmetric pixel actor-critic on a fine-tuned ImageNet ResNet-18 trunk over the two benchmark cameras.

Port of assembly-nist cap/resnet_nets.py + cap/pixel_nets.py + assembly_policy/rl/nets.py, trimmed to what the pixel
policy actually uses (the point-cloud SimBa nets are gone). Module names are kept, so an assembly-nist checkpoint loads.

  img uint8 [B, 2, H, W, 3] -> ResNet-18 conv1..layer3 (ImageNet init, BatchNorm frozen) -> spatial-softmax keypoints
      + channel means per camera -> Linear -> LayerNorm -> tanh latent (128), shared by actor and critic
  proprio 150 = 5 frames x (tcp pose 7, twist 6, arm joints 7, fingers 2, previous action 8) -> running normalization
  heads: Linear 256 -> 2 x [x + Linear(SiLU(Linear(LN(x), 1024)), 256)] -> LN -> Linear (8 action mean | 1 value)
  Gaussian policy with a state-independent log-std.
No privileged information anywhere: neither head sees the part pose.
"""
from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

LATENT = 128
TRUNK_CH = 256                       # layer3 output channels
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class RunningNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, clip=10.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.eps, self.clip = eps, clip

    @torch.no_grad()
    def update(self, x):
        x = x.reshape(-1, x.shape[-1]).float()
        n = torch.tensor(float(x.shape[0]), device=x.device)
        if float(n) == 0:
            return
        s1, s2 = x.sum(0), (x * x).sum(0)
        bm = s1 / n
        bv = (s2 / n - bm * bm).clamp_min(0)
        tot = self.count + n
        delta = bm - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + bv * n + delta * delta * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return ((x - self.mean) / torch.sqrt(self.var + self.eps)).clamp(-self.clip, self.clip)


class ResBlock(nn.Module):
    def __init__(self, dim=256, expansion=1024):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, expansion)
        self.fc2 = nn.Linear(expansion, dim)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, expansion=1024, blocks=2):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.blocks = nn.Sequential(*[ResBlock(hidden, expansion) for _ in range(blocks)])
        self.ln = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.head(self.ln(self.blocks(self.inp(x))))


class RandomShift(nn.Module):
    """DrQ-v2 random shift: replicate-pad by `pad` pixels, then crop at a random integer offset per image."""

    def __init__(self, pad=4):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, _, h, w = x.shape
        x = F.pad(x, (self.pad,) * 4, mode="replicate")
        eps = 1.0 / (h + 2 * self.pad)
        ar = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        ar = ar.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base = torch.cat([ar, ar.transpose(1, 0)], dim=2).unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype) * (2.0 / (h + 2 * self.pad))
        return F.grid_sample(x, base + shift, padding_mode="zeros", align_corners=False)


class SpatialSoftmax(nn.Module):
    """Per-channel softmax over the feature map -> expected (x, y) in [-1, 1]: C keypoints = 2C numbers."""

    def __init__(self, h, w):
        super().__init__()
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("px", xs.reshape(1, 1, -1)); self.register_buffer("py", ys.reshape(1, 1, -1))
        self.temp = nn.Parameter(torch.ones(1))

    def forward(self, x):
        p = F.softmax(x.flatten(2) / self.temp, dim=-1)                       # [B, C, HW]
        return torch.cat([(p * self.px).sum(-1), (p * self.py).sum(-1)], -1)   # [B, 2C]


def _freeze_bn(m: nn.Module):
    for name, c in m.named_children():
        if isinstance(c, nn.BatchNorm2d):
            f = FrozenBatchNorm2d(c.num_features, eps=c.eps)
            f.weight.copy_(c.weight.data); f.bias.copy_(c.bias.data)
            f.running_mean.copy_(c.running_mean); f.running_var.copy_(c.running_var)
            setattr(m, name, f)
        else:
            _freeze_bn(c)


class ResNetEncoder(nn.Module):
    def __init__(self, res=128, n_cam=2, out=LATENT, freeze=False, pretrained=True):
        super().__init__()
        w = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=w)
        self.trunk = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
        _freeze_bn(self.trunk)
        if freeze:
            for p in self.trunk.parameters():
                p.requires_grad_(False)
        hw = (res + 15) // 16
        self.ss = SpatialSoftmax(hw, hw)
        self.proj = nn.Linear(n_cam * (2 * TRUNK_CH + TRUNK_CH), out)
        self.ln = nn.LayerNorm(out)
        self.register_buffer("mean", torch.tensor(_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(_STD).view(1, 3, 1, 1))
        self.n_cam, self.res, self.freeze = n_cam, res, freeze
        self.amp = bool(int(os.environ.get("HYBRIDRSI_AMP", "1")))     # bf16 autocast for the trunk; heads stay fp32

    def forward(self, img):
        """img uint8 [B, n_cam, H, W, 3] -> latent [B, out]."""
        B = img.shape[0]
        x = img.permute(0, 1, 4, 2, 3).reshape(B * self.n_cam, 3, img.shape[2], img.shape[3]).float() / 255.0
        x = (x - self.mean) / self.std
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (self.amp and x.is_cuda) else contextlib.nullcontext()
        with ctx:
            if self.freeze:
                with torch.no_grad():
                    h = self.trunk(x)
            else:
                h = self.trunk(x)
        h = h.float()
        f = torch.cat([self.ss(h), h.mean((2, 3))], -1).reshape(B, -1)
        return torch.tanh(self.ln(self.proj(f)))


class ResNetActorCritic(nn.Module):
    def __init__(self, res=128, proprio_dim=150, act_dim=8, latent=LATENT, freeze=False, pretrained=True,
                 init_std=1.0, max_std=2.0, min_std=1e-3):
        super().__init__()
        self.log_std_max = float(torch.log(torch.tensor(max_std)))
        self.log_std_min = float(torch.log(torch.tensor(min_std)))
        self.norm = RunningNorm(proprio_dim)
        self.encoder = ResNetEncoder(res, out=latent, freeze=freeze, pretrained=pretrained)
        self.actor = Head(latent + proprio_dim, act_dim)
        self.critic = Head(latent + proprio_dim, 1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(torch.log(torch.tensor(init_std)))))
        nn.init.orthogonal_(self.actor.head.weight, 0.01); nn.init.zeros_(self.actor.head.bias)
        nn.init.orthogonal_(self.critic.head.weight, 1.0); nn.init.zeros_(self.critic.head.bias)

    def std(self):
        return self.log_std.clamp(self.log_std_min, self.log_std_max).exp()

    @torch.no_grad()
    def project_(self):
        """Keep log_std inside its bounds in place (the clamp in std() has zero gradient outside them)."""
        self.log_std.clamp_(self.log_std_min, self.log_std_max)

    def backbone_parameters(self):
        return [p for p in self.encoder.trunk.parameters() if p.requires_grad]

    def other_parameters(self):
        bb = {id(p) for p in self.encoder.trunk.parameters()}
        return [p for p in self.parameters() if id(p) not in bb and p.requires_grad]

    def encode(self, img):
        return self.encoder(img)

    def dist(self, z, prop):
        mean = self.actor(torch.cat([z, self.norm(prop)], -1))
        return torch.distributions.Normal(mean, self.std().expand_as(mean))

    def value(self, z, prop):
        return self.critic(torch.cat([z, self.norm(prop)], -1)).squeeze(-1)
