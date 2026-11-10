"""SAC from pixels on the HybridRSI cell, on frozen ResNet-18 features (rl/sac).

Algorithm: ManiSkill3 examples/baselines/sac/sac_rgbd.py (vectorised-env SAC: tanh-Gaussian actor, twin critics with
Polyak targets, automatic temperature, update-to-data ratio over parallel envs) adapted to this cell:
  * observations are the benchmark's two cameras (rl/ppo/pixel_env.py, EE-delta actions) + 150-d proprio; the images go
    through the frozen ImageNet ResNet-18 trunk once, at collection, after a random-shift augmentation, and the replay
    buffer stores the pooled 1536-d feature instead of pixels (rl/sac/nets.py). The trainable encoder is a 1536 -> 128
    projection shared by actor and critics, trained by the critic loss (the actor sees it detached), as in sac_rgbd;
  * RLPD-style critic: LayerNorm MLPs, an ensemble of --n-critics with the target taken as the min over --n-min-q of
    them (2 / 2 = plain SAC), and no entropy term in the critic target unless --backup-entropy (RLPD/SERL default for
    sparse rewards; the actor is still max-entropy with automatic temperature);
  * reward = the simulator's 1[success] - control cost; success and safety stops are terminal (no bootstrap),
    time-outs bootstrap; training episodes are capped at --episode-steps control steps (near starts need < 100).
Starts: a fixed bank of --family starts from the training seeds (default near_up: part upright above its fixture, hand
straight down), drawn uniformly.

  PYTHONPATH=.:benchmark:rl/ppo:rl/sac MUJOCO_GL=egl python rl/sac/train_sac.py --task P4 --run-dir runs/sac/P4_near_up
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark"), str(ROOT / "rl" / "ppo"), str(ROOT / "rl" / "sac")]

import api  # noqa: E402
from assembly_policy.sim.env import HISTORY, PROPRIO_DIM, EnvConfig, parse_scales  # noqa: E402
from assembly_policy.sim.tasks import task_name  # noqa: E402
from nets import RandomShift, ResNetEncoder  # noqa: E402  (rl/ppo/nets.py)
from pixel_env import RES, VecEnv  # noqa: E402
from sac_nets import FEAT_DIM, SACModel, pool_features  # noqa: E402
from train_resnet import load_bank  # noqa: E402

PROP = HISTORY * PROPRIO_DIM


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--family", default="near_up", help="start family of the training bank")
    ap.add_argument("--bank-size", type=int, default=640)
    ap.add_argument("--bank-seed", type=int, default=api.TRAIN_SEEDS.start)
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--episode-steps", type=int, default=150, help="training episode cap in 25 Hz control steps")
    ap.add_argument("--total-transitions", type=int, default=20_000_000)
    # SAC
    ap.add_argument("--utd", type=float, default=0.5, help="gradient steps per collected transition")
    ap.add_argument("--training-freq", type=int, default=None,
                    help="ManiSkill sac_rgbd cadence: every env walks training_freq // envs steps, then training_freq * utd "
                         "gradient steps (their default 64 with 16 envs = 4 steps, 16 updates); default: 1 step, envs * utd updates")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--learning-starts", type=int, default=5000, help="random actions until this many transitions")
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--n-critics", type=int, default=2)
    ap.add_argument("--n-min-q", type=int, default=2, help="critics in the target min (RLPD subset)")
    ap.add_argument("--no-layer-norm", action="store_true")
    ap.add_argument("--target-entropy", type=float, default=None, help="default -act_dim")
    ap.add_argument("--init-alpha", type=float, default=1.0)
    ap.add_argument("--backup-entropy", action="store_true",
                    help="add -alpha*log pi(a') to the critic target (vanilla SAC); off by default as in RLPD/SERL for sparse rewards")
    ap.add_argument("--z-dim", type=int, default=128)
    # environment / actions (as rl/ppo/train_resnet.py)
    ap.add_argument("--action-space", default="ee", choices=["ee", "joint"])
    ap.add_argument("--ee-step-m", type=float, default=0.0025)
    ap.add_argument("--ee-step-rad", type=float, default=0.0436)
    ap.add_argument("--action-scale-joints", default="0.05+0.05+0.05+0.06+0.07+0.1+0.17")
    ap.add_argument("--cost-coef", type=float, default=2e-5)
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--image", type=int, default=RES)
    ap.add_argument("--aug-pad", type=int, default=4, help="random shift before the encoder (0 = none)")
    ap.add_argument("--encoder", default="resnet", choices=["resnet", "plainconv"],
                    help="resnet = frozen ResNet-18 features in the buffer; plainconv = ManiSkill sac_rgbd CNN trained from "
                         "scratch on raw pixels stored in the buffer (cameras concatenated along channels)")
    ap.add_argument("--hidden", default="256,256", help="actor/critic MLP hidden sizes (ManiSkill: 512,256)")
    ap.add_argument("--no-prop-norm", action="store_true", help="feed raw proprio (ManiSkill has no state normalisation)")
    # infrastructure
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20, help="iterations (one iteration = one step of every env)")
    ap.add_argument("--save-every", type=int, default=500, help="iterations")
    ap.add_argument("--max-hours", type=float, default=1e9)
    return ap.parse_args(argv)


class ReplayBuffer:
    """Ring buffer on the training device: pooled features (fp16), raw proprio, action, reward, terminal, next obs."""

    def __init__(self, cap, act_dim, dev, obs_shape=(FEAT_DIM,), obs_dtype=torch.float16):
        self.cap, self.dev, self.i, self.full = cap, dev, 0, False
        self.feat = torch.zeros(cap, *obs_shape, dtype=obs_dtype, device=dev)
        self.nfeat = torch.zeros(cap, *obs_shape, dtype=obs_dtype, device=dev)
        self.prop = torch.zeros(cap, PROP, device=dev)
        self.nprop = torch.zeros(cap, PROP, device=dev)
        self.act = torch.zeros(cap, act_dim, device=dev)
        self.rew = torch.zeros(cap, device=dev)
        self.term = torch.zeros(cap, device=dev)

    def add(self, feat, prop, act, rew, term, nfeat, nprop):
        n = len(rew)
        idx = (torch.arange(n, device=self.dev) + self.i) % self.cap
        self.feat[idx] = feat; self.nfeat[idx] = nfeat; self.prop[idx] = prop; self.nprop[idx] = nprop
        self.act[idx] = act; self.rew[idx] = rew; self.term[idx] = term
        self.i = (self.i + n) % self.cap
        self.full = self.full or self.i < n

    def __len__(self):
        return self.cap if self.full else self.i

    def sample(self, b):
        idx = torch.randint(0, len(self), (b,), device=self.dev)
        return (self.feat[idx], self.prop[idx], self.act[idx], self.rew[idx], self.term[idx], self.nfeat[idx], self.nprop[idx])


class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device)
        self.run = Path(a.run_dir); self.run.mkdir(parents=True, exist_ok=True)
        np.random.seed(a.seed); torch.manual_seed(a.seed)
        self.act_dim = 7 if a.action_space == "ee" else 8
        self.bank = load_bank(self.run, a.task, a.family, a.bank_size, a.bank_seed)
        self.rng = np.random.default_rng(a.seed)
        self.plain = a.encoder == "plainconv"
        self.trunk = None if self.plain else ResNetEncoder(a.image, freeze=True).to(self.dev).eval()   # frozen ImageNet ResNet-18
        self.aug = RandomShift(a.aug_pad).to(self.dev) if a.aug_pad > 0 else None
        hidden = tuple(int(x) for x in a.hidden.split(","))
        self.model = SACModel(PROP, self.act_dim, a.z_dim, a.n_critics, not a.no_layer_norm, a.init_alpha, encoder=a.encoder,
                              image=a.image, hidden=hidden, prop_norm=not a.no_prop_norm, last_act=self.plain).to(self.dev)
        self.q_opt = torch.optim.Adam(list(self.model.encoder.parameters()) + list(self.model.critic.parameters()), lr=a.lr)
        self.pi_opt = torch.optim.Adam(self.model.actor.parameters(), lr=a.lr)
        self.alpha_opt = torch.optim.Adam([self.model.log_alpha], lr=a.lr)
        self.target_entropy = -float(self.act_dim) if a.target_entropy is None else a.target_entropy
        self.it, self.transitions, self.updates = 0, 0, 0
        self.ecfg = EnvConfig(nthread=a.nthread, cost_coef=a.cost_coef, horizon_s=a.episode_steps * 0.04,
                              action_scale_joints=parse_scales(a.action_scale_joints))
        self.steps_per_iter = max(1, a.training_freq // a.envs) if a.training_freq else 1
        self.updates_per_iter = int(a.envs * self.steps_per_iter * a.utd)
        self.args = dict(vars(a), env={k: v for k, v in dataclasses.asdict(self.ecfg).items() if k != "nthread"},
                         derived=dict(steps_per_iteration=self.steps_per_iter, transitions_per_iteration=a.envs * self.steps_per_iter,
                                      updates_per_iteration=self.updates_per_iter,
                                      episode_max_steps=a.episode_steps, action_dim=self.act_dim, feat_dim=FEAT_DIM,
                                      target_entropy=self.target_entropy, backup_entropy=a.backup_entropy,
                                      backbone="plainconv from scratch (pixels in buffer)" if self.plain else "frozen"))
        (self.run / "config.json").write_text(json.dumps(self.args, indent=1))
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        enc = (f"ManiSkill PlainConv from scratch on pixels in the buffer" if self.plain
               else f"frozen ResNet-18 -> {FEAT_DIM}-d features in the buffer")
        print(f"[sac] task {a.task} ({task_name(a.task)}), {a.family} bank of {len(self.bank)} starts, {a.envs} envs, "
              f"{a.action_space} actions ({self.act_dim}-D), episodes <= {a.episode_steps} steps, {a.image}x{a.image} x 2 cameras "
              f"-> {enc} ({a.buffer_size}), {n / 1e6:.2f}M trainable "
              f"parameters, {a.n_critics} critics (min over {a.n_min_q}), utd {a.utd} ({self.steps_per_iter} step(s) of every env, "
              f"then {self.updates_per_iter} updates), aug pad {a.aug_pad}", flush=True)
        self.wb = None
        if os.environ.get("WANDB_PROJECT"):
            import wandb
            base = f"{a.task}_{a.family}_s{a.seed}"
            suffix = self.run.name[len(base):] if self.run.name.startswith(base) else "_" + self.run.name
            name = f"sac_resnet_frozen_{a.action_space}_{task_name(a.task)}_{a.family}_s{a.seed}{suffix}"
            self.wb = wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ.get("WANDB_ENTITY"),
                                 group=f"{task_name(a.task)}_{a.family}", name=name, config=self.args, dir=str(self.run),
                                 tags=["sac", "resnet", "pixel+proprio", a.action_space, a.family, a.task],
                                 id=os.environ.get("WANDB_RUN_ID", self.run.name.replace("/", "-")), resume="allow")
            self.wb.define_metric("transitions"); self.wb.define_metric("*", step_metric="transitions")
        self._maybe_resume()
        if self.plain:
            self.buffer = ReplayBuffer(a.buffer_size, self.act_dim, self.dev, (a.image, a.image, 6), torch.uint8)
        else:
            self.buffer = ReplayBuffer(a.buffer_size, self.act_dim, self.dev)
        t0 = np.random.default_rng(a.seed + 7).integers(0, a.episode_steps, a.envs)   # desynchronised episode clocks
        self.env = VecEnv(a.task, a.envs, self.ecfg, self._provider, t0, res=a.image, action_space=a.action_space,
                          ee_step_m=a.ee_step_m, ee_step_rad=a.ee_step_rad)

    # ------------------------------------------------------------- helpers
    def _provider(self, k, finished=None, outcomes=None):
        e = self.rng.integers(0, len(self.bank), k)
        return e, self.bank[e]

    @torch.no_grad()
    def _featurize(self, o, ids=None):
        """obs dict -> (features fp16 [B, 1536], proprio fp32 [B, 150]); images are shifted once, here."""
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        img = torch.as_tensor(sel(o["img"]), device=self.dev)
        if self.aug is not None:
            B, C, H, W, _ = img.shape
            x = self.aug(img.permute(0, 1, 4, 2, 3).reshape(B * C, 3, H, W).float())
            img = x.round_().clamp_(0, 255).to(torch.uint8).reshape(B, C, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()
        if self.plain:                                                          # [B, 2, H, W, 3] -> [B, H, W, 6]
            feat = img.permute(0, 2, 3, 1, 4).reshape(img.shape[0], img.shape[2], img.shape[3], -1).contiguous()
        else:
            feat = pool_features(self.trunk.features(img)).to(torch.float16)
        prop = torch.as_tensor(sel(o["proprio"]), dtype=torch.float32, device=self.dev)
        return feat, prop

    def save(self):
        d = dict(model=self.model.state_dict(), q_opt=self.q_opt.state_dict(), pi_opt=self.pi_opt.state_dict(),
                 alpha_opt=self.alpha_opt.state_dict(), it=self.it, transitions=self.transitions, updates=self.updates, args=self.args)
        torch.save(d, self.run / "checkpoint.pt.tmp"); os.replace(self.run / "checkpoint.pt.tmp", self.run / "checkpoint.pt")

    def _maybe_resume(self):
        p = self.run / "checkpoint.pt"
        if not p.exists():
            return
        d = torch.load(p, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.q_opt.load_state_dict(d["q_opt"]); self.pi_opt.load_state_dict(d["pi_opt"]); self.alpha_opt.load_state_dict(d["alpha_opt"])
        self.it, self.transitions, self.updates = d["it"], d["transitions"], d["updates"]
        print(f"[sac] resumed at iteration {self.it} ({self.transitions} transitions); the replay buffer restarts empty", flush=True)

    # ------------------------------------------------------------- SAC update (sac_rgbd.py)
    def update(self):
        a, m = self.a, self.model
        feat, prop, act, rew, term, nfeat, nprop = self.buffer.sample(a.batch)
        prop, nprop = m.norm(prop), m.norm(nprop)
        with torch.no_grad():
            z2 = m.encoder(nfeat)
            a2, logp2, _ = m.actor.sample(z2, nprop)
            q2 = m.critic_target(z2, nprop, a2)                                   # [n_critics, B]
            if a.n_min_q < a.n_critics:
                q2 = q2[torch.randperm(a.n_critics, device=self.dev)[:a.n_min_q]]
            v2 = q2.min(0).values - (m.alpha * logp2 if a.backup_entropy else 0.0)
            target = rew + a.gamma * (1.0 - term) * v2
        z = m.encoder(feat)
        q = m.critic(z, prop, act)
        qf_loss = ((q - target.unsqueeze(0)) ** 2).mean()
        self.q_opt.zero_grad(set_to_none=True); qf_loss.backward(); self.q_opt.step()
        zd = z.detach()
        pi, logp, _ = m.actor.sample(zd, prop)
        q_pi = m.critic(zd, prop, pi).min(0).values
        pi_loss = (m.alpha.detach() * logp - q_pi).mean()
        self.pi_opt.zero_grad(set_to_none=True); pi_loss.backward(); self.pi_opt.step()
        alpha_loss = -(m.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True); alpha_loss.backward(); self.alpha_opt.step()
        m.soft_update(a.tau)
        self.updates += 1
        return dict(qf=qf_loss.item(), pi=pi_loss.item(), alpha=m.alpha.item(), entropy=-logp.mean().item(), q=q.mean().item())

    # ------------------------------------------------------------- loop
    def train(self):
        a, N = self.a, self.a.envs
        log_f = open(self.run / "log.jsonl", "a")
        t_start = t_log = time.time()
        feat, prop = self._featurize(self.env.observe())
        st = dict(done=0, success=0, safety=0, ep_len=[], sim=0.0, losses=[])
        trans_log = self.transitions
        n_up = self.updates_per_iter
        while self.transitions < a.total_transitions:
          for _step in range(self.steps_per_iter):
            with torch.no_grad():
                if not a.no_prop_norm:
                    self.model.norm.update(prop)
                if self.transitions < a.learning_starts:
                    act = torch.rand(N, self.act_dim, device=self.dev) * 2 - 1
                else:
                    act = self.model.act(feat, prop)
            t1 = time.perf_counter()
            r, done, info, final_obs, finished = self.env.step(act.cpu().numpy())
            st["sim"] += time.perf_counter() - t1
            nfeat, nprop = self._featurize(self.env.observe())                   # post-reset obs of done envs
            ids = np.nonzero(done)[0]
            if len(ids):
                ff, pp = self._featurize(final_obs)                               # the real next obs of the done envs
                nf, npr = nfeat.clone(), nprop.clone(); nf[ids] = ff; npr[ids] = pp
            else:
                nf, npr = nfeat, nprop
            term = torch.as_tensor(done & ~info["timeout"], dtype=torch.float32, device=self.dev)
            self.buffer.add(feat, prop, act, torch.as_tensor(r, dtype=torch.float32, device=self.dev), term, nf, npr)
            feat, prop = nfeat, nprop
            self.transitions += N
            if len(ids):
                st["done"] += len(ids); st["success"] += int(info["success"][ids].sum()); st["safety"] += int(info["safety"][ids].sum())
                st["ep_len"] += info["ep_len"][ids].tolist()
          self.it += 1
          if True:
            if self.transitions >= a.learning_starts and len(self.buffer) >= a.batch:
                self.model.train()
                for _ in range(n_up):
                    st["losses"].append(self.update())
                self.model.eval()
            if self.it % a.log_every == 0:
                n_ep = max(st["done"], 1); L = st["losses"]
                rec = dict(it=self.it, transitions=self.transitions, updates=self.updates,
                           fps=(self.transitions - trans_log) / (time.time() - t_log), t_sim=st["sim"],
                           episodes=st["done"], success=st["success"] / n_ep, safety=st["safety"] / n_ep,
                           ep_len=float(np.mean(st["ep_len"])) if st["ep_len"] else 0.0, buffer=len(self.buffer),
                           **{k: float(np.mean([x[k] for x in L])) for k in (L[0].keys() if L else [])})
                log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                if self.wb is not None:
                    self.wb.log(rec, step=self.it)
                print(f"it {self.it} T={self.transitions / 1e6:.2f}M fps={rec['fps']:.0f} success={rec['success']:.3f} "
                      f"safety={rec['safety']:.3f} len={rec['ep_len']:.0f} eps={st['done']} "
                      + (f"qf={rec['qf']:.4f} q={rec['q']:.3f} alpha={rec['alpha']:.3f} H={rec['entropy']:.2f} " if L else "")
                      + f"sim={st['sim']:.1f}s", flush=True)
                st = dict(done=0, success=0, safety=0, ep_len=[], sim=0.0, losses=[]); t_log = time.time(); trans_log = self.transitions
            if self.it % a.save_every == 0:
                self.save()
            if (time.time() - t_start) / 3600 > a.max_hours:
                self.save(); break
        self.save()
        if self.wb is not None:
            self.wb.finish()


def main():
    Trainer(parse()).train()


if __name__ == "__main__":
    main()
