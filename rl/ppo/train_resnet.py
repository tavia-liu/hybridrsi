"""PPO from pixels on the HybridRSI cell: symmetric actor-critic on a fine-tuned ImageNet ResNet-18 (rl/ppo/nets.py).

Port of assembly-nist cap/train_resnet.py (+ cap/train_pixel.py + assembly_policy/train.py, AlphaNIST Algorithm 1) to
this repository: same simulator (assembly_policy.sim.env: 8-D joint-increment + gripper actions at 25 Hz, reward
1[d_asm < 1 mm] - control cost, termination on success / joint-velocity limit / 15 s), same PPO (32-step rollouts
that slice across episodes, GAE, clipped surrogate + clipped value loss, adaptive-KL learning rate, 5 epochs x 4
minibatches, action-rate smoothness term from iteration 5000), same 150-d proprio history; the images are the
benchmark's two cameras (rl/ppo/pixel_env.py) instead of assembly-nist's global/wrist pair. No oracle part pose and no
point clouds anywhere. Starts: a fixed bank of `family` starts drawn from the training seeds (api.sample_starts), with
a uniform or the paper's beta (SGS) reset sampler. Random-shift augmentation is applied once when a frame is
collected and the shifted frame is what is stored (assembly-nist gated this on model.training, which is False during
collection, so its ResNet run was effectively un-augmented).

  PYTHONPATH=.:benchmark:rl/ppo MUJOCO_GL=egl python rl/ppo/train_resnet.py --task P4 --run-dir runs/ppo/P4_near --envs 128
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
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark"), str(ROOT / "rl" / "ppo")]

import api  # noqa: E402
from assembly_policy.sim.env import HISTORY, PROPRIO_DIM, EnvConfig, parse_scales  # noqa: E402
from assembly_policy.sim.tasks import task_name  # noqa: E402
from nets import RandomShift, ResNetActorCritic  # noqa: E402
from pixel_env import RES, VecEnv  # noqa: E402
from sampler import ResetSampler  # noqa: E402

OBS = ("prop", "img")


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--task", default="P4")
    ap.add_argument("--family", default="near", help="start family of the training bank (near = part held above its fixture)")
    ap.add_argument("--bank-size", type=int, default=640, help="start states in the bank")
    ap.add_argument("--bank-seed", type=int, default=api.TRAIN_SEEDS.start, help="seed of the bank (training seeds only)")
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--iters", type=int, default=30000)
    # PPO
    ap.add_argument("--steps", type=int, default=32, help="rollout steps per environment and iteration")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4, help="initial learning rate (adaptive-KL schedule)")
    ap.add_argument("--desired-kl", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.005)
    ap.add_argument("--vf-coef", type=float, default=1.0)
    ap.add_argument("--max-grad", type=float, default=1.0)
    ap.add_argument("--init-std", type=float, default=1.0)
    ap.add_argument("--min-std", type=float, default=1e-3)
    ap.add_argument("--max-std", type=float, default=2.0)
    ap.add_argument("--micro-batch", type=int, default=65536, help="samples per forward/backward chunk (GPU memory)")
    # reset sampler
    ap.add_argument("--sampler", default="uniform", choices=["beta", "uniform"], help="beta = AlphaNIST SGS")
    ap.add_argument("--eps-q", type=float, default=1e-4, help="beta sampler floor")
    ap.add_argument("--history", type=int, default=50, help="outcomes kept per bank entry (L)")
    # environment
    ap.add_argument("--action-scale-joints", default="0.05+0.05+0.05+0.06+0.07+0.1+0.17",
                    help="rad per unit action for each arm joint ('+'-separated)")
    ap.add_argument("--cost-coef", type=float, default=2e-5, help="control cost c (actions, joint speeds, torques)")
    ap.add_argument("--action-rate-coef", type=float, default=1e-4,
                    help="smoothness: reward -= coef * sum_j (a_t,j - a_t-1,j)^2 over the arm actions")
    ap.add_argument("--action-rate-start", type=int, default=5000, help="iteration from which the action-rate term is on")
    ap.add_argument("--nthread", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    # encoder
    ap.add_argument("--image", type=int, default=RES, help="camera resolution fed to the encoder (square)")
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)
    ap.add_argument("--freeze-backbone", action="store_true")
    ap.add_argument("--aug-pad", type=int, default=4, help="random-shift pad in pixels (0 = no augmentation)")
    # infrastructure
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--keep-every", type=int, default=100, help="also keep model_<it>.pt every this many iterations")
    ap.add_argument("--max-hours", type=float, default=1e9, help="save and exit after this wall time")
    return ap.parse_args(argv)


def load_bank(run: Path, task, family, size, seed):
    """Fixed bank of start states for this run (made once from the training seeds, kept with the run)."""
    p = run / "bank.npz"
    if p.exists():
        z = np.load(p)
        if len(z["states"]) == size and str(z["family"]) == family and int(z["seed"]) == seed:
            return z["states"]
    assert seed >= api.TRAIN_SEEDS.start, "the bank must come from training seeds"
    states = api.sample_starts(task, size, seed=seed, family=family)
    np.savez(p, states=states, family=family, seed=seed, task=task)
    return states


class _ScaledGroup(dict):
    """Optimizer param group whose lr is scaled by lr_mult whenever the trainer sets it (adaptive-KL schedule)."""

    def __setitem__(self, k, v):
        if k == "lr":
            v = v * self.get("lr_mult", 1.0)
        super().__setitem__(k, v)


class Trainer:
    def __init__(self, a):
        self.a = a
        self.dev = torch.device(a.device)
        self.run = Path(a.run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        np.random.seed(a.seed); torch.manual_seed(a.seed)
        self.bank = load_bank(self.run, a.task, a.family, a.bank_size, a.bank_seed)
        self.sampler = ResetSampler(len(self.bank), a.sampler, a.eps_q, a.history, seed=a.seed)
        self.model = ResNetActorCritic(res=a.image, freeze=a.freeze_backbone, init_std=a.init_std, max_std=a.max_std,
                                       min_std=a.min_std).to(self.dev)
        groups = [dict(params=self.model.other_parameters(), lr=a.lr, lr_mult=1.0)]
        if not a.freeze_backbone:
            groups.append(dict(params=self.model.backbone_parameters(), lr=a.lr * a.backbone_lr_mult, lr_mult=a.backbone_lr_mult))
        self.opt = torch.optim.Adam(groups)
        self.lr, self.it, self.transitions = a.lr, 0, 0
        self._wrap_groups()
        self.aug = RandomShift(a.aug_pad).to(self.dev) if a.aug_pad > 0 else None
        self.ecfg = EnvConfig(nthread=a.nthread, cost_coef=a.cost_coef, action_scale_joints=parse_scales(a.action_scale_joints))
        self.args = dict(vars(a), env={k: v for k, v in dataclasses.asdict(self.ecfg).items() if k != "nthread"})
        n = sum(p.numel() for p in self.model.parameters())
        nb = sum(p.numel() for p in self.model.encoder.trunk.parameters())
        self.args["derived"] = dict(rollout_transitions=a.envs * a.steps, minibatch_size=a.envs * a.steps // a.minibatches,
                                    updates_per_iteration=a.epochs * a.minibatches,
                                    total_transitions_planned=a.iters * a.envs * a.steps,
                                    episode_max_steps=int(round(self.ecfg.horizon_s / self.ecfg.control_dt)))
        (self.run / "config.json").write_text(json.dumps(self.args, indent=1))
        print(f"[ppo] task {a.task} ({task_name(a.task)}), {a.family} bank of {len(self.bank)} starts (seed {a.bank_seed}), "
              f"{a.envs} envs, {a.image}x{a.image} x 2 cameras, {n / 1e6:.2f}M parameters, backbone {nb / 1e6:.2f}M "
              f"({'frozen' if a.freeze_backbone else f'lr x{a.backbone_lr_mult}'}), aug pad {a.aug_pad}", flush=True)
        self.wb = None
        if os.environ.get("WANDB_PROJECT"):
            import wandb
            name = f"ppo_resnet_{task_name(a.task)}_{a.family}_s{a.seed}"
            self.wb = wandb.init(project=os.environ["WANDB_PROJECT"], entity=os.environ.get("WANDB_ENTITY"),
                                 group=f"{task_name(a.task)}_{a.family}", name=name, config=self.args, dir=str(self.run),
                                 tags=["ppo", "resnet", "pixel+proprio", a.family, a.task],
                                 id=os.environ.get("WANDB_RUN_ID", self.run.name.replace("/", "-")), resume="allow")
            self.wb.define_metric("transitions")                     # default x-axis: environment steps, not iterations
            self.wb.define_metric("*", step_metric="transitions")
        self._maybe_resume()
        horizon = int(round(self.ecfg.horizon_s / self.ecfg.control_dt))
        t0 = np.random.default_rng(a.seed + 7).integers(0, horizon, a.envs)   # desynchronized episode clocks
        self.env = VecEnv(a.task, a.envs, self.ecfg, self._provider, t0, res=a.image)

    # ------------------------------------------------------------- helpers
    def _wrap_groups(self):
        self.opt.param_groups = [g if isinstance(g, _ScaledGroup) else _ScaledGroup(g) for g in self.opt.param_groups]
        self._set_lr()

    def _set_lr(self):
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def _provider(self, k, finished=None, outcomes=None):
        """Record finished episodes in the sampler, then draw k bank entries."""
        if finished is not None and len(finished):
            self.sampler.update(finished, outcomes)
        e = self.sampler.sample(k)
        return e, self.bank[e]

    def _obs(self, o, ids=None):
        sel = (lambda x: x) if ids is None else (lambda x: x[ids])
        prop = torch.as_tensor(sel(o["proprio"]), dtype=torch.float32, device=self.dev)
        img = torch.as_tensor(sel(o["img"]), device=self.dev)                 # uint8 [B, 2, H, W, 3]
        if self.aug is not None:                                             # shifted once, at collection
            B, C, H, W, _ = img.shape
            x = img.permute(0, 1, 4, 2, 3).reshape(B * C, 3, H, W).float()
            with torch.no_grad():
                x = self.aug(x)
            img = x.round_().clamp_(0, 255).to(torch.uint8).reshape(B, C, 3, H, W).permute(0, 1, 3, 4, 2).contiguous()
        return dict(prop=prop, img=img)

    # ------------------------------------------------------------- checkpoints
    def save(self, keep=False):
        torch.save(self.sampler.state_dict(), self.run / "sampler.pt.tmp"); os.replace(self.run / "sampler.pt.tmp", self.run / "sampler.pt")
        d = dict(model=self.model.state_dict(), opt=self.opt.state_dict(), lr=self.lr, it=self.it,
                 transitions=self.transitions, args=self.args)
        torch.save(d, self.run / "checkpoint.pt.tmp"); os.replace(self.run / "checkpoint.pt.tmp", self.run / "checkpoint.pt")
        if keep:
            torch.save(dict(model=self.model.state_dict(), it=self.it, transitions=self.transitions, args=self.args),
                       self.run / f"model_{self.it:06d}.pt")

    def _maybe_resume(self):
        p = self.run / "checkpoint.pt"
        if not p.exists():
            return
        d = torch.load(p, map_location=self.dev, weights_only=False)
        self.model.load_state_dict(d["model"])
        self.opt.load_state_dict(d["opt"])
        self.model.project_()
        self.lr, self.it, self.transitions = d["lr"], d["it"], d["transitions"]
        self._wrap_groups()                                                  # load_state_dict rebuilds plain dicts
        sp = self.run / "sampler.pt"
        if sp.exists():
            st = torch.load(sp, weights_only=False)
            if len(st["cnt"]) == self.sampler.M:
                self.sampler.load_state_dict(st)
        print(f"[ppo] resumed at iteration {self.it}", flush=True)

    # ------------------------------------------------------------- rollout
    def collect(self):
        a, N, T = self.a, self.a.envs, self.a.steps
        rate = a.action_rate_coef if self.it >= a.action_rate_start else 0.0
        buf = {k: [] for k in OBS}
        acts, logps, vals, rews, dones, mus = [], [], [], [], [], []
        st = dict(done=0, success=0, safety=0, ep_len=[], sim=0.0)
        self.model.eval()
        for t in range(T):
            ob = self._obs(self.env.observe())
            with torch.no_grad():
                self.model.norm.update(ob["prop"])
                z = self.model.encode(ob["img"])
                dist = self.model.dist(z, ob["prop"])
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                v = self.model.value(z, ob["prop"])
                # smoothness term; the previous arm action is the newest proprio frame's columns 22..28
                prev = ob["prop"].view(N, HISTORY, PROPRIO_DIM)[:, -1, 22:29]
                rate_pen = (rate * ((act[:, :7] - prev) ** 2).sum(1)).cpu().numpy()
            t1 = time.perf_counter()
            r, done, info, final_obs, finished = self.env.step(act.cpu().numpy())
            st["sim"] += time.perf_counter() - t1
            r = r.astype(np.float32) - rate_pen.astype(np.float32)
            ids = np.nonzero(done)[0]
            if len(ids) and info["timeout"][ids].any():   # time-outs bootstrap from the final state's value
                sel = np.nonzero(info["timeout"][ids])[0]
                obf = self._obs(final_obs, sel)
                with torch.no_grad():
                    r[ids[sel]] += a.gamma * self.model.value(self.model.encode(obf["img"]), obf["prop"]).cpu().numpy()
            for k in OBS:
                buf[k].append(ob[k])
            acts.append(act); logps.append(logp); vals.append(v); mus.append(dist.mean)
            rews.append(torch.as_tensor(r, device=self.dev))
            dones.append(torch.as_tensor(done, device=self.dev))
            if len(ids):
                st["done"] += len(ids); st["success"] += int(info["success"][ids].sum()); st["safety"] += int(info["safety"][ids].sum())
                st["ep_len"] += info["ep_len"][ids].tolist()
        ob = self._obs(self.env.observe())
        with torch.no_grad():
            last_v = self.model.value(self.model.encode(ob["img"]), ob["prop"])
        batch = {k: torch.stack(v) for k, v in buf.items()}
        batch.update(act=torch.stack(acts), logp=torch.stack(logps), val=torch.stack(vals), mu=torch.stack(mus),
                     rew=torch.stack(rews), done=torch.stack(dones).float())
        adv = torch.zeros_like(batch["rew"]); gae = torch.zeros(N, device=self.dev)
        for t in reversed(range(T)):                         # GAE
            nv = last_v if t == T - 1 else batch["val"][t + 1]
            nd = 1.0 - batch["done"][t]
            delta = batch["rew"][t] + a.gamma * nv * nd - batch["val"][t]
            gae = delta + a.gamma * a.lam * nd * gae
            adv[t] = gae
        batch["ret"], batch["adv"] = adv + batch["val"], adv
        self.transitions += N * T
        return batch, st

    # ------------------------------------------------------------- PPO update
    def update(self, batch):
        a = self.a
        self.model.train()
        B = batch["rew"].numel()
        flat = {k: v.reshape(B, *v.shape[2:]) for k, v in batch.items()}
        flat["adv"] = (flat["adv"] - flat["adv"].mean()) / (flat["adv"].std() + 1e-8)
        mb = B // a.minibatches
        sigma_old = self.model.std().detach().clone()
        logs = dict(pi=0.0, vf=0.0, ent=0.0, kl=0.0)
        for _ in range(a.epochs):
            perm = torch.randperm(B, device=self.dev)
            for i in range(a.minibatches):
                idx = perm[i * mb:(i + 1) * mb]
                self.opt.zero_grad(set_to_none=True)
                kl = torch.zeros((), device=self.dev)
                for c0 in range(0, len(idx), a.micro_batch):     # gradient accumulation over micro-batches
                    cidx = idx[c0:c0 + a.micro_batch]
                    w = len(cidx) / len(idx)
                    ob = {k: flat[k][cidx] for k in OBS}
                    z = self.model.encode(ob["img"])
                    d = self.model.dist(z, ob["prop"])
                    logp = d.log_prob(flat["act"][cidx]).sum(-1)
                    ent = d.entropy().sum(-1).mean()
                    v = self.model.value(z, ob["prop"])
                    with torch.no_grad():   # analytic KL(old || new) between diagonal Gaussians
                        mu_o, mu_n, sd_n = flat["mu"][cidx], d.mean, d.stddev
                        kl += w * (torch.log(sd_n / sigma_old) + (sigma_old ** 2 + (mu_o - mu_n) ** 2) / (2 * sd_n ** 2) - 0.5).sum(-1).mean()
                    ratio = torch.exp(logp - flat["logp"][cidx])
                    A = flat["adv"][cidx]
                    pi_loss = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                    v_old, ret = flat["val"][cidx], flat["ret"][cidx]
                    v_clip = v_old + (v - v_old).clamp(-a.clip, a.clip)
                    vf_loss = torch.max((v - ret) ** 2, (v_clip - ret) ** 2).mean()
                    (w * (pi_loss + a.vf_coef * vf_loss - a.ent * ent)).backward()
                    logs["pi"] += w * pi_loss.item(); logs["vf"] += w * vf_loss.item(); logs["ent"] += w * ent.item()
                if kl.item() > 2.0 * a.desired_kl:                # adaptive-KL learning rate (rsl_rl schedule)
                    self.lr = max(1e-5, self.lr / 1.5)
                elif 0.0 < kl.item() < a.desired_kl / 2.0:
                    self.lr = min(1e-3, self.lr * 1.5)
                self._set_lr()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), a.max_grad)
                self.opt.step()
                self.model.project_()
                logs["kl"] += kl.item()
        n = a.epochs * a.minibatches
        return {k: v / n for k, v in logs.items()}

    def train(self):
        a = self.a
        log_f = open(self.run / "log.jsonl", "a")
        t_start = time.time()
        while self.it < a.iters:
            t0 = time.time()
            batch, st = self.collect()
            t1 = time.time()
            losses = self.update(batch)
            self.it += 1
            n_ep = max(st["done"], 1)
            rec = dict(it=self.it, transitions=self.transitions, fps=a.envs * a.steps / (time.time() - t0),
                       t_sim=st["sim"], t_render=self.env.env.stats["render"], t_update=time.time() - t1, lr=self.lr,
                       std=float(self.model.std().mean()), episodes=st["done"], success=st["success"] / n_ep,
                       safety=st["safety"] / n_ep, ep_len=float(np.mean(st["ep_len"])) if st["ep_len"] else 0.0,
                       visited=float((self.sampler.cnt > 0).mean()), **losses)
            self.env.env.stats["render"] = 0.0
            log_f.write(json.dumps(rec) + "\n"); log_f.flush()
            if self.wb is not None:
                self.wb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=self.it)
            if self.it % 10 == 0 or self.it <= 3:
                print(f"it {self.it} T={self.transitions / 1e6:.1f}M fps={rec['fps']:.0f} train_success={rec['success']:.3f} "
                      f"safety={rec['safety']:.3f} len={rec['ep_len']:.0f} kl={losses['kl']:.4f} lr={self.lr:.1e} "
                      f"std={rec['std']:.3f} sim={st['sim']:.1f}s render={rec['t_render']:.1f}s update={rec['t_update']:.1f}s", flush=True)
            if self.it % a.save_every == 0 or self.it == a.iters:
                self.save(keep=self.it % a.keep_every == 0 or self.it == a.iters)
            if (time.time() - t_start) / 3600 > a.max_hours:
                self.save()
                break
        if self.wb is not None:
            self.wb.finish()


def main():
    Trainer(parse()).train()


if __name__ == "__main__":
    main()
