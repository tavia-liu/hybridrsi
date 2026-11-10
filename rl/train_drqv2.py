"""DrQ-v2 (facebookresearch/drqv2, unmodified, in third_party/) on the HybridRSI cell, logged to wandb.

    PYTHONPATH=.:benchmark:third_party/drqv2 MUJOCO_GL=egl python rl/train_drqv2.py task=P4 family=near

The training loop is drqv2/train.py with three changes: the environment is rl/near_env.py, eval reports the success
rate, and every metric also goes to wandb. The agent, replay buffer, augmentation and logger are DrQ-v2's own.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "rl"), str(ROOT / "third_party" / "drqv2"), str(ROOT), str(ROOT / "benchmark")]

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import wandb  # noqa: E402
from dm_env import specs  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import near_env  # noqa: E402
import utils  # noqa: E402  (drqv2)
from logger import Logger  # noqa: E402  (drqv2)
import replay_buffer  # noqa: E402  (drqv2)
from replay_buffer import ReplayBufferStorage, make_replay_loader  # noqa: E402  (drqv2)

torch.backends.cudnn.benchmark = True


def _worker_init_fn(worker_id):
    """drqv2's version seeds `random` with a numpy uint32, which Python >= 3.11 rejects; same seeding, plain ints."""
    import random
    seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(seed % 2**32)
    random.seed(seed)


replay_buffer._worker_init_fn = _worker_init_fn


def make_agent(obs_spec, state_spec, action_spec, cfg, proprio):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    if proprio:
        from omegaconf import open_dict
        with open_dict(cfg):                      # the yaml is in struct mode; these two keys exist only for the proprio agent
            cfg._target_ = "drqv2_proprio.DrQV2ProprioAgent"
            cfg.state_shape = state_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}", flush=True)
        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()
        self.agent = make_agent(self.train_env.observation_spec(), self.train_env.state_spec(), self.train_env.action_spec(),
                                self.cfg.agent, self.cfg.proprio)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._updates = 0

    def setup(self):
        cfg = self.cfg
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb)
        self.train_env = near_env.make(cfg, cfg.seed)
        self.eval_env = near_env.make(cfg, cfg.seed + 1000)
        data_specs = (self.train_env.observation_spec(),) + ((self.train_env.state_spec(),) if cfg.proprio else ()) + (
            self.train_env.action_spec(), specs.Array((1,), np.float32, "reward"), specs.Array((1,), np.float32, "discount"))
        buffer_dir = Path(cfg.buffer_dir) if cfg.buffer_dir else self.work_dir / "buffer"
        buffer_dir.parent.mkdir(parents=True, exist_ok=True)
        self.replay_storage = ReplayBufferStorage(data_specs, buffer_dir)
        if cfg.proprio:
            import drqv2_proprio
            self.replay_loader = drqv2_proprio.make_replay_loader(buffer_dir, cfg.replay_buffer_size, cfg.batch_size,
                                                                  cfg.replay_buffer_num_workers, cfg.save_snapshot, cfg.nstep,
                                                                  cfg.discount, _worker_init_fn)
        else:
            self.replay_loader = make_replay_loader(buffer_dir, cfg.replay_buffer_size, cfg.batch_size,
                                                    cfg.replay_buffer_num_workers, cfg.save_snapshot, cfg.nstep, cfg.discount)
        self._replay_iter = None
        try:
            from assembly_policy.sim.tasks import task_name
            group = task_name(cfg.task)
        except Exception:
            group = cfg.task
        name = f"drqv2{'' if cfg.proprio else '_pixelonly'}_{group}_{cfg.family}_s{cfg.seed}"
        wandb.init(project=cfg.wandb.project, group=f"{group}_{cfg.family}", name=name,
                   tags=["drqv2", "pixel+proprio" if cfg.proprio else "pixel", cfg.family, cfg.task],
                   config=OmegaConf.to_container(cfg, resolve=True),
                   dir=str(self.work_dir), mode=cfg.wandb.mode, resume="allow", id=cfg.wandb.id or None)

    def _act(self, time_step, eval_mode):
        if self.cfg.proprio:
            return self.agent.act(time_step.observation, time_step.state, self.global_step, eval_mode=eval_mode)
        return self.agent.act(time_step.observation, self.global_step, eval_mode=eval_mode)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward, successes, safeties = 0, 0, 0.0, 0, 0
        video = []
        while episode < self.cfg.num_eval_episodes:
            time_step = self.eval_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self._act(time_step, eval_mode=True)
                time_step = self.eval_env.step(action)
                if episode == 0:
                    video.append(time_step.observation[-3:])            # newest eye_in_hand frame (C,H,W)
                total_reward += time_step.reward
                step += 1
            out = self.eval_env.last_outcome
            successes += int(bool(out.get("success")))
            safeties += int(bool(out.get("safety")))
            episode += 1
        n = max(episode, 1)
        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / n)
            log("episode_length", step * self.cfg.action_repeat / n)
            log("episode", self.global_episode)
            log("step", self.global_step)
        rec = dict(frame=self.global_frame, step=self.global_step, episode=self.global_episode,
                   success_rate=successes / n, safety_rate=safeties / n, episode_reward=total_reward / n,
                   episode_length=step * self.cfg.action_repeat / n)
        payload = {f"eval/{k}": v for k, v in rec.items()}
        if video:
            import imageio.v2 as imageio
            path = self.work_dir / "eval_videos" / f"{self.global_frame}.mp4"
            path.parent.mkdir(exist_ok=True)
            frames = [np.ascontiguousarray(f.transpose(1, 2, 0)) for f in video[::2]]       # newest eye_in_hand frame
            imageio.mimwrite(path, frames, fps=6, codec="libx264", quality=6, macro_block_size=4)   # local only (user)
        wandb.log(payload, step=self.global_frame)
        print(f"[eval] frame {self.global_frame}: success {successes}/{n}  safety {safeties}/{n}", flush=True)

    def train(self):
        cfg = self.cfg
        train_until_step = utils.Until(cfg.num_train_frames, cfg.action_repeat)
        seed_until_step = utils.Until(cfg.num_seed_frames, cfg.action_repeat)
        eval_every_step = utils.Every(cfg.eval_every_frames, cfg.action_repeat)
        episode_step, episode_reward = 0, 0.0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        metrics = None
        recent = []
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                out = self.train_env.last_outcome
                recent.append((int(bool(out.get("success"))), int(bool(out.get("safety")))))
                recent = recent[-50:]
                if metrics is not None:
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, ty="train") as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)
                    wandb.log({"train/fps": episode_frame / elapsed_time, "train/episode_reward": episode_reward,
                               "train/episode_length": episode_frame, "train/episode": self.global_episode,
                               "train/buffer_size": len(self.replay_storage), "train/total_time_h": total_time / 3600,
                               "train/success_rate_50": float(np.mean([r[0] for r in recent])),
                               "train/safety_rate_50": float(np.mean([r[1] for r in recent]))}, step=self.global_frame)
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step)
                if cfg.save_snapshot:
                    self.save_snapshot()
                episode_step, episode_reward = 0, 0.0
            if eval_every_step(self.global_step):
                self.logger.log("eval_total_time", self.timer.total_time(), self.global_frame)
                self.eval()
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self._act(time_step, eval_mode=False)
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                if metrics:
                    self._updates += 1
                    if self._updates % cfg.log_every_updates == 0:
                        wandb.log({f"agent/{k}": v for k, v in metrics.items()}, step=self.global_frame)
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1
        self.save_snapshot()
        wandb.finish()

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        payload = {k: self.__dict__[k] for k in ["agent", "timer", "_global_step", "_global_episode", "_updates"]}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        with (self.work_dir / "snapshot.pt").open("rb") as f:
            payload = torch.load(f, weights_only=False)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="near", version_base=None)
def main(cfg):
    workspace = Workspace(cfg)
    if (Path.cwd() / "snapshot.pt").exists():
        print("resuming from snapshot.pt", flush=True)
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
