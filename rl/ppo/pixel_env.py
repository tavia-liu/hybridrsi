"""The cell's batched simulator with the benchmark's two cameras as the observation (PPO training only).

PixelEnv = BatchedAssemblyEnv + `img` = uint8 [N, 2, RES, RES, 3]: the same agentview / eye_in_hand views that
api.camera() hands a policy (perception.Cameras: part orange, fixture blue, everything else grey; no masks, no poses),
rendered at RES x RES after every control step. The simulator's part pose stays in the obs dict for logging only; the
trainer never feeds it to the actor or the critic.

VecEnv = auto-resetting PixelEnv with a learner-side reset provider (assembly-nist rl/vecenv.LocalVecEnv).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark")]

from assembly_policy.sim.env import BatchedAssemblyEnv, EnvConfig  # noqa: E402
from assembly_policy.sim.scene import build_xml  # noqa: E402
from perception import CAMERA_NAMES, Cameras  # noqa: E402

RES = int(os.environ.get("HYBRIDRSI_PPO_IMAGE", "128"))
OBS_KEYS = ("proprio", "img")


class PixelEnv(BatchedAssemblyEnv):
    def __init__(self, task_name, n_envs, ecfg: EnvConfig | None = None, res=RES):
        super().__init__([task_name], np.zeros(n_envs, dtype=np.int64), ecfg)
        self.cams = Cameras(build_xml(task_name, visual=True), w=res, h=res)
        self.res = res
        self.img = np.zeros((self.N, len(CAMERA_NAMES), res, res, 3), np.uint8)
        self.stats = dict(render=0.0, steps=0)

    def _refresh(self, ids):
        ids = np.atleast_1d(ids)
        if len(ids) == 0:
            return
        t0 = time.perf_counter()
        for i in ids:
            views = self.cams.capture(self.state[i])
            for c, name in enumerate(CAMERA_NAMES):
                self.img[i, c] = views[name]["rgb"]
        self.stats["render"] += time.perf_counter() - t0; self.stats["steps"] += len(ids)

    def set_states(self, env_ids, states, task_ids=None):
        super().set_states(env_ids, states, task_ids)
        self._refresh(env_ids)

    def step(self, action):
        out = super().step(action)
        self._refresh(np.arange(self.N))
        return out

    def observe(self):
        o = super().observe()
        o["img"] = self.img.copy()
        return o


class VecEnv:
    """Auto-resetting PixelEnv. provider(k, finished_entries=None, outcomes=None) -> (entries, states): it records
    the finished episodes' outcomes in the sampler, then draws k bank entries. Each env holds a pre-sampled `next`
    entry and resets to it the moment its episode ends; step() returns the pre-reset obs rows of the done envs."""

    def __init__(self, task_name, n_envs, ecfg: EnvConfig, provider, t0=None, res=RES):
        self.provider = provider
        self.env = PixelEnv(task_name, n_envs, ecfg, res)
        self.N, self.horizon = n_envs, self.env.horizon
        e, s = provider(n_envs)
        self.entry = e
        self.env.set_states(np.arange(n_envs), s)
        self.next_entry, self.next_state = provider(n_envs)
        if t0 is not None:
            self.env.t[:] = t0

    def observe(self):
        return self.env.observe()

    def step(self, action):
        ep_len = self.env.t + 1
        r, done, info = self.env.step(action)
        ids = np.nonzero(done)[0]
        final_obs = None
        finished = self.entry[ids].copy()
        if len(ids):
            o = self.env.observe()
            final_obs = {k: o[k][ids] for k in OBS_KEYS}
            self.env.set_states(ids, self.next_state[ids])
            self.entry[ids] = self.next_entry[ids]
            e, s = self.provider(len(ids), finished, info["success"][ids])
            self.next_entry[ids], self.next_state[ids] = e, s
        info["ep_len"] = ep_len
        return r, done, info, final_obs, finished

    def close(self):
        self.env.close()
