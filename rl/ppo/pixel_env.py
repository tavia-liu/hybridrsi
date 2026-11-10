"""The cell's batched simulator with the benchmark's two cameras as the observation (PPO training only).

PixelEnv = BatchedAssemblyEnv + `img` = uint8 [N, 2, RES, RES, 3]: the same agentview / eye_in_hand views that
api.camera() hands a policy (perception.Cameras: part orange, fixture blue, everything else grey; no masks, no poses),
rendered at RES x RES after every control step. The simulator's part pose stays in the obs dict for logging only; the
trainer never feeds it to the actor or the critic.

Action space (action_space=):
  "ee"    7 numbers in [-1, 1]: TCP translation (x, y, z, up to ee_step_m per control step), TCP rotation (rotation
          vector in the world frame, up to ee_step_rad per control step) and the gripper (> 0.5 open, < -0.5 close,
          else keep). The same action as rl/near_env.py (DrQ-v2): the TCP displacement is turned into joint increments
          with one damped-least-squares step on the TCP Jacobian (the robot's own kinematics), clipped to the joint
          action scale, and handed to the joint PD. Default.
  "joint" 8 numbers: the simulator's native 7 relative joint-position increments + gripper (AlphaNIST).

VecEnv = auto-resetting PixelEnv with a learner-side reset provider (assembly-nist rl/vecenv.LocalVecEnv).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark")]

from assembly_policy.sim.env import BatchedAssemblyEnv, EnvConfig  # noqa: E402
from assembly_policy.sim.scene import build_xml  # noqa: E402
from perception import CAMERA_NAMES, Cameras  # noqa: E402

RES = int(os.environ.get("HYBRIDRSI_PPO_IMAGE", "128"))
OBS_KEYS = ("proprio", "img")


class PixelEnv(BatchedAssemblyEnv):
    def __init__(self, task_name, n_envs, ecfg: EnvConfig | None = None, res=RES, action_space="ee",
                 ee_step_m=0.0025, ee_step_rad=0.0436, ik_damping=1e-3):
        super().__init__([task_name], np.zeros(n_envs, dtype=np.int64), ecfg)
        assert action_space in ("ee", "joint"), action_space
        self.action_space, self.ee_step_m, self.ee_step_rad, self.ik_damping = action_space, float(ee_step_m), float(ee_step_rad), float(ik_damping)
        self.action_dim = 7 if action_space == "ee" else 8
        self.cams = Cameras(build_xml(task_name, visual=True), w=res, h=res)
        self.res = res
        self.img = np.zeros((self.N, len(CAMERA_NAMES), res, res, 3), np.uint8)
        self.stats = dict(render=0.0, steps=0, ik=0.0)
        self._ik_d = mujoco.MjData(self.models[0].model)
        self._jp, self._jr = np.zeros((3, self.models[0].model.nv)), np.zeros((3, self.models[0].model.nv))
        self._scale = np.asarray(self.ecfg.action_scale_joints, float) if len(self.ecfg.action_scale_joints) else np.full(7, self.ecfg.action_scale)

    def ee_to_joint(self, a):
        """[N, 7] EE-delta actions -> [N, 8] joint actions for BatchedAssemblyEnv.step (one DLS step per env)."""
        a = np.clip(np.asarray(a, np.float64).reshape(self.N, 7), -1.0, 1.0)
        t0 = time.perf_counter()
        m, d, idx = self.models[0].model, self._ik_d, self.idx
        q = self.state[:, self.sq]
        ja = np.zeros((self.N, 8))
        eye = self.ik_damping * np.eye(6)
        for i in range(self.N):
            d.qpos[idx.arm_qpos] = q[i]
            mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
            mujoco.mj_jacSite(m, d, self._jp, self._jr, idx.tcp_site)
            J = np.vstack([self._jp[:, idx.arm_dof], self._jr[:, idx.arm_dof]])
            dx = np.r_[a[i, :3] * self.ee_step_m, a[i, 3:6] * self.ee_step_rad]
            dq = J.T @ np.linalg.solve(J @ J.T + eye, dx)
            ja[i, :7] = np.clip(dq / self._scale, -1.0, 1.0)
        ja[:, 7] = a[:, 6]
        self.stats["ik"] += time.perf_counter() - t0
        return ja

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
        if self.action_space == "ee":
            action = self.ee_to_joint(action)
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

    def __init__(self, task_name, n_envs, ecfg: EnvConfig, provider, t0=None, res=RES, **env_kw):
        self.provider = provider
        self.env = PixelEnv(task_name, n_envs, ecfg, res, **env_kw)
        self.action_dim = self.env.action_dim
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
