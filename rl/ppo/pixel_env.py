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

import multiprocessing as mp
from multiprocessing import shared_memory

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark")]
os.environ.setdefault("CAP_CAM_RES", "1024x1024")      # offscreen buffer large enough for renders up to 1024 px (must precede perception)
os.environ.setdefault("CAP_CAM_RES", "1024x1024")      # offscreen buffer large enough for up to 1024 px renders (must precede perception)

from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv, EnvConfig  # noqa: E402
from assembly_policy.sim.scene import build_xml  # noqa: E402
from perception import CAMERA_NAMES, FIXTURE_RGBA, OTHER_RGBA, PART_RGBA, Cameras, on_render_thread  # noqa: E402

RES = int(os.environ.get("HYBRIDRSI_PPO_IMAGE", "128"))
RENDER_WORKERS = int(os.environ.get("HYBRIDRSI_RENDER_WORKERS", "8"))   # 0/1 = render in this process
OBS_KEYS = ("proprio", "img")


def _rgb_views(cams, state):
    """RGB of both cameras at `state` (perception.Cameras.capture without the depth pass; same colour marking)."""
    cams.m.geom_rgba[cams.other_geoms] = OTHER_RGBA
    cams.m.geom_rgba[cams.part_geoms] = PART_RGBA
    cams.m.geom_rgba[cams.fixture_geoms] = FIXTURE_RGBA
    mujoco.mj_setState(cams.m, cams.d, state, STATE_SPEC)
    mujoco.mj_forward(cams.m, cams.d)
    out = {}
    for name, cam_id in cams.cams.items():
        r = cams.rs[name]
        r.update_scene(cams.d, camera=cam_id)
        out[name] = r.render().copy()
    return out


def _render_worker(task_name, res, state_shm, img_shm, n, nstate, conn):
    """One render process: its own EGL context and Cameras; renders the env ids it is sent from the shared state array
    into the shared image array. Rendering is GPU-call bound (~0.9 ms per 128x128 image), so processes scale it."""
    import traceback
    os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        cams = Cameras(build_xml(task_name, visual=True), w=res, h=res)
        st_shm = shared_memory.SharedMemory(name=state_shm); im_shm = shared_memory.SharedMemory(name=img_shm)
        states = np.ndarray((n, nstate), np.float64, buffer=st_shm.buf)
        imgs = np.ndarray((n, len(CAMERA_NAMES), res, res, 3), np.uint8, buffer=im_shm.buf)
        conn.send("ready")
        while True:
            ids = conn.recv()
            if ids is None:
                break
            for i in ids:
                v = on_render_thread(_rgb_views, cams, states[i])     # the GL context lives on the render thread
                for c, name in enumerate(CAMERA_NAMES):
                    imgs[i, c] = v[name]
            conn.send(len(ids))
    except Exception:
        print("[render worker] " + traceback.format_exc().replace("\n", " | "), file=sys.stderr, flush=True)
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            st_shm.close(); im_shm.close()
        except Exception:
            pass


class RenderPool:
    """Renders env images in `workers` processes; the caller writes states to `self.states` and reads `self.imgs`."""

    def __init__(self, task_name, res, n, nstate, workers):
        self.n, self.workers = n, workers
        self._st = shared_memory.SharedMemory(create=True, size=n * nstate * 8)
        self._im = shared_memory.SharedMemory(create=True, size=n * len(CAMERA_NAMES) * res * res * 3)
        self.states = np.ndarray((n, nstate), np.float64, buffer=self._st.buf)
        self.imgs = np.ndarray((n, len(CAMERA_NAMES), res, res, 3), np.uint8, buffer=self._im.buf)
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_render_worker, args=(task_name, res, self._st.name, self._im.name, n, nstate, b), daemon=True)
            p.start(); self.conns.append(a); self.procs.append(p)
        for a in self.conns:                      # workers report once their GL context and cameras exist
            msg = a.recv()
            if msg != "ready":
                raise RuntimeError(f"render worker failed to start: {msg}")

    def render(self, ids):
        ids = np.asarray(ids)
        parts = [ids[w::self.workers] for w in range(self.workers)]
        for c, part in zip(self.conns, parts):
            if len(part):
                c.send(part.tolist())
        for c, part in zip(self.conns, parts):
            if len(part):
                msg = c.recv()
                if msg != len(part):
                    raise RuntimeError(f"render worker failed: {msg}")

    def close(self):
        for c in self.conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
        self._st.close(); self._st.unlink(); self._im.close(); self._im.unlink()


class PixelEnv(BatchedAssemblyEnv):
    def __init__(self, task_name, n_envs, ecfg: EnvConfig | None = None, res=RES, action_space="ee",
                 ee_step_m=0.0025, ee_step_rad=0.0436, ik_damping=1e-3, render_workers=RENDER_WORKERS):
        super().__init__([task_name], np.zeros(n_envs, dtype=np.int64), ecfg)
        assert action_space in ("ee", "joint"), action_space
        self.action_space, self.ee_step_m, self.ee_step_rad, self.ik_damping = action_space, float(ee_step_m), float(ee_step_rad), float(ik_damping)
        self.action_dim = 7 if action_space == "ee" else 8
        self.res = res
        self.rpool = RenderPool(task_name, res, self.N, self.nstate, render_workers) if render_workers > 1 else None
        self.cams = None if self.rpool is not None else Cameras(build_xml(task_name, visual=True), w=res, h=res)
        self.img = self.rpool.imgs if self.rpool is not None else np.zeros((self.N, len(CAMERA_NAMES), res, res, 3), np.uint8)
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
        """Render the RGB of both cameras for envs `ids` (the policy only sees RGB; no depth pass)."""
        ids = np.atleast_1d(ids)
        if len(ids) == 0:
            return
        t0 = time.perf_counter()
        if self.rpool is not None:
            self.rpool.states[ids] = self.state[ids]
            self.rpool.render(ids)
        else:
            for i in ids:
                rgb = on_render_thread(_rgb_views, self.cams, self.state[i])
                for c, name in enumerate(CAMERA_NAMES):
                    self.img[i, c] = rgb[name]
        self.stats["render"] += time.perf_counter() - t0; self.stats["steps"] += len(ids)

    def close(self):
        if self.rpool is not None:
            self.rpool.close()
        super().close()

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
