"""The HybridRSI cell as a dm_env for DrQ-v2 (training only; the benchmark API underneath is untouched).

Observation: the two cameras (agentview, eye_in_hand), 84x84 RGB each, concatenated along channels -> (84, 84, 6)
uint8 under the key "pixels" (DrQ-v2's FrameStackWrapper stacks 3 of them -> 18 x 84 x 84). Nothing else.
Action: 7 numbers in [-1, 1]: TCP translation (x, y, z, up to ee_step_m per control step), TCP rotation (rotation
vector, up to ee_step_rad per control step) and the gripper (> 0.5 open, < -0.5 close, else keep, as api.step's
action[7]). Each action is held for `repeat` 25 Hz control steps (12.5 Hz decisions for repeat=2); the cameras are
rendered once per decision. Reward: 1 when the simulator reports success (episode ends, discount 0), 0 otherwise;
a safety stop ends the episode with -safety_penalty; running out of `episode_steps` control steps truncates it
(discount 1, so the critic bootstraps). Starts: `family` starts (near = part held above its fixture) sampled from
the training seeds.
"""
from __future__ import annotations

import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark")]
IMAGE = int(os.environ.get("HYBRIDRSI_RL_IMAGE", "84"))
os.environ.setdefault("CAP_CAM_RES", f"{IMAGE}x{IMAGE}")      # must precede the perception import

import dm_env  # noqa: E402
import numpy as np  # noqa: E402
from dm_env import StepType, specs  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import api  # noqa: E402
from assembly_policy.sim.robot import mat_to_quat, quat_to_mat  # noqa: E402

CAMERAS = ("agentview", "eye_in_hand")


class NearEnv(dm_env.Environment):
    def __init__(self, task="P4", family="near", episode_steps=375, repeat=2, ee_step_m=0.0025, ee_step_rad=0.0436,
                 gripper_action=True, safety_penalty=0.0, seed=0):
        self.task, self.family, self.repeat = task, family, int(repeat)
        self.ee_step_m, self.ee_step_rad = float(ee_step_m), float(ee_step_rad)
        self.gripper_action, self.safety_penalty = bool(gripper_action), float(safety_penalty)
        self.env = api.make_env(task, steps=episode_steps)
        self.rng = np.random.default_rng(seed)
        self.api = None
        self.done = True
        self.last_outcome = {}

    # ---------------------------------------------------------------- dm_env
    def observation_spec(self):
        return {"pixels": specs.BoundedArray((IMAGE, IMAGE, 3 * len(CAMERAS)), np.uint8, 0, 255, "pixels")}

    def action_spec(self):
        return specs.BoundedArray((7,), np.float32, -1.0, 1.0, "action")

    def reset(self):
        seed = int(self.rng.integers(api.TRAIN_SEEDS.start, 2**31 - 1))
        s0 = api.sample_starts(self.task, 1, seed=seed, family=self.family)[0]
        self.env.set_states([0], s0[None], task_ids=[0])
        self.api = api.AssemblyAPI(self.env, self.rng, record=False)
        self.done = False
        self.last_outcome = {}
        return dm_env.TimeStep(StepType.FIRST, None, None, self._obs())

    def step(self, action):
        if self.done:
            return self.reset()
        a = np.clip(np.asarray(action, np.float32).reshape(7), -1.0, 1.0)
        try:
            for _ in range(self.repeat):
                self._control_step(a)
        except api.EpisodeOver:
            self.done = True
            self.last_outcome = dict(self.api.outcome)
            obs = self._obs()
            if self.last_outcome.get("success"):
                return dm_env.termination(reward=1.0, observation=obs)
            if self.last_outcome.get("safety"):
                return dm_env.termination(reward=-self.safety_penalty, observation=obs)
            return dm_env.truncation(reward=0.0, observation=obs, discount=1.0)
        return dm_env.transition(reward=0.0, observation=self._obs(), discount=1.0)

    # ---------------------------------------------------------------- internals
    def _control_step(self, a):
        o = self.api.observe()
        pos = o["tcp_pos"] + a[:3] * self.ee_step_m
        R = Rotation.from_rotvec(a[3:6] * self.ee_step_rad).as_matrix() @ quat_to_mat(o["tcp_quat"])
        q_target, ok = self.api.ik(pos, mat_to_quat(R[None])[0])
        ja = np.zeros(8)
        if ok:
            ja[:7] = np.clip((q_target - o["joints"]) / self.api.scale, -1.0, 1.0)
        ja[7] = a[6] if self.gripper_action else -1.0
        self.api.step(ja)

    def _obs(self):
        v = self.api.camera()
        return {"pixels": np.concatenate([v[c]["rgb"] for c in CAMERAS], axis=2)}


# ---------------------------------------------------------------------- DrQ-v2's wrappers (from drqv2/dmc.py, MIT)
class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        return tuple.__getitem__(self, attr)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env, self._num_frames, self._pixels_key = env, num_frames, pixels_key
        self._frames = deque([], maxlen=num_frames)
        shape = env.observation_spec()[pixels_key].shape
        self._obs_spec = specs.BoundedArray(shape=np.concatenate([[shape[2] * num_frames], shape[:2]], axis=0),
                                            dtype=np.uint8, minimum=0, maximum=255, name="observation")

    def _transform(self, time_step):
        assert len(self._frames) == self._num_frames
        return time_step._replace(observation=np.concatenate(list(self._frames), axis=0))

    def _pixels(self, time_step):
        return time_step.observation[self._pixels_key].transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        px = self._pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(px)
        return self._transform(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        self._frames.append(self._pixels(time_step))
        return self._transform(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        return self._augment(self._env.reset())

    def step(self, action):
        return self._augment(self._env.step(action), action)

    def _augment(self, time_step, action=None):
        if action is None:
            spec = self.action_spec()
            action = np.zeros(spec.shape, dtype=spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation, step_type=time_step.step_type, action=action,
                                reward=time_step.reward or 0.0, discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(cfg, seed):
    """NearEnv with DrQ-v2's frame stacking; action repeat happens inside NearEnv (one render per decision)."""
    env = NearEnv(task=cfg.task, family=cfg.family, episode_steps=cfg.episode_steps, repeat=cfg.action_repeat,
                  ee_step_m=cfg.ee_step_m, ee_step_rad=cfg.ee_step_rad, gripper_action=cfg.gripper_action,
                  safety_penalty=cfg.safety_penalty, seed=seed)
    env = FrameStackWrapper(env, cfg.frame_stack, "pixels")
    return ExtendedTimeStepWrapper(env)
