"""The policy-facing API of the benchmark: one episode of one NIST task-board assembly.

The simulator accepts 8-D relative joint increments at 25 Hz (q_target = q + scale*a, a[7] latches the gripper).
AssemblyAPI wraps that with what a real cell offers a policy: the robot's state, two RGB-D cameras, a
one-sentence instruction in natural language and task-agnostic kinematics (IK, servo to a pose, gripper). Nothing
else: no part pose, no goal pose, no part mask, no CAD model, no pose estimator. In the images the part to assemble
is orange and its fixture blue; everything else is grey. The simulator's part pose and assembly distance are never
exposed.

Seeds: 0..50 are the held-out evaluation starts (EVAL_SEEDS; single part: all 51 per task, board: EVAL_BOARDS = 0..9),
every seed from 51 up is for development and training (TRAIN_SEEDS). A seed sets only where the part lies (and, on the
board, the order of the parts); the arm always starts at home with the gripper open.

`spec()` renders the documentation of every method from the docstrings below, so what a policy is told cannot
drift from what it can call. A policy is `policy(api)`; `run_episode()` runs it on one start state.
"""
from __future__ import annotations

import numpy as np
import mujoco

from assembly_policy.sim.env import STATE_SPEC, BatchedAssemblyEnv, EnvConfig
from assembly_policy.sim.resets import ResetGenerator
from assembly_policy.sim.robot import mat_to_quat, solve_ik
from assembly_policy.sim.scene import build_xml
from assembly_policy.sim.tasks import TASKS

METHODS = ("instruction", "observe", "step", "constants", "camera", "ik", "move_to", "move_to_joints", "set_gripper", "wait")
START_FAMILY = "board"                   # the only start: part dropped loose on the board, arm at home, gripper open
CONTROL_DT = 0.04                     # 25 Hz
STEPS_PER_PART = 6000                 # the episode budget, in control steps (= 240 s at 25 Hz)
EVAL_SEEDS = range(0, 51)             # held-out; the cell server hands them out only with its token
EVAL_BOARDS = range(0, 10)            # the scored boards
TRAIN_SEEDS = range(51, 2**31 - 1)    # development and training

INSTRUCTIONS = {
    "rod": "Pick up the orange rod and insert it all the way into the hole of the blue fixture.",
    "peg": "Pick up the orange peg and insert it all the way into the hole of the blue fixture.",
    "gear": "Pick up the orange gear and push it all the way down onto the shaft of the blue fixture.",
    "nut": "Pick up the orange nut and screw it all the way down the bolt of the blue fixture.",
    "connector": "Pick up the orange connector and plug it all the way into the socket of the blue fixture.",
}


class EpisodeOver(Exception):
    """The episode terminated (success, safety violation, or the step budget ran out)."""


class AssemblyAPI:
    """One episode of one task. A policy drives it through these methods only."""

    def __init__(self, env: BatchedAssemblyEnv, rng: np.random.Generator, record: bool = False):
        self.env, self.rng = env, rng
        self.tm = env.models[0]
        self.m = self.tm.model
        self._ik_d = mujoco.MjData(self.m)
        self._obs_d = mujoco.MjData(self.m)     # scratch for the contact solver in observe()
        self.task = TASKS[env.task_names[0]]
        ecfg = env.ecfg
        self.scale = (np.asarray(ecfg.action_scale_joints, float) if len(ecfg.action_scale_joints)
                      else np.full(7, ecfg.action_scale))
        self.trace: list[dict] = []
        self.frames: list | None = [] if record else None   # per-step states, for offline rendering
        self.key_states: list = []                           # state at every recorded primitive call (keyframes)
        self.done = False
        self.outcome: dict = {}
        # start states may hand over an already-closed gripper (pickup/inhand/partial/near);
        # env.set_states latches grip_cmd from the finger opening, so mirror it or the first
        # servo step would re-open the fingers and drop the part
        self.grip_open = bool(env.grip_cmd[0] > 0)
        self._a7 = 1.0 if self.grip_open else -1.0      # what the kin primitives send on action[7]
        # the cameras: the only way the part reaches a policy
        from perception import Cameras
        if hasattr(env, "tasks"):                                       # multi-part board (benchmark/multi_env.py)
            from multi_scene import build_multi_xml
            xml, suffix = build_multi_xml(env.tasks, visual=True), f"_{self.task.name}"
        else:
            xml, suffix = build_xml(self.task.name, visual=True), ""
        self._cams = Cameras(xml, part_body=f"part{suffix}", fixture_body=f"fixture{suffix}",
                             fixture_site=f"fixture_frame{suffix}")
        self._views_cache = None

    def _views(self):
        """The camera views at this control step (rendered once per step)."""
        if self._views_cache is None or self._views_cache[0] != int(self.env.t[0]):
            self._views_cache = (int(self.env.t[0]), self._cams.capture(self.env.state[0]))
        return self._views_cache[1]

    # ------------------------------------------------------------------ enforcement
    def _rec(self, call, **kw):
        """Append a trace entry: the primitive, its result, and what the robot could measure afterwards."""
        s = self.env.state[0]
        fk = self.tm.fk.forward(self.env.state[:, self.env.sq])
        _, wrench = self._external_load(self._jacobian())
        e = dict(t=round(float(self.env.t[0]) * self.env.ecfg.control_dt, 2), call=call, **kw,
                 tcp=np.round(fk["tcp_p"][0], 4).tolist(),
                 grip=round(float(s[self.env.sf].sum()), 4), wrench=np.round(wrench, 1).tolist())
        if self.frames is not None:
            self.key_states.append(s.copy())
            e["key"] = len(self.key_states) - 1
        self.trace.append(e)

    # ------------------------------------------------------------------ raw
    def instruction(self):
        """The task, in one sentence. The part to assemble is the orange one and its fixture the blue one in the
        camera images; everything else on the board is grey. Where they are, and the part's shape and size, are
        not given anywhere else: read them off the images."""
        return INSTRUCTIONS[self.task.family]

    def constants(self):
        """Fixed numbers of the MDP (same every episode)."""
        return dict(control_hz=round(1.0 / self.env.ecfg.control_dt), step_budget=int(self.env.horizon),
                    action_dim=8, action_scale=self.scale.round(4).tolist(),
                    joint_limits=self.env.idx.jnt_range.round(3).tolist(),
                    velocity_limit=self.env.vlim.round(3).tolist())

    def observe(self):
        """Robot state, limited to what a real robot cell reports. Neither the part nor its goal is in here:
        both are seen only through the cameras (camera()).

        `step` / `steps_left` count 25 Hz control steps against the episode's step budget (every primitive below
        consumes them; thinking time does not). pos are metres in the world frame, quat are (w,x,y,z). `tcp` is the fingertip
        centre, its +z is the approach direction (palm -> fingertips) and +y the finger closing
        axis. `joint_torque_ext` [7] and `tcp_wrench_ext` [6: force xyz, torque xyz, world
        frame] are the external load the environment exerts on the arm, as libfranka's
        tau_ext_hat / O_F_ext_hat_K. The simulator's assembly distance is NOT exposed: no
        robot can measure it.
        """
        s = self.env.state[0]
        fk = self.tm.fk.forward(self.env.state[:, self.env.sq])
        J = self._jacobian()
        tau_ext, wrench = self._external_load(J)
        return dict(
            step=int(self.env.t[0]), steps_left=int(self.env.horizon - self.env.t[0]),
            tcp_pos=fk["tcp_p"][0].copy(), tcp_quat=mat_to_quat(fk["tcp_R"][:1])[0],
            joints=s[self.env.sq].copy(), joint_vel=s[self.env.sqd].copy(),
            gripper_width=float(s[self.env.sf].sum()),
            gripper_open_cmd=self.grip_open,
            gripper_target_width=round(float(self.env.grip_cmd[0]) * 2, 4),
            tcp_jacobian=J,
            joint_torque_ext=tau_ext, tcp_wrench_ext=wrench,
        )

    def camera(self):
        """The two RGB-D cameras at this control step: dict(agentview=..., eye_in_hand=...), each dict(rgb [H,W,3]
        uint8, depth [H,W] metres along the optical axis, K 3x3 intrinsics, cam_pos [3], cam_R [3,3]
        camera-to-world). The part to assemble is orange, its fixture blue, everything else grey. No segmentation,
        no poses: where the part and the fixture are must be read off the images. `agentview` is fixed in front
        of the board, opposite the robot, looking down at it; `eye_in_hand` rides on the hand looking along the
        fingers. A pixel (u,v) with depth d is at cam_pos + cam_R @ [(u+.5-cx)/fx*d, -(v+.5-cy)/fy*d, -d] in the
        world. Resolution {RES}; no sensor noise."""
        return {n: dict(v) for n, v in self._views().items()}

    def _external_load(self, J):
        """External joint torques from the constraint solver and the TCP wrench they imply.

        mj_forward on the current full state recomputes contacts without stepping;
        qfrc_constraint is the generalized force the environment (contacts, joint limits)
        applies to the arm. A real Franka estimates the same quantities from its joint torque
        sensors every cycle, so exposing them models a robot that can feel a jam.
        """
        d = self._obs_d
        mujoco.mj_setState(self.m, d, self.env.state[0], STATE_SPEC)
        mujoco.mj_forward(self.m, d)
        tau_ext = d.qfrc_constraint[self.env.idx.arm_dof].copy()
        wrench = np.linalg.pinv(J.T) @ tau_ext          # tau = J^T f  ->  f (6,)
        return tau_ext, wrench

    def _jacobian(self):
        """6x7 TCP Jacobian at the current joints (rows: linear xyz, then angular xyz).

        A real Franka controller has this -- libfranka publishes it every cycle -- so
        withholding it would model a robot that does not know its own kinematics.
        """
        d = self._ik_d
        d.qpos[self.env.idx.arm_qpos] = self.env.state[0, self.env.sq]
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        mujoco.mj_jacSite(self.m, d, jp, jr, self.env.idx.tcp_site)
        return np.vstack([jp[:, self.env.idx.arm_dof], jr[:, self.env.idx.arm_dof]])

    def step(self, action):
        """One 25 Hz control step. action[0:7] are relative joint targets in units of
        action_scale (clipped to [-1,1]); action[7] > 0.5 commands fully open (0.08 m), < -0.5
        fully closed, in between keeps the current gripper target (set_gripper(width=) sets an
        intermediate one). Returns observe(). Raises EpisodeOver when the episode ends."""
        return self._step(np.asarray(action, float).reshape(8))

    # ------------------------------------------------------------------ kin
    def ik(self, pos, quat, q_init=None):
        """Damped-least-squares IK for the TCP pose. Returns (joints[7], reachable)."""
        q, ok = solve_ik(self.m, self._ik_d, self.env.idx, np.asarray(pos, float),
                         np.asarray(quat, float),
                         q_init=self.env.state[0, self.env.sq] if q_init is None else q_init,
                         rng=self.rng, restarts=2)
        return q, bool(ok)

    def move_to(self, pos, quat, timeout_s=3.0, tol=0.01, gain=1.0):
        """Servo the TCP to a world pose. Returns True if the joint target was reached within
        `tol` rad on every joint. Consumes steps of the budget; raises EpisodeOver when it runs out."""
        q_target, reachable = self.ik(pos, quat)
        if not reachable:
            self._rec("move_to", ok=False, why="unreachable", pos=np.round(pos, 4).tolist())
            return False
        ok = self.move_to_joints(q_target, timeout_s=timeout_s, tol=tol, gain=gain, _rec=False)
        self._rec("move_to", ok=ok, why="" if ok else "servo_timeout",
                  pos=np.round(pos, 4).tolist(), err=self._joint_err(q_target))
        return ok

    def move_to_joints(self, q_target, timeout_s=3.0, tol=0.01, gain=1.0, _rec=True):
        """Servo the arm to a joint configuration, holding the current gripper command."""
        q_target = np.asarray(q_target, float).reshape(7)
        a = np.zeros(8)
        for _ in range(max(1, int(timeout_s / self.env.ecfg.control_dt))):
            err = q_target - self.env.state[0, self.env.sq]
            if np.abs(err).max() < tol:
                break
            a[:7] = np.clip(gain * err / self.scale, -1.0, 1.0)
            a[7] = self._a7
            self._step(a)
        ok = bool(np.abs(q_target - self.env.state[0, self.env.sq]).max() < tol)
        if _rec:
            self._rec("move_to_joints", ok=ok, err=self._joint_err(q_target))
        return ok

    def set_gripper(self, open_: bool | None = None, hold_s=0.5, width: float | None = None):
        """Command the gripper and hold the arm still while the fingers act. Either open_=True/False
        (fully open 0.08 m / fully closed, force-limited) or width= a target opening in metres
        (0 .. 0.08), as a Franka Hand's move(width). The target persists until the next command."""
        if width is not None:
            per_finger = float(np.clip(width / 2.0, self.env.ecfg.grip_close, 0.04))
            self.env.grip_cmd[0] = per_finger
            self.grip_open = per_finger > 0.03
            self._a7 = 0.0                              # hold: action[7] in the deadband keeps grip_cmd
        else:
            self.grip_open = bool(open_)
            self._a7 = 1.0 if self.grip_open else -1.0
        a = np.zeros(8)
        a[7] = self._a7
        for _ in range(max(1, int(hold_s / self.env.ecfg.control_dt))):
            self._step(a)
        self._rec("set_gripper", open=self.grip_open, target=round(float(self.env.grip_cmd[0]) * 2, 4),
                  width=round(float(self.env.state[0, self.env.sf].sum()), 4))

    def wait(self, seconds=0.2):
        """Hold the current joint and gripper targets."""
        a = np.zeros(8)
        a[7] = self._a7
        for _ in range(max(1, int(seconds / self.env.ecfg.control_dt))):
            self._step(a)

    # ------------------------------------------------------------------ internals
    def _joint_err(self, q_target):
        return round(float(np.abs(q_target - self.env.state[0, self.env.sq]).max()), 4)

    def _step(self, a):
        if self.done:
            raise EpisodeOver()
        if self.frames is not None:
            self.frames.append(self.env.state[0].copy())
        _, done, info = self.env.step(a[None, :])
        if bool(done[0]):
            self.done = True
            self.outcome = dict(success=bool(info["success"][0]), safety=bool(info["safety"][0]),
                                timeout=bool(info["timeout"][0]),
                                steps=int(self.env.t[0]))
            raise EpisodeOver()
        return self.observe()


# ---------------------------------------------------------------------- prompt spec
def spec():
    """The API documentation a policy is given, rendered from the method docstrings."""
    import inspect
    lines = ["Methods of `api`, the only interface to the robot and the cameras:", ""]
    for name in METHODS:
        fn = getattr(AssemblyAPI, name)
        doc = " ".join((fn.__doc__ or "").split())
        if name == "camera":
            from perception import W, H
            doc = doc.replace("{RES}", f"{W}x{H} for both")
        sig = str(inspect.signature(fn)).replace("(self, ", "(").replace("(self)", "()").split(" -> ")[0]
        lines.append(f"  api.{name}{sig}\n      {doc}")
    lines += ["", "There is no insert/align/reorient/screw primitive; the assembly strategy is yours to compute. The task is "
                  "given only by instruction(): no part pose, no goal pose, no segmentation, no CAD model. Everything about the "
                  "part and its fixture comes from your own vision on camera() images (RGB + depth, known intrinsics and "
                  "camera poses); the episode succeeds when the part is fully seated in the fixture.",
              "`api.EpisodeOver` is raised as soon as the episode ends -- let it propagate."]
    return "\n".join(lines)


# ---------------------------------------------------------------------- episode plumbing
def make_env(task_name, nthread=1, steps=STEPS_PER_PART):
    """One-env simulator for `task_name` with a budget of `steps` control steps (the simulator keeps it as seconds)."""
    return BatchedAssemblyEnv([task_name], [0], EnvConfig(nthread=nthread, horizon_s=float(steps) * CONTROL_DT))


def sample_starts(task_name, n, seed=0, family=START_FAMILY):
    """n start states: the part dropped at a random pose on the board and settled, the arm at home, the gripper
    open. The seed sets only the part's pose."""
    states, _ = ResetGenerator(task_name, seed=seed).generate(family, n)
    return states


def run_episode(env, state, policy, seed=0, record=False):
    """Run `policy(api)` on one start state. Returns the outcome and the primitive trace."""
    env.set_states([0], np.asarray(state)[None, :], task_ids=[0])
    api = AssemblyAPI(env, np.random.default_rng(seed), record=record)
    err = None
    try:
        policy(api)
    except EpisodeOver:
        pass
    except Exception as e:                       # a buggy policy is a result, not a crash
        err = f"{type(e).__name__}: {e}"
    if not api.done:                             # policy returned early; let the clock run out
        try:
            while True:
                api.wait(0.2)
        except EpisodeOver:
            pass
    out = dict(**api.outcome, error=err, trace=api.trace)
    if api.frames is not None:
        out["frames"] = np.asarray(api.frames)
        out["keyframes"] = np.asarray(api.key_states) if api.key_states else np.zeros((0, api.frames and len(api.frames[0]) or 0))
    return out
