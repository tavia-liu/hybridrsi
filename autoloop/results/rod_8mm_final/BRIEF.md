# Task

Write a policy for a 7-DoF Franka arm with a parallel-jaw gripper: pick up one part (R8) lying on the board
and bring it to its target pose on the fixture, in a physics simulator. An episode ends with SUCCESS when the part
reaches the target pose (both keypoints within 1 mm), TIMEOUT after 40 s, SAFETY if a joint exceeds its
velocity limit, or ERROR on an exception in your code.

Every episode starts with the part at a random pose on the board, the robot at home and the gripper open. The robot
has two cameras (a fixed one over the board and one on the wrist) and no other knowledge of where the part is: use
api.camera() and write your own vision, or api.locate_part(), the tracker built on them. You are scored on held-out
starts you never see, by success rate and completion time. Write one program that decides from the API; never
special-case a start or hard-code numbers read off one trace.

# How this loop works

You are the coder. You write the plan and the stage files; the loop runs the assembled policy on the debug set
(start ids in starts.json, drawn from the training bank), an independent auditor judges every failed start from
its images and trace, and you get a report: per start the outcome, your verifiers' verdicts, the auditor's verdicts
and its account of what happened. Nobody reads your replies except for their last word, READY or EXPERIMENT.

The debug set grows 2 -> 5 -> 10 -> 15 -> 20 starts: when a round passes every start, the next size runs in the same round;
passing all 20 ends the run. You have 25 rounds. Artifacts of every start land in attempts/rNN/<id>/:
trace.json (every primitive call with time, result, tcp, the tracker's part estimate, gripper width, external
wrench), summary.txt (outcome, compressed trace, your stdout), keyframes/*.png (the two cameras right after every
primitive call: left the global camera, right the wrist camera), sheet.png (16 frames spanning the episode) and
episode.mp4 (for humans; the pngs are for you).

Everything outside this directory is off limits and blocked. The shell is read-only inspection; edit files with the
Edit and Write tools.

# Program structure

plan.md: first line `Stages: a -> b -> c` (short lowercase names), then per stage what it does, the state of the
PART it must produce, and how the verifier observes that through the API. stages/NN_<name>.py (NN = 01, 02, ...)
holds the code of one stage, in plan order. The stage files run in order in one shared namespace with `api`,
`np` and `stage_done` defined, so helpers and variables of an earlier stage are visible to later ones. Every stage
ends with exactly one call stage_done(ok, evidence): ok is the verdict of the stage's own verifier, from a fresh
observation of the part, never from robot readings alone; evidence is a short factual string. FAIL stops the
policy (later stages are skipped and the episode runs out). Keep it to a few stages (e.g. locate, grasp, lift,
align, insert) and keep verifiers honest: they see only what the API reports.

# API

API level 'final'. You may call ONLY these methods on `api`:

  api.observe()
      Robot state, limited to what a real robot cell reports. The part is NOT in here: it is seen only through the cameras (camera()) or the tracker built on them (locate_part()). pos are metres in the world frame, quat are (w,x,y,z). `goal_*` is the pose the part must reach (success_threshold_m on both of its keypoints). `tcp` is the fingertip centre, its +z is the approach direction (palm -> fingertips) and +y the finger closing axis. `joint_torque_ext` [7] and `tcp_wrench_ext` [6: force xyz, torque xyz, world frame] are the external load the environment exerts on the arm, as libfranka's tau_ext_hat / O_F_ext_hat_K. The simulator's assembly distance is NOT exposed: no robot can measure it.
  api.step(action)
      One 25 Hz control step. action[0:7] are relative joint targets in units of action_scale (clipped to [-1,1]); action[7] > 0.5 commands fully open (0.08 m), < -0.5 fully closed, in between keeps the current gripper target (set_gripper(width=) sets an intermediate one). Returns observe(). Raises EpisodeOver when the episode ends.
  api.constants()
      Fixed numbers of the MDP (same every episode).
  api.camera()
      The cameras at this control step: dict(global=..., global2=..., wrist=...), each dict(rgb [H,W,3] uint8, depth [H,W] metres along the optical axis, mask [H,W] bool = the part's instance mask (a segmentation model's output), K 3x3 intrinsics, cam_pos [3], cam_R [3,3] camera-to-world). `global` and `global2` are fixed over the board (front-right and front-left), `wrist` rides on the hand looking along the fingers. A pixel (u,v) with depth d is at cam_pos + cam_R @ [(u+.5-cx)/fx*d, -(v+.5-cy)/fy*d, -d] in the world. Resolution global/global2 1280x960, wrist 640x480; no sensor noise.
  api.ik(pos, quat, q_init=None)
      Damped-least-squares IK for the TCP pose. Returns (joints[7], reachable).
  api.move_to(pos, quat, timeout_s=3.0, tol=0.01, gain=1.0)
      Servo the TCP to a world pose. Returns True if the joint target was reached within `tol` rad on every joint. Consumes episode time; raises EpisodeOver if it runs out.
  api.move_to_joints(q_target, timeout_s=3.0, tol=0.01, gain=1.0, _rec=True)
      Servo the arm to a joint configuration, holding the current gripper command.
  api.set_gripper(open_: 'bool | None' = None, hold_s=0.5, width: 'float | None' = None)
      Command the gripper and hold the arm still while the fingers act. Either open_=True/False (fully open 0.08 m / fully closed, force-limited) or width= a target opening in metres (0 .. 0.08), as a Franka Hand's move(width). The target persists until the next command.
  api.wait(seconds=0.2)
      Hold the current joint and gripper targets.
  api.locate_part()
      Pose of the part estimated from the cameras: FoundationPose (RGB-D + mask + the part's CAD mesh) run on the camera that currently sees the part best and tracked from step to step. Returns (pos [3], quat [4], conf) with conf = dict(n_points = part pixels in that camera, rms_mm = None, stale = steps since the estimate was last refreshed, views = part pixels per camera, cam = the camera used). pos/quat are None until the part has been seen once; while it is hidden the last estimate is returned (stale > 0). It is an estimate: a few mm / degrees off at best, worse when the part is small in the image or mostly hidden; symmetric parts have an arbitrary roll.
  api.part_model()
      The part's CAD mesh in the part frame (the frame locate_part() and goal_* refer to): dict(vertices [N,3] metres, faces [M,3] vertex indices, bbox_min [3], bbox_max [3]). Same every episode.

Calling anything else raises LevelError. There is no insert/align/reorient/screw primitive; the assembly strategy is yours to compute. The part's pose is never given: see it through camera() (write your own vision on the images) or locate_part() (the provided tracker).
`api.EpisodeOver` is raised as soon as the episode ends -- let it propagate.

Also allowed: numpy, and quat_mul, quat_conj, quat_to_mat, mat_to_quat, quat_apply, axis_angle_quat from
assembly_policy.sim.robot. Quaternions are (w,x,y,z), positions metres in the world frame, +z up; tcp +z points
palm->fingertips, +y is the finger closing axis. Constants: {"control_hz": 25, "horizon_s": 40.0, "action_dim": 8, "action_scale": [0.05, 0.05, 0.05, 0.06, 0.07, 0.1, 0.17], "success_threshold_m": 0.001, "joint_limits": [[-2.897, 2.897], [-1.763, 1.763], [-2.897, 2.897], [-3.072, -0.07], [-2.897, 2.897], [-0.018, 3.752], [-2.897, 2.897]], "velocity_limit": [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], "task": "R8", "family": "rod"}

# Audit and focus stage

The auditor judges every stage of every failed start from the keyframes, the sheet and the trace. Your printed
verdicts are claims: a PASS the images do not support counts as FAIL, and a start that does not end in SUCCESS is
a failure whatever your verifiers say. The report names the focus stage: the earliest stage that failed on some
start. Work on that stage; if the evidence shows an earlier stage is the real cause, revise that one and say so.

# Experiments

When you are not sure of the cause, experiment before you fix. An experiment is a short program on one or a few
starts that does the minimum to reach the situation in question and prints what it measures. Write
experiments/NN_<slug>/hypothesis.md with these lines first:
    Hypothesis: <what you believe causes the failure>
    Prediction: <what the probe will show if it is true, and if it is false>
    Prefix: <a -> b>        (optional: stage names to run first, so the probe starts from the state they leave)
    Starts: <ids>           (from starts.json)
and experiments/NN_<slug>/experiment.py: plain code over the same API in the same namespace as the stages (it may
use the prefix stages' variables). End your reply with the single word EXPERIMENT; the loop runs it and hands back
result.txt (stdout, compressed trace, artifacts under experiments/NN_<slug>/<id>/). A prefix stage's FAIL does not
stop an experiment. At most 3 experiments per round; experiments never change verdicts. An experiment
before the first plan is allowed (e.g. to look at the cameras and the part).

# Lab notebook

diagnosis.md is your running notebook: append one entry per round, never rewrite earlier ones. Before changing
code, check the hypotheses of your previous entry against the new evidence (supported or refuted); then write
`focus: <stage> -- starts <ids>`, what happened on each failed start as confirmed in BOTH the trace and the
images you looked at (cite them), the root cause, and the one change you make with the prediction it makes for
the next round, or the experiment that decides between hypotheses.

# Now

Write plan.md and the stage files (or an experiment first) and end your reply with READY (or EXPERIMENT).
