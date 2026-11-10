# HybridRSI: NIST task-board assembly with cameras only

MuJoCo simulation of the NIST Assembly Task Board 1 (20 parts, Franka Panda), built on
[AssembleNIST](https://github.com/suniique/assembly-nist). The policy gets what a real cell gives it and nothing more:

| given | not given |
| --- | --- |
| robot state (joints, TCP pose, gripper width, external joint torques / TCP wrench, Jacobian) | the part's pose |
| three RGB-D cameras (two fixed over the board, one on the wrist) with intrinsics and camera poses | the part's segmentation mask |
| the goal pose of the part and its success threshold | the part's CAD model |
| joint-space control at 25 Hz, IK, servo-to-pose, gripper | any pose estimator or tracker |

Parts, fixtures and the board all use the same plain metal colour (`HYBRIDRSI_COLORS=debug` restores the
orange part / blue fixture rendering for debugging).

A policy is a Python file defining `policy(api)`; `api` (benchmark/api.py) is its only interface.
`PYTHONPATH=.:benchmark python -c "import api; print(api.spec())"` prints everything it can call.

```
PYTHONPATH=.:benchmark MUJOCO_GL=egl
python benchmark/eval_protocol.py --task R8 --policy examples/noop_policy.py      # one part, 100 held-out starts, videos
python benchmark/multi_board.py --tasks board20 --policy examples/noop_policy.py  # all 20 parts on the board
sbatch --export=ALL,CMD="..." gpu.sbatch                                          # any command on a GPU (EGL rendering)
```

Scored setting: `robot.board(seed)` is one continuous task, all 20 parts physically on the board in a random
order, 240 s per part pooled, scored by the parts seated at the end over 10 held-out boards
(`benchmark/eval_remote.py --board`). The cell can also run as a service for a sandboxed agent
(`benchmark/server.py` + `benchmark/robot.py`, see their docstrings).

Setup: `pip install -r requirements.txt` and build the SDF plugin once with `plugin/build.sh` (MuJoCo loads it
from `plugin/build/`). Rendering needs a GPU with EGL.
