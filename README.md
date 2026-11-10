# HybridRSI: NIST task-board assembly from cameras and a sentence

MuJoCo simulation of the NIST Assembly Task Board 1 (20 parts, Franka Panda), built on
[AssembleNIST](https://github.com/suniique/assembly-nist). The policy gets what a real cell gives it and nothing more:

| given | not given |
| --- | --- |
| one sentence: `api.instruction()`, e.g. *Pick up the orange rod and insert it all the way into the hole of the blue fixture.* | the part's pose or its goal pose |
| robot state (joints, TCP pose, gripper width, external joint torques / TCP wrench, Jacobian) | the part's segmentation mask |
| three RGB-D cameras (two fixed over the board, one on the wrist) with intrinsics and camera poses | the part's CAD model or size |
| joint-space control at 25 Hz, IK, servo-to-pose, gripper | any pose estimator or tracker |

In the images the part to assemble is orange and its fixture blue; every other part and fixture is grey. That colour
marking is the only grounding of the sentence.

A policy is a Python file defining `policy(api)`; `api` (benchmark/api.py) is its only interface.
`PYTHONPATH=.:benchmark python -c "import api; print(api.spec())"` prints everything it can call.

## Starts and seeds

Every episode starts the same way: the arm at its home configuration with the gripper open, the part(s) dropped at
random poses on the board and settled. There are no other start families. A seed sets only where the parts lie (and,
on the board, the order in which they are scored).

| seeds | use |
| --- | --- |
| 0 .. 50 | held-out evaluation: single part = all 51 seeds per task; board = seeds 0..9 (10 boards). The cell server hands these out only with its token. |
| 51 and up | development and training |

## Scoring

The benchmark score is `robot.board(seed)`: one continuous task with all 20 parts physically on the board in a random
order, a pooled budget of 6000 control steps per part (25 Hz, so 240 s of robot time per part; thinking time is free), scored by the parts seated at the end, over the 10 held-out boards (seeds 0..9)
(`benchmark/eval_remote.py --board`). Single-part episodes (`benchmark/eval_protocol.py`, `robot.start`) use the same
start and are for development.

```
PYTHONPATH=.:benchmark MUJOCO_GL=egl
python benchmark/eval_protocol.py --task R8 --policy examples/noop_policy.py      # one part, 51 held-out starts, videos
python benchmark/multi_board.py --tasks board20 --policy examples/noop_policy.py  # all 20 parts on the board
sbatch --export=ALL,CMD="..." gpu.sbatch                                          # any command on a GPU (EGL rendering)
python tests/smoke_gpu.py /tmp/smoke                                              # plumbing check on a GPU node
```

The cell can also run as a service for a sandboxed agent (`benchmark/server.py` + `benchmark/robot.py`, see their
docstrings): the sandbox sees `robot.py` only.

Setup: `pip install -r requirements.txt` and build the SDF plugin once with `plugin/build.sh` (MuJoCo loads it
from `plugin/build/`). Rendering needs a GPU with EGL.
