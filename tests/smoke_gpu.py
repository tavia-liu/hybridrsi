"""Smoke test on a GPU node (PYTHONPATH=.:benchmark MUJOCO_GL=egl python tests/smoke_gpu.py <out>): spec, one short episode, camera dumps, one held-out start with video."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = [".", "benchmark"]
import api  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
spec = api.spec()
assert "locate_part" not in spec and "part_model" not in spec and "goal_pos" not in spec, spec
print(spec[:600], "...\n")

env = api.make_env("R8", horizon_s=4.0)
s0 = api.sample_starts("R8", 1, seed=0)[0]
env.set_states([0], s0[None], task_ids=[0])
a = api.AssemblyAPI(env, np.random.default_rng(0), record=True)
t0 = time.time()
views = a.camera()
print("camera() keys:", {k: sorted(v) for k, v in views.items()}, f"{time.time() - t0:.1f}s")
for k, v in views.items():
    assert set(v) == {"rgb", "depth", "K", "cam_pos", "cam_R"}, (k, sorted(v))
from PIL import Image  # noqa: E402
for k, v in views.items():
    Image.fromarray(v["rgb"]).save(out / f"{k}.png")
obs = a.observe()
assert not any(key.startswith(("part", "goal")) for key in obs), list(obs)
print("instruction:", a.instruction())
print("observe() keys:", sorted(obs))
try:
    a.set_gripper(True); a.wait(5.0)
except api.EpisodeOver:
    pass
print("outcome:", a.outcome, "trace keys:", sorted(a.trace[-1]) if a.trace else None)
assert a.trace and "part" not in a.trace[-1]

# held-out protocol, one start, with video
import eval_protocol  # noqa: E402
pol = eval_protocol.load_policy(Path("examples/noop_policy.py"))
env2 = api.make_env("R8", horizon_s=3.0)
summary, rows = eval_protocol.evaluate(env2, pol, "R8", n=1, seed=0, record=True,
                                       on_episode=lambda k, r: eval_protocol.Video("R8").write(r, out / "e000.mp4"), verbose=False)
print("eval:", json.dumps(summary)[:300])
assert (out / "e000.mp4").exists()
print("SMOKE OK")
