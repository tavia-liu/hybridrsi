"""Board smoke test on a GPU node: the colour marking follows the selected part (python tests/smoke_board_gpu.py <out>)."""
import sys
from pathlib import Path
sys.path[:0] = [".", "benchmark"]
from PIL import Image
import server
from assembly_policy.sim.tasks import resolve_tasks
from multi_board import PRESETS

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
b = server.Board(seed=0, tasks=resolve_tasks(PRESETS["board5"]), steps_per_part=250, record=False)
for part in b.tasks[:2]:
    b.select(part)
    print(part, b.api.instruction())
    v = b.api.camera()["agentview"]
    Image.fromarray(v["rgb"]).save(out / f"board_{part}.png")
print("BOARD SMOKE OK")
