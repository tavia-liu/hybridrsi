"""Helpers of the loop: debug starts, one episode with stdout captured, traces, and the evidence it writes for the
coder and the auditor (keyframes and videos from the two cameras the policy itself sees)."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

import api

DT = 0.04


# ---------------------------------------------------------------- starts
def debug_starts(st):
    """First n of a fixed shuffle of training seeds (a larger set contains the smaller); the start state of a seed is
    what the benchmark's reset generator gives for it (part loose on the board, arm at home, gripper open)."""
    n = st["schedule"][st["stage"]]
    seeds = api.TRAIN_SEEDS.start + np.random.default_rng(0).permutation(1000)[:n]
    return [dict(id=f"s{j:02d}", seed=int(s)) for j, s in enumerate(seeds)]


def start_state(task, start):
    return api.sample_starts(task, 1, seed=start["seed"])[0]


def write_starts(run: Path, st):
    (run / "starts.json").write_text(json.dumps([s["id"] for s in debug_starts(st)]))


# ---------------------------------------------------------------- policies
def stage_verdicts(trace, names):
    """{stage: 'ok' | 'FAIL: evidence' | '-'} from the stage_done trace entries (- = never reached)."""
    v = {n: "-" for n in names}
    for e in trace:
        if e.get("call") == "stage_done" and e.get("stage") in v:
            v[e["stage"]] = "ok" if e.get("ok") else "FAIL: " + str(e.get("evidence", ""))
    return v


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location("auto_policy_" + hashlib.sha1(path.read_bytes()).hexdigest()[:8], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.policy


def run_one(env, task, start, policy, seed):
    """Run policy on one start with stdout captured. Returns the api.run_episode result + stdout."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        r = api.run_episode(env, start_state(task, start), policy, seed=seed, record=True)
    r["stdout"] = buf.getvalue()[-4000:]
    return r


def outcome_line(r):
    why = ("ERROR" if r.get("error") else "SUCCESS" if r.get("success") else "SAFETY" if r.get("safety")
           else "TIMEOUT" if r.get("timeout", True) else "ENDED")
    steps = r.get("steps", 0)
    return f"{why}  steps={steps} ({steps * DT:.1f}s)" + (f"  error={r['error']}" if r.get("error") else "")


def compress_trace(trace):
    """One line per primitive; consecutive identical (call, ok, why) entries merged with a count."""
    lines, prev, pkey, n = [], None, None, 0

    def flush():
        if prev is not None:
            lines.append(prev + (f"  x{n}" if n > 1 else ""))

    for e in trace:
        if e["call"] == "stage_done":
            core = f"[STAGE {e.get('stage')}] {'PASS' if e.get('ok') else 'FAIL'} {e.get('evidence', '')}"
        elif e["call"] == "set_gripper":
            core = f"set_gripper({'open' if e.get('open') else 'close'}) -> width {e.get('width')}"
        elif e["call"] in ("move_to", "move_to_joints"):
            core = f"{e['call']} -> {'ok' if e.get('ok') else 'FAIL:' + str(e.get('why', ''))}" + (f" err={e['err']}" if "err" in e else "")
        else:
            core = e["call"]
        line = f"t={e['t']:6.2f}  {core}  tcp={e.get('tcp')} grip={e.get('grip')} wrench={e.get('wrench')}"
        key = (e["call"], e.get("ok"), e.get("why"), e.get("open"))
        if prev is not None and key == pkey:
            n += 1
            prev = line
        else:
            flush()
            prev, pkey, n = line, key, 1
    flush()
    return lines


# ---------------------------------------------------------------- rendering (EGL; GPU node)
def label(img, lines, colors):
    from PIL import Image, ImageDraw
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(over)
    w = int(max(dr.textlength(t) for t in lines)) + 12
    dr.rectangle([0, 0, w, 12 + 13 * len(lines)], fill=(15, 17, 22, 140))
    for i, (t, col) in enumerate(zip(lines, colors)):
        dr.text((6, 4 + 13 * i), t, fill=col)
    return np.asarray(Image.alpha_composite(img, over).convert("RGB"))


class Vis:
    """Exactly the two cameras the policy gets (agentview | eye_in_hand, benchmark resolution), side by side."""

    def __init__(self, task):
        from assembly_policy.sim.scene import build_xml
        import perception
        self.cams = perception.Cameras(build_xml(task, visual=True))

    def frame(self, state):
        v = self.cams.capture(state)
        return np.concatenate([v["agentview"]["rgb"], v["eye_in_hand"]["rgb"]], axis=1)

    def keyframes(self, res, out: Path):
        from PIL import Image
        out.mkdir(parents=True, exist_ok=True)
        kf, trace = res.get("keyframes"), res["trace"]
        if kf is not None and len(kf):
            for e in trace:
                if "key" in e:
                    img = label(Image.fromarray(self.frame(kf[e["key"]])).convert("RGBA"),
                                [f"t={e['t']:.2f}s  after {e['call']}", "left: agentview   right: eye_in_hand"],
                                [(235, 235, 235, 255), (190, 200, 215, 255)])
                    Image.fromarray(img).save(out / f"k{e['key']:02d}_t{e['t']:06.2f}_{e['call']}.png")
        img = label(Image.fromarray(self.frame(res["frames"][-1])).convert("RGBA"),
                    ["final   left: agentview   right: eye_in_hand", outcome_line(res)],
                    [(235, 235, 235, 255), (255, 200, 120, 255)])
        Image.fromarray(img).save(out / "final.png")

    def video(self, res, out: Path, fps=24, every=4):
        import imageio.v2 as imageio
        frames = [self.frame(s) for s in res["frames"][::every]]
        imageio.mimwrite(out, frames + [frames[-1]] * (fps // every), fps=fps // every, codec="libx264", quality=7, macro_block_size=8)

    def sheet(self, res, path: Path, n=16, cols=4):
        """n frames spanning the episode, tiled, each labelled with its time."""
        from PIL import Image
        frames = res.get("frames")
        if frames is None or not len(frames):
            return
        idx = np.unique(np.linspace(0, len(frames) - 1, n).round().astype(int))
        tiles = [Image.fromarray(label(Image.fromarray(self.frame(frames[i])).convert("RGBA"), [f"t={i * DT:.1f}s"],
                                       [(235, 235, 235, 255)])) for i in idx]
        w, h = tiles[0].size
        rows = -(-len(tiles) // cols)
        out = Image.new("RGB", (cols * w, rows * h))
        for k, t in enumerate(tiles):
            out.paste(t, ((k % cols) * w, (k // cols) * h))
        out.resize((cols * w // 2, rows * h // 2)).save(path)


def make_vis(task):
    try:
        return Vis(task)
    except Exception as e:
        print(f"[warn] rendering unavailable ({type(e).__name__}: {e}); no keyframes", file=sys.stderr)
        return None


def write_attempt(adir: Path, start, r, code: Path | None, vis: Vis | None, video=False):
    adir.mkdir(parents=True, exist_ok=True)
    if code is not None:
        shutil.copy(code, adir / "code.py")
    (adir / "trace.json").write_text(json.dumps(r["trace"], indent=0))
    summ = [f"start {start['id']}", outcome_line(r), "",
            "TRACE (compressed; full detail in trace.json):", *compress_trace(r["trace"]), "",
            "POLICY STDOUT/STDERR:", r.get("stdout", "").rstrip() or "(none)"]
    (adir / "summary.txt").write_text("\n".join(summ) + "\n")
    if vis is not None and len(r.get("frames", ())):
        try:
            vis.keyframes(r, adir / "keyframes")
            vis.sheet(r, adir / "sheet.png")
            if video:
                vis.video(r, adir / "episode.mp4")
        except Exception as e:                                # rendering must never break the loop
            (adir / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    return summ
