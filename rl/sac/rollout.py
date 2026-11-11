"""Roll out a rl/sac checkpoint on a few bank starts, record both cameras as a tiled video and log every action.

  PYTHONPATH=.:benchmark:rl/ppo:rl/sac MUJOCO_GL=egl python rl/sac/rollout.py --run runs/sac/<run> [--n 6] [--mode both]

Writes <run>/rollouts/<mode>_it<iteration>.mp4 (cells = [agentview | eye_in_hand] per env rendered at --video-res with an
overlay of step, assembly distance, gripper width and the last action; 12 fps = every 2nd control step)
and <mode>_it<iteration>_actions.txt: one line per control step and env with the 7-D action (EE dx dy dz, drot xyz, gripper),
TCP position, gripper width, assembly distance (simulator-side, for diagnosis only), reward and the outcome.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "benchmark"), str(ROOT / "rl" / "ppo"), str(ROOT / "rl" / "sac")]

from assembly_policy.sim.env import EnvConfig, parse_scales  # noqa: E402
from nets import ResNetEncoder  # noqa: E402
from pixel_env import PixelEnv  # noqa: E402
from sac_nets import SACModel, pool_features  # noqa: E402
from assembly_policy.sim.scene import build_xml  # noqa: E402
from perception import CAMERA_NAMES, Cameras  # noqa: E402


def _label(frame, text):
    from PIL import Image, ImageDraw
    im = Image.fromarray(frame); dr = ImageDraw.Draw(im)
    dr.rectangle([0, 0, 8 + 7 * len(text), 16], fill=(0, 0, 0)); dr.text((4, 2), text, fill=(255, 255, 0))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--mode", default="both", choices=["deterministic", "stochastic", "both"])
    ap.add_argument("--first", type=int, default=0, help="first bank entry to use")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--video-res", type=int, default=512, help="video rendering resolution per camera (the policy still sees --image)")
    o = ap.parse_args()
    run = Path(o.run); dev = torch.device(o.device)
    ck = torch.load(run / "checkpoint.pt", map_location=dev, weights_only=False); a = ck["args"]
    plain = a.get("encoder", "resnet") == "plainconv"
    hidden = tuple(int(x) for x in a.get("hidden", "256,256").split(","))
    m = SACModel(150, 7 if a["action_space"] == "ee" else 8, a["z_dim"], a["n_critics"], not a["no_layer_norm"], a["init_alpha"],
                 encoder=a.get("encoder", "resnet"), image=a["image"], hidden=hidden, prop_norm=not a.get("no_prop_norm", False),
                 last_act=plain).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    trunk = None if plain else ResNetEncoder(a["image"], freeze=True).to(dev).eval()
    bank = np.load(run / "bank.npz")["states"]; S = bank[o.first:o.first + o.n]; N = len(S)
    ecfg = EnvConfig(nthread=4, cost_coef=a["cost_coef"], horizon_s=a["episode_steps"] * 0.04,
                     action_scale_joints=parse_scales(a["action_scale_joints"]))
    env = PixelEnv(a["task"], N, ecfg, res=a["image"], action_space=a["action_space"], ee_step_m=a["ee_step_m"],
                   ee_step_rad=a["ee_step_rad"], render_workers=4)
    out = run / "rollouts"; out.mkdir(exist_ok=True)
    hi = Cameras(build_xml(a["task"], visual=True), w=o.video_res, h=o.video_res)      # sharper views for the video only
    modes = ["deterministic", "stochastic"] if o.mode == "both" else [o.mode]
    for mode in modes:
        env.set_states(np.arange(N), S)
        done = np.zeros(N, bool); outcome = [""] * N; frames = []; lines = []
        d0 = env.assembly_distance() * 1000
        last_act = np.zeros((N, 7))
        for t in range(a["episode_steps"]):
            ob = env.observe(); img = ob["img"]
            if t % 2 == 0:
                dnow = env.assembly_distance() * 1000; wnow = env.state[:, env.sf].sum(1)
                cells = []
                for i in range(N):
                    v = hi.capture(env.state[i])
                    cell = np.concatenate([v[n]["rgb"] for n in CAMERA_NAMES], 1)
                    txt = f"env{i} t={t:3d} d={dnow[i]:5.1f}mm grip={wnow[i]*1000:4.1f}mm a=[{' '.join(f'{x:+.1f}' for x in last_act[i])}] {outcome[i]}"
                    cells.append(_label(cell, txt))
                cols = min(2, N); rows = [np.concatenate(cells[r * cols:(r + 1) * cols], 1) for r in range((N + cols - 1) // cols)]
                w = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
                frames.append(np.concatenate(rows, 0))
            with torch.no_grad():
                imt = torch.as_tensor(img, device=dev)
                if plain:
                    feat = imt.permute(0, 2, 3, 1, 4).reshape(N, imt.shape[2], imt.shape[3], -1).contiguous()
                else:
                    feat = pool_features(trunk.features(imt)).half()
                prop = torch.as_tensor(ob["proprio"], dtype=torch.float32, device=dev)
                act = m.act(feat, prop, deterministic=(mode == "deterministic")).cpu().numpy()
            act[done] = 0; act[done, -1] = -1
            last_act = act.copy()
            tcp = env._fk(np.arange(N))["tcp_p"]; width = env.state[:, env.sf].sum(1); d = env.assembly_distance() * 1000
            r, dn, info = env.step(act)
            for i in range(N):
                if done[i]:
                    continue
                oc = "SUCCESS" if info["success"][i] else ("SAFETY" if info["safety"][i] else ("TIMEOUT" if info["timeout"][i] else ""))
                lines.append(f"t={t:3d} env={i} a=[{' '.join(f'{x:+.2f}' for x in act[i])}] tcp=[{tcp[i,0]:.4f} {tcp[i,1]:.4f} {tcp[i,2]:.4f}] "
                             f"grip={width[i]:.4f} d_asm={d[i]:6.2f}mm r={r[i]:+.4f} {oc}")
            for i in np.nonzero(dn & ~done)[0]:
                outcome[i] = "SUCCESS" if info["success"][i] else ("SAFETY" if info["safety"][i] else "TIMEOUT")
            done |= dn
            if done.all():
                break
        import imageio
        tag = f"{mode}_it{ck['it']}"
        imageio.mimwrite(out / f"{tag}.mp4", frames, fps=12, codec="libx264", quality=8)
        head = [f"run {run}  checkpoint it {ck['it']}  transitions {ck['transitions']}  alpha {m.alpha.item():.4f}  mode {mode}",
                "action = [EE dx dy dz (x ee_step_m), drot x y z (x ee_step_rad), gripper (>0.5 open, <-0.5 close)]",
                f"outcomes: " + ", ".join(f"env{i}: {outcome[i] or 'running'} (start d_asm {d0[i]:.1f} mm)" for i in range(N)), ""]
        (out / f"{tag}_actions.txt").write_text("\n".join(head + lines) + "\n")
        print(f"{tag}: " + ", ".join(f"env{i}={outcome[i] or 'running'}" for i in range(N)), flush=True)
    env.close()


if __name__ == "__main__":
    main()
