"""The cameras: what a real cell has instead of the simulator's part pose.

Three RGB-D cameras -- two fixed over the board (front-right, front-left) and the wrist camera (robosuite
eye-in-hand) -- give RGB, metric depth, intrinsics and the camera pose. Nothing else is rendered for the policy:
no segmentation mask, no part pose, no CAD model.

    c = Cameras(xml_with_meshes)
    views = c.capture(state)      # {"global": dict(rgb, depth, K, cam_pos, cam_R), "global2": ..., "wrist": ...}
"""
from __future__ import annotations

import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from assembly_policy.sim.env import STATE_SPEC

GLOBAL_CAM = dict(pos=(0.80, -0.42, 0.52), lookat=(0.50, 0.0, 0.02), fovy=34.0)   # front-right, the board fills the frame
GLOBAL_CAM2 = dict(pos=(0.80, 0.42, 0.52), lookat=(0.50, 0.0, 0.02), fovy=34.0)   # front-left: sees what the hand hides from the right
WRIST_CAM = dict(pos="0.05 0 0", quat="0 0.707108 0.707108 0", fovy="75")           # robosuite eye_in_hand
W, H = (int(v) for v in os.environ.get("CAP_CAM_RES", "1280x960").split("x"))            # fixed camera
WW, WH = (int(v) for v in os.environ.get("CAP_WRIST_RES", "640x480").split("x"))         # wrist camera
_RENDERERS = {}        # model hash -> (model, data, renderers): one EGL context per model per process
# An EGL context can be current in one thread only, so every GL call of the process (renderer creation, capture,
# video rendering) runs on this single thread; other threads wait for their result.
_RENDER_THREAD = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")


def on_render_thread(fn, *args, **kwargs):
    """Run fn on the process's render thread and return its result (re-raises its exception)."""
    if threading.current_thread().name.startswith("render"):
        return fn(*args, **kwargs)
    return _RENDER_THREAD.submit(fn, *args, **kwargs).result()


def add_cameras(xml: str) -> str:
    root = ET.fromstring(xml)
    hand = root.find(".//body[@name='hand']")
    if hand.find("camera[@name='wrist']") is None:
        ET.SubElement(hand, "camera", name="wrist", mode="fixed", pos=WRIST_CAM["pos"], quat=WRIST_CAM["quat"],
                      fovy=WRIST_CAM["fovy"])
    world = root.find("worldbody")
    for name, cam in (("global", GLOBAL_CAM), ("global2", GLOBAL_CAM2)):
        if world.find(f"camera[@name='{name}']") is not None:
            continue
        pos, look = np.array(cam["pos"]), np.array(cam["lookat"])
        z = pos - look; z /= np.linalg.norm(z)                 # a MuJoCo camera looks along its -z
        x = np.cross([0.0, 0.0, 1.0], z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        ET.SubElement(world, "camera", name=name, mode="fixed", pos=" ".join(f"{v:.6f}" for v in pos),
                      xyaxes=" ".join(f"{v:.6f}" for v in np.r_[x, y]), fovy=str(cam["fovy"]))
    g = root.find("visual/global")
    if g is None:
        vis = root.find("visual") if root.find("visual") is not None else ET.SubElement(root, "visual")
        g = ET.SubElement(vis, "global")
    g.set("offwidth", str(max(int(g.get("offwidth", "0")), W))); g.set("offheight", str(max(int(g.get("offheight", "0")), H)))
    return ET.tostring(root, encoding="unicode")


class Cameras:
    def __init__(self, xml: str, w=W, h=H):
        key = hashlib.sha1(xml.encode()).hexdigest()

        def build():
            if key not in _RENDERERS:
                m = mujoco.MjModel.from_xml_string(add_cameras(xml))
                _RENDERERS[key] = (m, mujoco.MjData(m), {"global": mujoco.Renderer(m, h, w), "global2": mujoco.Renderer(m, h, w),
                                                         "wrist": mujoco.Renderer(m, WH, WW)})
        on_render_thread(build)
        self.m, self.d, self.rs = _RENDERERS[key]
        self.cams = {n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n) for n in ("global", "global2", "wrist")}

    def _view(self, name, cam_id):
        r = self.rs[name]
        r.update_scene(self.d, camera=cam_id)
        rgb = r.render().copy()
        r.enable_depth_rendering(); depth = r.render().copy(); r.disable_depth_rendering()
        h, w = r.height, r.width
        fy = (h / 2) / math.tan(math.radians(float(self.m.cam_fovy[cam_id])) / 2)
        K = np.array([[fy, 0.0, w / 2], [0.0, fy, h / 2], [0.0, 0.0, 1.0]])
        return dict(rgb=rgb, depth=depth.astype(np.float32), K=K,
                    cam_pos=self.d.cam_xpos[cam_id].copy(), cam_R=self.d.cam_xmat[cam_id].reshape(3, 3).copy())

    def capture(self, state):
        """The three cameras at this physics state. depth in metres (z along the optical axis); the camera frame has
        x right, y up, z backwards (MuJoCo): world = cam_pos + cam_R @ [(u+.5-cx)/f*d, -(v+.5-cy)/f*d, -d]."""
        def cap():
            mujoco.mj_setState(self.m, self.d, state, STATE_SPEC)
            mujoco.mj_forward(self.m, self.d)
            return {n: self._view(n, c) for n, c in self.cams.items()}
        return on_render_thread(cap)
