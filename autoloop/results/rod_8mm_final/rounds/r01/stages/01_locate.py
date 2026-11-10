# locate: find the rod on the board with the tracker, cross-check with own depth/mask vision.
from assembly_policy.sim.robot import quat_to_mat, mat_to_quat, quat_mul, quat_conj, axis_angle_quat
np.set_printoptions(precision=4, suppress=True)

_M = api.part_model()
PB0 = np.asarray(_M["bbox_min"], float)
PB1 = np.asarray(_M["bbox_max"], float)
_ext = PB1 - PB0
AX = int(np.argmax(_ext))                      # long axis of the rod in the part frame
_mid = (PB0 + PB1) / 2
E0 = _mid.copy(); E0[AX] = PB0[AX]             # the two axis end points in the part frame
E1 = _mid.copy(); E1[AX] = PB1[AX]
L = float(_ext[AX])
RAD = float(np.delete(_ext, AX).max() / 2)
EZ = np.array([0.0, 0.0, 1.0])


def Tm(p, R):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p
    return M


def Tinv(M):
    R = M[:3, :3]; p = M[:3, 3]
    return Tm(-R.T @ p, R.T)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def tcp_T():
    o = api.observe()
    return Tm(np.asarray(o["tcp_pos"], float), quat_to_mat(np.asarray(o["tcp_quat"], float)))


def fresh_est(tries=8):
    p = q = c = None
    for _ in range(tries):
        p, q, c = api.locate_part()
        if p is not None and c.get("stale", 1) == 0:
            break
        api.wait(0.04)
    if p is None:
        return None, None, c
    return np.asarray(p, float), np.asarray(q, float), c


def ends_world(p, q):
    R = quat_to_mat(q)
    return p + R @ E0, p + R @ E1


def mask_cloud(names=("global", "global2")):
    cams = api.camera()
    pts = []
    for n in names:
        c = cams[n]
        vs, us = np.nonzero(c["mask"])
        if len(vs) == 0:
            continue
        d = c["depth"][vs, us]
        K = np.asarray(c["K"], float)
        P = np.stack([(us + .5 - K[0, 2]) / K[0, 0] * d, -(vs + .5 - K[1, 2]) / K[1, 1] * d, -d], 1)
        pts.append(np.asarray(c["cam_pos"], float) + P @ np.asarray(c["cam_R"], float).T)
    return np.concatenate(pts) if pts else np.zeros((0, 3))


OBS0 = api.observe()
GOAL_P = np.asarray(OBS0["goal_pos"], float)
GOAL_R = quat_to_mat(np.asarray(OBS0["goal_quat"], float))
_g0, _g1 = GOAL_P + GOAL_R @ E0, GOAL_P + GOAL_R @ E1
GB, GT = (_g0, _g1) if _g0[2] <= _g1[2] else (_g1, _g0)   # goal bottom / top axis points
print("goal bottom", GB, "top", GT, "L", L, "RAD", RAD)

P_EST, Q_EST, C_EST = fresh_est()
ok = False
ev = "no tracker estimate"
if P_EST is not None:
    e0, e1 = ends_world(P_EST, Q_EST)
    ctr = (e0 + e1) / 2
    ax_w = unit(e1 - e0)
    cloud = mask_cloud()
    dc = np.linalg.norm(cloud.mean(0)[:2] - ctr[:2]) if len(cloud) else 1.0
    ok = (abs(ax_w[2]) < 0.35 and 0.0 < ctr[2] < 0.04 and len(cloud) > 100 and dc < 0.015)
    ev = (f"tracker ctr={np.round(ctr, 4).tolist()} axis_z={ax_w[2]:.2f} cam={C_EST.get('cam')} "
          f"mask_pts={len(cloud)} mask-vs-tracker xy={dc * 1000:.1f}mm")
print(ev)
stage_done(ok, ev)
