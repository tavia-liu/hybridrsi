# locate: find the rod with own vision (mask + depth of the two global cameras, cylinder fit);
# the tracker is only printed for comparison (r01: it goes stale / wrong when the hand occludes the rod).
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


def tracker():
    p, q, c = api.locate_part()
    if p is None:
        return None
    p = np.asarray(p, float); R = quat_to_mat(np.asarray(q, float))
    return p + R @ E0, p + R @ E1, c


def backproject(c, sel):
    vs, us = np.nonzero(sel)
    d = c["depth"][vs, us]
    K = np.asarray(c["K"], float)
    P = np.stack([(us + .5 - K[0, 2]) / K[0, 0] * d, -(vs + .5 - K[1, 2]) / K[1, 1] * d, -d], 1)
    return np.asarray(c["cam_pos"], float) + P @ np.asarray(c["cam_R"], float).T


def fit_rod(names=("global", "global2"), cams=None):
    """Cylinder fit (known radius) to the part-mask depth points. Returns dict with axis dir a,
    axis point, visible end points p0/p1 (along a), extent, rms, n."""
    cams = cams if cams is not None else api.camera()
    P, C = [], []
    for n in names:
        c = cams[n]
        m = np.asarray(c["mask"], bool)
        # drop mask boundary pixels (depth may belong to the background)
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]; inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]; inner[:, :-1] &= m[:, 1:]
        if inner.sum() < 30:
            continue
        pts = backproject(c, inner)
        P.append(pts); C.append(np.repeat(np.asarray(c["cam_pos"], float)[None], len(pts), 0))
    if not P:
        return None
    P = np.concatenate(P); C = np.concatenate(C)
    for _ in range(2):                                  # robust line
        mu = np.median(P, 0)
        a = np.linalg.svd(P - mu, full_matrices=False)[2][0]
        dperp = np.linalg.norm((P - mu) - np.outer((P - mu) @ a, a), axis=1)
        keep = dperp < 0.012
        P, C = P[keep], C[keep]
        if len(P) < 30:
            return None
    mu = P.mean(0)
    a = np.linalg.svd(P - mu, full_matrices=False)[2][0]
    b1 = unit(np.cross(a, EZ) if abs(a[2]) < 0.9 else np.cross(a, [1.0, 0, 0]))
    b2 = np.cross(a, b1)
    X = np.stack([(P - mu) @ b1, (P - mu) @ b2], 1)
    vc = (C - P).mean(0)
    vc2 = unit(np.array([vc @ b1, vc @ b2]))
    cc = X.mean(0) - vc2 * RAD * 0.75
    w = np.ones(len(X), bool)
    for it in range(25):
        d = X[w] - cc
        r = np.linalg.norm(d, axis=1) + 1e-9
        J = -d / r[:, None]
        cc = cc + np.linalg.lstsq(J, -(r - RAD), rcond=None)[0]
        if it == 12:
            w = np.abs(np.linalg.norm(X - cc, axis=1) - RAD) < 0.0015
    res = np.abs(np.linalg.norm(X[w] - cc, axis=1) - RAD)
    ap = mu + cc[0] * b1 + cc[1] * b2
    t = (P[w] - ap) @ a
    t0, t1 = np.percentile(t, 0.3), np.percentile(t, 99.7)
    return dict(a=a, ap=ap, p0=ap + t0 * a, p1=ap + t1 * a, ext=float(t1 - t0),
                rms=float(np.sqrt((res ** 2).mean())), n=int(w.sum()))


def rod_ends(f, anchor_far_from=None):
    """Both rod ends (length L) from a fit; if the visible extent is short, anchor on the visible end
    farther from `anchor_far_from` (the other end is hidden by the fingers)."""
    if f["ext"] > L - 0.004 or anchor_far_from is None:
        c = (f["p0"] + f["p1"]) / 2
        return c - f["a"] * L / 2, c + f["a"] * L / 2
    d0 = np.linalg.norm(f["p0"] - anchor_far_from); d1 = np.linalg.norm(f["p1"] - anchor_far_from)
    if d0 > d1:
        return f["p0"], f["p0"] + f["a"] * L
    return f["p1"] - f["a"] * L, f["p1"]


def heightmap_points(cams, center, radius=0.15):
    pts = []
    for n in ("global", "global2"):
        c = cams[n]
        sel = ~np.asarray(c["mask"], bool) & (c["depth"] > 0)
        sel[::2, :] = False; sel[:, ::2] = False
        P = backproject(c, sel)
        P = P[np.linalg.norm(P[:, :2] - center[:2], axis=1) < radius]
        pts.append(P)
    return np.concatenate(pts)


OBS0 = api.observe()
GOAL_P = np.asarray(OBS0["goal_pos"], float)
GOAL_R = quat_to_mat(np.asarray(OBS0["goal_quat"], float))
_g0, _g1 = GOAL_P + GOAL_R @ E0, GOAL_P + GOAL_R @ E1
GB, GT = (_g0, _g1) if _g0[2] <= _g1[2] else (_g1, _g0)   # goal bottom / top axis points
print("goal bottom", GB, "top", GT, "L", L, "RAD", RAD)

CAMS0 = api.camera()
FIT0 = fit_rod(cams=CAMS0)
tr = tracker()
ok = False
ev = "no mask fit"
if FIT0 is not None:
    R0E0, R0E1 = rod_ends(FIT0)
    ROD_CTR0 = (R0E0 + R0E1) / 2
    BOARD_Z = ROD_CTR0[2] - RAD
    HM = heightmap_points(CAMS0, ROD_CTR0)
    dtr = np.linalg.norm(((tr[0] + tr[1]) / 2 - ROD_CTR0)[:2]) * 1000 if tr else -1
    ok = (abs(FIT0["a"][2]) < 0.3 and FIT0["rms"] < 0.0012 and FIT0["n"] > 300
          and abs(FIT0["ext"] - L) < 0.006 and 0.0 < ROD_CTR0[2] < 0.04)
    ev = (f"fit ctr={np.round(ROD_CTR0, 4).tolist()} axis={np.round(FIT0['a'], 2).tolist()} ext={FIT0['ext']*1000:.1f}mm "
          f"rms={FIT0['rms']*1000:.2f}mm n={FIT0['n']} tracker-vs-fit xy={dtr:.1f}mm")
print(ev)
stage_done(ok, ev)
