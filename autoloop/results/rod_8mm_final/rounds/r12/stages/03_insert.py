# insert: the rod is held semi-rigidly (exp04 s00: it follows a slow hand rotation). Lift high, translate
# so the rod centre is where it will be, rotate in small IK-chain steps about the rod centre until the rod
# is vertical (axis-angle interpolation; chain checked against joint limits / IK branch, otherwise the next
# yaw candidate), go above the bore, closed-loop camera alignment of tilt + tip position, step down,
# release inside the bore, lift; verify.
use_meas = (EB_MEAS is not None and np.linalg.norm(EB_MEAS - EB_PRED) < 0.010
            and np.linalg.norm(EC_MEAS - EC_PRED) < 0.014)
EB_USE, EC_USE = (EB_MEAS, EC_MEAS) if use_meas else (EB_PRED, EC_PRED)
print("rod-in-hand model:", "measured" if use_meas else "predicted")
_lim = np.asarray(api.constants()["joint_limits"], float)


def log_rot(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-6:
        return np.zeros(3)
    if th > np.pi - 1e-4:
        k = unit(np.sqrt(np.clip((np.diag(R) + 1) / 2, 0, 1)))
        return k * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th


def exp_rot(v):
    th = np.linalg.norm(v)
    if th < 1e-9:
        return np.eye(3)
    k = v / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def chain_for(plan, R_0, q_start):
    R_ins = plan["poses"]["high"][1]
    c_t = (plan["Eb"] + plan["Ec"]) / 2
    C_W = plan["poses"]["high"][0] + R_ins @ c_t
    dv = log_rot(R_0.T @ R_ins)
    n = max(4, int(np.ceil(np.linalg.norm(dv) / np.radians(11))))
    # start pose: same orientation, rod centre at C_W
    q, reach = api.ik(C_W - R_0 @ c_t, mat_to_quat(R_0), q_init=q_start)
    if not reach:
        return None
    q_prev = np.asarray(q, float)
    wps = [q_prev]
    for kk in range(1, n + 1):
        Rk = R_0 @ exp_rot(dv * kk / n)
        q, reach = api.ik(C_W - Rk @ c_t, mat_to_quat(Rk), q_init=q_prev)
        q = np.asarray(q, float)
        if (not reach or np.abs(q - q_prev).max() > 0.6
                or np.min(np.minimum(q - _lim[:, 0], _lim[:, 1] - q)) < 0.03):
            return None
        wps.append(q)
        q_prev = q
    if np.abs(wps[-1] - plan["poses"]["high"][2]).max() > 0.3:
        return None
    return dict(C_W=C_W, wps=wps, deg=np.degrees(np.linalg.norm(dv)))


T0 = tcp_T()
servo(T0[:3, 3] + EZ * max(0.0, 0.20 - T0[2, 3]), T0[:3, :3], n=60, k=0.2)     # lift to 20 cm
T0 = tcp_T()
q_now = np.asarray(api.observe()["joints"], float)
plans = plan_insert(EB_USE, EC_USE, q_now)
if not plans and use_meas:
    EB_USE, EC_USE = EB_PRED, EC_PRED
    plans = plan_insert(EB_USE, EC_USE, q_now)
plan = chain = None
for pl in plans[:8]:
    ch = chain_for(pl, T0[:3, :3], q_now)
    print(f"  cand psi {np.degrees(pl['psi']):.0f} cost {pl['cost']:.2f} chain {'ok' if ch else 'bad'}")
    if ch:
        plan, chain = pl, ch
        break
if plan is None:
    stage_done(False, "no reachable vertical insertion pose with a valid rotation chain")
R_ins = plan["poses"]["high"][1]
print("insert psi", np.degrees(plan["psi"]), "tcp z dir", np.round(R_ins[:, 2], 3), f"rotate {chain['deg']:.0f} deg")
api.move_to_joints(chain["wps"][0], timeout_s=4.0, gain=0.3)
for kk, q in enumerate(chain["wps"][1:]):
    last = kk == len(chain["wps"]) - 2
    api.move_to_joints(q, timeout_s=2.0 if last else 1.2, tol=0.01 if last else 0.04, gain=0.35)
print(f"after rotation: hand orientation off the plan by {np.degrees(np.linalg.norm(log_rot(tcp_T()[:3, :3].T @ R_ins))):.1f} deg")
api.move_to_joints(plan["poses"]["pre"][2], timeout_s=3.0, gain=0.4)


def measure_hanging():
    """Rod bottom point and downward axis from the camera fit, or None if the rod is not in the hand."""
    f = fit_rod()
    if f is None:
        return None
    T = tcp_T()
    lo = f["p0"] if f["p0"][2] < f["p1"][2] else f["p1"]
    m0, m1 = rod_ends(f, anchor_far_from=lo + EZ * 1.0)
    mb, mc = (m0, m1) if m0[2] < m1[2] else (m1, m0)
    if np.linalg.norm((mb + mc) / 2 - T[:3, 3]) > 0.06:
        return None
    return dict(b=mb, a=unit(mb - mc), T=T, f=f)


def align(z_bottom, iters=4, tag=""):
    """Rotate/translate the TCP so the rod axis is vertical and its bottom is at (goal xy, z_bottom)."""
    m = None
    for it in range(iters):
        api.wait(0.2)
        m = measure_hanging()
        if m is None:
            print(f"align{tag} {it}: rod not seen in hand")
            return None
        tilt = np.degrees(np.arccos(np.clip(-m["a"][2], -1, 1)))
        err = np.array([GB[0], GB[1], z_bottom]) - m["b"]
        print(f"align{tag} {it}: bottom {np.round(m['b'], 4)} tilt {tilt:.2f}deg xy err {np.round(err[:2]*1000, 2)}mm "
              f"z err {err[2]*1000:.1f}mm ext {m['f']['ext']*1000:.0f} rms {m['f']['rms']*1000:.2f}")
        if np.linalg.norm(err[:2]) < 0.0004 and tilt < 0.6 and abs(err[2]) < 0.002:
            return m
        if np.linalg.norm(err) > 0.03 or tilt > 20:
            print("align: error too large, not trusted")
            return None
        Rc = rot_a_to_b(m["a"], -EZ)
        p_new = (m["b"] + err) + Rc @ (m["T"][:3, 3] - m["b"])
        servo(p_new, Rc @ m["T"][:3, :3], n=30, k=0.4)
    api.wait(0.2)
    m2 = measure_hanging()
    return m2 if m2 is not None else m


m = align(Z_TOP + 0.015, iters=5)
if m is None:
    stage_done(False, "rod not in hand or not near vertical above the fixture")

Z_REL = max(GB[2] + 0.004, Z_TOP - 0.007)        # release height of the rod bottom (7 mm inside the bore)
z_targets = [Z_TOP + 0.004] + list(np.arange(Z_TOP - 0.002, Z_REL - 1e-4, -0.003)) + [Z_REL]
retries = 0
i = 0
b_now = m["b"].copy()
while i < len(z_targets):
    zb = z_targets[i]
    T = tcp_T()
    tgt = T[:3, 3] + EZ * (zb - b_now[2])
    servo(tgt, T[:3, :3], n=20, k=0.4)
    o = api.observe()
    short = o["tcp_pos"][2] - tgt[2]
    print(f"step bottom->{zb:.4f}: tcp short by {short*1000:.1f}mm F/T={np.round(o['tcp_wrench_ext'], 1)}")
    b_now = b_now + EZ * (zb - b_now[2]) + EZ * short
    if short > 0.0015:
        retries += 1
        if retries > 2:
            break
        T = tcp_T()
        servo(T[:3, 3] + EZ * (Z_TOP + 0.008 - b_now[2]), T[:3, :3], n=25, k=0.4)
        mm = align(Z_TOP + 0.008, iters=3, tag="R")
        if mm is None:
            break
        b_now = mm["b"].copy()
        i = 0
        continue
    i += 1
print("release at rod bottom ~", round(float(b_now[2]), 4))
T = tcp_T()
servo(T[:3, 3], T[:3, :3], n=12, grip=1.0, k=0.4, min_n=12)     # open while holding the pose
servo(T[:3, 3] + EZ * 0.08, T[:3, :3], n=30, k=0.4)
api.wait(0.3)

f = fit_rod()
ok = False
ev = "no mask fit"
if f is not None:
    top = f["p1"] if f["p1"][2] > f["p0"][2] else f["p0"]
    a = unit(f["p1"] - f["p0"])
    lam = (GT[2] - top[2]) / a[2] if abs(a[2]) > 0.5 else 0.0
    xy = np.linalg.norm((top + lam * a - GT)[:2])
    dz = top[2] - GT[2]
    ok = abs(a[2]) > 0.98 and xy < 0.002 and abs(dz) < 0.0025
    ev = f"mask fit: axis z {abs(a[2]):.3f}, top z {top[2]:.4f} vs goal {GT[2]:.4f} ({dz*1000:+.1f}mm), axis xy off {xy*1000:.1f}mm"
print(ev)
stage_done(ok, ev)
