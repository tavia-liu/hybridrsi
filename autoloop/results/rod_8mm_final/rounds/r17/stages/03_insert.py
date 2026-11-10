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
    n = max(4, int(np.ceil(np.linalg.norm(dv) / np.radians(15))))   # r12: 16 steps took 13 s
    # start pose: same orientation, rod centre at C_W
    q, reach = api.ik(C_W - R_0 @ c_t, mat_to_quat(R_0), q_init=q_start)
    if not reach:
        return "start unreachable"
    q_prev = np.asarray(q, float)
    wps = [q_prev]
    for kk in range(1, n + 1):
        Rk = R_0 @ exp_rot(dv * kk / n)
        q, reach = api.ik(C_W - Rk @ c_t, mat_to_quat(Rk), q_init=q_prev)
        q = np.asarray(q, float)
        if not reach:
            return f"step {kk} unreachable"
        if np.abs(q - q_prev).max() > 0.6:
            return f"step {kk} jump {np.abs(q - q_prev).max():.2f}"
        if np.min(np.minimum(q - _lim[:, 0], _lim[:, 1] - q)) < 0.03:
            return f"step {kk} at joint limit"
        wps.append(q)
        q_prev = q
    # r14: the plan's own IK branch may differ from the chain's end; re-solve pre (and check fin) from
    # the chain end instead of rejecting the chain
    q_pre, r1 = api.ik(plan["poses"]["pre"][0], mat_to_quat(R_ins), q_init=wps[-1])
    q_fin, r2 = api.ik(plan["poses"]["fin"][0], mat_to_quat(R_ins), q_init=np.asarray(q_pre, float))
    q_pre = np.asarray(q_pre, float); q_fin = np.asarray(q_fin, float)
    if not (r1 and r2) or np.abs(q_pre - wps[-1]).max() > 0.8 or np.abs(q_fin - q_pre).max() > 0.8 \
            or np.min(np.minimum(q_fin - _lim[:, 0], _lim[:, 1] - q_fin)) < 0.03:
        return "pre/fin not reachable from the chain end"
    return dict(C_W=C_W, wps=wps, q_pre=q_pre, deg=np.degrees(np.linalg.norm(dv)))


T0 = tcp_T()
servo(T0[:3, 3] + EZ * max(0.0, 0.20 - T0[2, 3]), T0[:3, :3], n=60, k=0.2)     # lift to 20 cm
T0 = tcp_T()
q_now = np.asarray(api.observe()["joints"], float)
plans = plan_insert(EB_USE, EC_USE, q_now)
if not plans and use_meas:
    EB_USE, EC_USE = EB_PRED, EC_PRED
    plans = plan_insert(EB_USE, EC_USE, q_now)
plan = chain = None
for pl in plans[:12]:
    ch = chain_for(pl, T0[:3, :3], q_now)
    print(f"  cand psi {np.degrees(pl['psi']):.0f} cost {pl['cost']:.2f} hand hits {pl['hits']} chain "
          f"{'ok' if isinstance(ch, dict) else ch}")
    if isinstance(ch, dict):
        plan, chain = pl, ch
        break
if plan is None:
    stage_done(False, "no reachable vertical insertion pose with a valid rotation chain")
R_ins = plan["poses"]["high"][1]
print("insert psi", np.degrees(plan["psi"]), "tcp z dir", np.round(R_ins[:, 2], 3), f"rotate {chain['deg']:.0f} deg")
G_CHAIN = 0.28 if GRIP_DEPTH > 0.0055 else 0.45
print(f"grip depth {GRIP_DEPTH*1000:.1f} mm -> chain gain {G_CHAIN}")
api.move_to_joints(chain["wps"][0], timeout_s=4.0, gain=0.3)
for kk, q in enumerate(chain["wps"][1:]):
    last = kk == len(chain["wps"]) - 2
    # r15 s02: a fingertip grip (rod centre > 5.5 mm along TCP z) let go early in the chain -> slower
    api.move_to_joints(q, timeout_s=2.0 if last else 0.9, tol=0.01 if last else 0.06, gain=G_CHAIN)
print(f"after rotation: hand orientation off the plan by {np.degrees(np.linalg.norm(log_rot(tcp_T()[:3, :3].T @ R_ins))):.1f} deg")
api.move_to_joints(chain["q_pre"], timeout_s=3.0, gain=0.4)


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


def top_rel():
    """Height of the highest visible rod point above the TCP (rises if the rod slides up in the grip)."""
    f = fit_rod()
    if f is None:
        return None
    return float(max(f["p0"][2], f["p1"][2]) - tcp_T()[2, 3])


XY_OFF = [np.zeros(2), np.array([0.0005, 0.0]), np.array([0.0, 0.0005]), np.array([-0.0005, 0.0])]


def align(z_bottom, iters=4, tag="", xy_off=np.zeros(2)):
    """Rotate/translate the TCP so the rod axis is vertical and its bottom is at (goal xy, z_bottom)."""
    m = None
    for it in range(iters):
        api.wait(0.2)
        m = measure_hanging()
        if m is None:
            print(f"align{tag} {it}: rod not seen in hand")
            return None
        tilt = np.degrees(np.arccos(np.clip(-m["a"][2], -1, 1)))
        err = np.array([GB[0] + xy_off[0], GB[1] + xy_off[1], z_bottom]) - m["b"]
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
TOP0 = top_rel()
BREL0 = b_now[2] - tcp_T()[2, 3]
spiral_done = False
print("rod top above TCP before the descent:", None if TOP0 is None else round(TOP0 * 1000, 1), "mm")


def spiral_entry():
    """Lateral spiral (rings 0.4/0.8/1.2/1.6 mm) at the rim height, pressing lightly at every point.
    Entered = TCP dropped > 2 mm below the start and the rod did not just slide up in the grip."""
    T0s = tcp_T()
    p0 = T0s[:3, 3].copy(); Rs = T0s[:3, :3]
    tc = top_rel()
    for r, n_a in ((0.0004, 6), (0.0008, 8), (0.0012, 10)):
        for ang in np.linspace(0, 2 * np.pi, n_a, endpoint=False) + r * 1000:
            d = np.array([r * np.cos(ang), r * np.sin(ang), 0.0])
            servo(p0 + d + EZ * 0.0008, Rs, n=4, k=0.6)
            servo(p0 + d - EZ * 0.004, Rs, n=5, k=0.3)
            z = float(api.observe()["tcp_pos"][2])
            if z < p0[2] - 0.002:
                tr = top_rel()
                sl = 0.0 if (tr is None or tc is None) else tr - tc
                print(f"    spiral r {r*1000:.1f}mm ang {np.degrees(ang) % 360:.0f}: tcp drop {(p0[2]-z)*1000:.1f}mm slip {sl*1000:.1f}mm")
                if (p0[2] - z) - sl > 0.0015:
                    servo(p0 + d - EZ * 0.004, Rs, n=10, k=0.4)
                    return True
                tc = tr if tr is not None else tc
                servo(p0 + d + EZ * 0.0008, Rs, n=6, k=0.5)
    servo(p0 + EZ * 0.0008, Rs, n=10, k=0.5)
    return False
while i < len(z_targets):
    zb = z_targets[i]
    T = tcp_T()
    tgt = T[:3, 3] + EZ * (zb - b_now[2])
    servo(tgt, T[:3, :3], n=20, k=0.4)
    o = api.observe()
    short = o["tcp_pos"][2] - tgt[2]
    # r15 s04: with a fingertip grip the rod slid UP in the fingers when its tip sat on the rim, so the
    # TCP was never short; watch the rod's top relative to the TCP once the tip should be in the bore
    slip = 0.0
    if zb < Z_TOP - 0.001 and TOP0 is not None:
        tr = top_rel()
        slip = 0.0 if tr is None else tr - TOP0
    print(f"step bottom->{zb:.4f}: tcp short by {short*1000:.1f}mm, rod slid up {slip*1000:.1f}mm, F={np.round(o['tcp_wrench_ext'][:3], 1)}")
    b_now = b_now + EZ * (zb - b_now[2]) + EZ * short
    if (short > 0.0015 or slip > 0.0015) and zb < Z_TOP and not spiral_done:
        # r16 s00: aligned to 0.1-0.3 mm by the camera, yet blocked at the fixture top 4 times in a row
        # (a camera bias larger than the bore clearance). Search for the bore entry: spiral around the
        # current xy, pressing lightly; the bore is found when the rod BOTTOM drops (TCP drop minus slip).
        spiral_done = True
        T = tcp_T()
        servo(T[:3, 3] + EZ * (max(slip, 0.0) + 0.0008), T[:3, :3], n=10, k=0.5)
        found = spiral_entry()
        if found:
            T = tcp_T()
            tr = top_rel()
            b_now = np.array([GB[0], GB[1], T[2, 3] + BREL0 + ((tr - TOP0) if (tr is not None and TOP0 is not None) else 0.0)])
            TOP0 = tr if tr is not None else TOP0
            BREL0 = b_now[2] - T[2, 3]
            print(f"  spiral: entered, rod bottom ~{b_now[2]:.4f}")
            z_targets = list(np.arange(b_now[2] - 0.003, Z_REL - 1e-4, -0.003)) + [Z_REL]
            i = 0
            continue
        print("  spiral: no entry found")
    if short > 0.0015 or slip > 0.0015:
        retries += 1
        if retries > 3:
            break
        T = tcp_T()
        servo(T[:3, 3] + EZ * (Z_TOP + 0.010 - b_now[2] + max(slip, 0.0)), T[:3, :3], n=25, k=0.4)
        mm = align(Z_TOP + 0.008, iters=3, tag="R", xy_off=XY_OFF[retries % len(XY_OFF)])
        TOP0 = top_rel()
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
api.wait(1.0)      # r14 s04: the verifier looked while the released rod was still sliding down the bore

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
