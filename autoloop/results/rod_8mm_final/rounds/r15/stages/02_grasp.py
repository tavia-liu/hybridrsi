# grasp: top-down centre pinch of the lying rod (fingers pre-opened narrow, grasp point checked against
# neighbouring obstacles in the depth heightmap), feel down until the fingertips touch the board, close
# with a pose-holding Jacobian servo; if the close ends far below the rod diameter (rod under the pad edge,
# exp04 s01) re-open a little and close again; lift; verify with the camera fit; retry after re-locating.
TCP_ABOVE_BOARD = 0.0110   # start height of the feel-down
TIP_REACH = 0.0089         # fingertips touch the board with the TCP ~8.8-9.2 mm above it (exp02/03)
W_PRE = 0.03               # pre-grasp opening
H_PRE = (Z_TOP - GB[2]) + 0.015   # rod bottom 15 mm above the fixture top before the descent
H_HIGH = 0.12
ASCALE = np.asarray(api.constants()["action_scale"], float)
R_GRASP_REF = np.eye(3)
HM_OBS = HM[HM[:, 2] > BOARD_Z + 0.003]          # obstacles (not the board, not the rod)
HMG_OBS = HM_G[HM_G[:, 2] > BOARD_Z + 0.003]


def rot_a_to_b(a, b):
    a, b = unit(a), unit(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(a @ b)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        p = unit(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(p, p) - np.eye(3)
    k = v / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def servo(pos, R, n=20, grip=0.0, k=0.5, min_n=0, tol=0.0003):
    """Cartesian servo of the TCP with the Jacobian, integrating its joint target (no steady-state
    offset under gravity, r07). grip: action[7] (-1 close, +1 open, 0 keep)."""
    o = api.observe()
    q_t = np.asarray(o["joints"], float).copy()
    for i in range(n):
        o = api.observe()
        e = np.asarray(pos, float) - np.asarray(o["tcp_pos"], float)
        Rc = quat_to_mat(np.asarray(o["tcp_quat"], float))
        w = 0.5 * sum(np.cross(Rc[:, j], R[:, j]) for j in range(3))
        if i >= min_n and np.linalg.norm(e) < tol and np.linalg.norm(w) < 0.004:
            break
        en = np.linalg.norm(e)
        if en > 0.01:
            e = e / en * 0.01
        J = np.asarray(o["tcp_jacobian"], float)
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), np.concatenate([k * e, k * w]))
        q_m = np.asarray(o["joints"], float)
        q_t = q_m + np.clip(q_t + dq - q_m, -0.6 * ASCALE, 0.6 * ASCALE)     # anti-windup
        api.step(np.concatenate([np.clip((q_t - q_m) / ASCALE, -1, 1), [grip]]))
    return o


def plan_insert(Eb, Ec, q_seed, verbose=False):
    """Eb/Ec = rod bottom/top axis points in the TCP frame. Rod vertical, bottom at GB(+dz), yaw about the
    vertical swept. Returns all reachable candidates sorted by cost (cheapest first)."""
    Rb = rot_a_to_b(Eb - Ec, -EZ)
    out = []
    lim = np.asarray(api.constants()["joint_limits"], float)
    radial = unit(np.array([GB[0], GB[1], 0.0]))
    for psi in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        R = rotz(psi) @ Rb
        poses = {}
        q_prev = q_seed
        for name, dz in (("high", H_HIGH), ("pre", H_PRE), ("fin", 0.0)):
            pos = GB + EZ * dz - R @ Eb
            q, reach = api.ik(pos, mat_to_quat(R), q_init=q_prev)
            q = np.asarray(q, float)
            if not reach or np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q)) < 0.05:
                break
            poses[name] = (pos, R, q)
            q_prev = q
        if len(poses) < 3:
            continue
        # hand clearance at the release pose (rod bottom 7 mm inside the bore) against the fixture area
        z_rel = max(GB[2] + 0.004, Z_TOP - 0.007)
        hf, hb = hand_hits(GB + EZ * (z_rel - GB[2]) - R @ Eb, R, HMG_OBS, half_open=RAD)
        cost = (50.0 if hf + hb > 10 else 0.0) + 0.3 * min(hf + hb, 100) +np.abs(poses["high"][2] - q_seed).sum() + 3 * np.abs(poses["fin"][2] - poses["high"][2]).sum()
        zdir = unit(np.array([R[0, 2], R[1, 2], 0.0]))
        cost += 8.0 * (1.0 - float(zdir @ radial))      # r05: hand between base and fixture
        ang = np.arccos(np.clip((np.trace(R_GRASP_REF.T @ R) - 1) / 2, -1, 1))
        cost += 3.0 * ang                               # r06: smaller hand rotation
        if verbose:
            print(f"  psi={np.degrees(psi):6.1f} cost={cost:.2f} tcp_z_dir={np.round(R[:, 2], 2)}")
        out.append(dict(psi=psi, cost=cost, poses=poses, Eb=Eb, Ec=Ec, hits=(hf, hb)))
    out.sort(key=lambda d: d["cost"])
    return out


def hand_hits(pos, R, P, half_open=0.015):
    """Heightmap points P inside a coarse model of the hand at TCP pose (pos, R): two fingers (TCP z
    -0.045..+0.009, |x| < 12 mm, |y| in [half_open, half_open+12 mm]) and the hand body (z < -0.045,
    up to -0.16; |x| < 40 mm, |y| < 105 mm). r12: the hand body landed on a tall post (s01, grasp) and on
    the part beside the fixture (s00, insertion)."""
    d = P - pos
    dx, dy, dz = d @ R[:, 0], d @ R[:, 1], d @ R[:, 2]
    fing = (dz > -0.045) & (dz < 0.007) & (np.abs(dx) < 0.012) & (np.abs(dy) > half_open) & (np.abs(dy) < half_open + 0.012)
    body = (dz <= -0.045) & (dz > -0.16) & (np.abs(dx) < 0.040) & (np.abs(dy) < 0.105)
    return int(fing.sum()), int(body.sum())


def in_tcp(T, p):
    return T[:3, :3].T @ (p - T[:3, 3])


def finger_clearance(g, R):
    y = R[:, 1]; x = R[:, 0]
    n = 0
    for sgn in (-1, 1):
        d = HM - (g + sgn * y * (W_PRE / 2 + 0.010))
        n += int(((np.abs(d @ x) < 0.014) & (np.abs(d @ y) < 0.012) & (HM[:, 2] > BOARD_Z + 0.003)
                  & (HM[:, 2] < g[2] + 0.06)).sum())
    return n


def choose_grasp(rE0, rE1):
    global R_GRASP_REF
    a_h = rE1 - rE0; a_h[2] = 0; a_h = unit(a_h)
    y_ax = unit(np.cross(EZ, a_h))
    if y_ax @ np.array([0, -1.0, 0]) < 0:
        y_ax = -y_ax
    Rg = np.column_stack([np.cross(y_ax, -EZ), y_ax, -EZ])
    R_GRASP_REF = Rg
    q_now = np.asarray(api.observe()["joints"], float)
    cands = []
    for k, (eA, eB) in enumerate(((rE0, rE1), (rE1, rE0))):    # eB = the end that goes down
        for s in (L / 2, L / 2 - 0.005, L / 2 - 0.010, L / 2 - 0.015):
            g = eA + unit(eB - eA) * s
            g = np.array([g[0], g[1], BOARD_Z + TCP_ABOVE_BOARD])
            Tg = Tm(g, Rg)
            Eb_t = in_tcp(Tg, eB); Ec_t = in_tcp(Tg, eA)
            Eb_t[1] = 0.0; Ec_t[1] = 0.0
            hf, hb = hand_hits(g, Rg, HM_OBS, half_open=W_PRE / 2)
            coll = finger_clearance(g, Rg) + hf + hb
            plans = plan_insert(Eb_t, Ec_t, q_now) if coll < 15 else []
            cost = (plans[0]["cost"] if plans else 1e9) + 60 * (L / 2 - s) + 0.5 * coll
            print(f"grasp cand end{k} s={s:.3f}: finger_pts={coll} plan={'none' if not plans else round(plans[0]['cost'], 2)}")
            cands.append((cost, k, s, g, Eb_t, Ec_t, coll))
    cands.sort(key=lambda c: c[0])
    return Rg, cands[0]


def feel_down_and_close(G, R, dz_extra=0.0):
    servo(G, R, n=30, k=0.5)
    o = api.observe()
    zc = float(o["tcp_pos"][2])
    for _ in range(40):
        o = api.observe()
        if float(o["tcp_wrench_ext"][2]) > 3.0:
            break
        zc = max(zc - 0.0003, BOARD_Z + TIP_REACH - 0.0015)
        servo(np.array([G[0], G[1], zc]), R, n=1, k=1.0)
    o = api.observe()
    g_hold = np.array([G[0], G[1], float(o["tcp_pos"][2]) + 0.0004 + dz_extra])
    print(f"  contact at tcp z {o['tcp_pos'][2]:.4f} Fz {o['tcp_wrench_ext'][2]:.1f} (board {BOARD_Z:.4f}); hold +{dz_extra*1000:.1f}mm")
    servo(g_hold, R, n=8, k=0.5, min_n=6)
    servo(g_hold, R, n=20, grip=-1.0, k=0.4, min_n=20)      # close while holding the pose
    w = float(api.observe()["gripper_width"])
    print(f"  closed: width {w:.4f} F {np.round(api.observe()['tcp_wrench_ext'][:3], 1)}")
    return g_hold, w


ok = False
ev = "not attempted"
EB_MEAS = EC_MEAS = None
rE0, rE1 = R0E0, R0E1
rod_z0 = ROD_CTR0[2]
# exp05 (s01): with the TCP at board contact the closing pads passed THROUGH the rod (inner faces 2.5 mm
# from its axis, rod unmoved, F ~0); earlier s01 runs lifted the rod by the fingertips when the TCP was
# 1-2.5 mm higher (r04-r06, r08). So the variations also raise the hold height.
JITTER = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0025), (0.006, np.radians(8), 0.0012)]
N_ATT = len(JITTER)


def push_clear(rE0, rE1):
    """r14 s03: the rod lay against the fixture, every grasp had > 1000 obstacle points and a finger landed
    on the fixture. Press the closed fingertips on the rod's top and drag/roll it 40 mm away from the
    nearby obstacles, then re-locate it."""
    c = (rE0 + rE1) / 2
    a = rE1 - rE0; a[2] = 0; a = unit(a)
    n = np.array([-a[1], a[0], 0.0])
    near = HM_OBS[(np.linalg.norm(HM_OBS[:, :2] - c[:2], axis=1) < 0.05)]
    side = float(np.sign(np.mean((near[:, :2] - c[:2]) @ n[:2]))) if len(near) else 1.0
    away = -side * n
    Rg = np.column_stack([np.cross(np.cross(EZ, a), -EZ), unit(np.cross(EZ, a)), -EZ])
    if Rg[:, 1] @ np.array([0, -1.0, 0]) < 0:
        Rg = Rg @ np.diag([-1.0, -1.0, 1.0])
    top = np.array([c[0], c[1], c[2] + RAD]) - away * 0.002
    print(f"  push: {len(near)} obstacle pts near the rod, pushing along {np.round(away, 2)}")
    api.move_to(top + EZ * 0.06, mat_to_quat(Rg), timeout_s=3.0)
    servo(top + EZ * 0.03, Rg, n=15, grip=-1.0, k=0.5, min_n=12)       # close the fingers
    zc = float(api.observe()["tcp_pos"][2])
    for _ in range(80):
        o = api.observe()
        if float(o["tcp_wrench_ext"][2]) > 3.0:
            break
        zc -= 0.0005
        servo(np.array([top[0], top[1], zc]), Rg, n=1, k=1.0, grip=-1.0)
    z_press = float(api.observe()["tcp_pos"][2]) - 0.001
    print(f"  push: touched rod top at tcp z {z_press + 0.001:.4f}")
    for kk in range(1, 21):
        p = top + away * 0.002 * kk
        servo(np.array([p[0], p[1], z_press]), Rg, n=2, k=0.8, grip=-1.0)
    servo(np.asarray(api.observe()["tcp_pos"], float) + EZ * 0.05, Rg, n=25, k=0.5, grip=-1.0)
    api.wait(0.1)
    f = fit_rod()
    if f is None or abs(f["a"][2]) > 0.3 or abs(f["ext"] - L) > 0.006:
        print("  push: re-locate failed")
        return rE0, rE1
    e0, e1 = rod_ends(f)
    print("  push: rod now at", np.round((e0 + e1) / 2, 4))
    return e0, e1


_, (_c0, *_rest) = choose_grasp(rE0, rE1)
if _rest[-1] > 40:
    rE0, rE1 = push_clear(rE0, rE1)
    rod_z0 = ((rE0 + rE1) / 2)[2]
for attempt in range(N_ATT):
    R_GRASP, (_, K_SEL, S_SEL, G_PT, EB_PRED, EC_PRED, COLL) = choose_grasp(rE0, rE1)
    # r11: identical retries gave identical failures -> vary the grasp point along the rod and the yaw
    dsh, dyaw, dzx = JITTER[attempt]
    if attempt > 0:
        ax_r = unit(np.array([R_GRASP[0, 0], R_GRASP[1, 0], 0.0]))      # TCP x = along the rod
        G_PT = G_PT + ax_r * dsh
        Rj = rotz(dyaw)
        EB_PRED = EB_PRED - np.array([dsh, 0, 0])      # the close re-aligns the rod across the fingers
        EC_PRED = EC_PRED - np.array([dsh, 0, 0])
        R_GRASP = Rj @ R_GRASP
    Q_GRASP = mat_to_quat(R_GRASP)
    print(f"attempt {attempt}: end {K_SEL} s {S_SEL:.3f} shift {dsh*1000:.0f}mm yaw {np.degrees(dyaw):.0f}deg finger_pts {COLL}")
    api.move_to(G_PT + EZ * (0.08 if attempt == 0 else 0.03), Q_GRASP, timeout_s=3.0)
    api.set_gripper(width=W_PRE, hold_s=0.3)
    api.move_to(G_PT + EZ * 0.015, Q_GRASP, timeout_s=2.0)
    g_hold, W_CLOSED = feel_down_and_close(G_PT, R_GRASP, dzx)
    # exp04/r11: a grip with the rod centre > ~5.5 mm along TCP z (at the rounded tips) pivots or drops
    # later. Check right after a 4 mm lift; if bad, put it back down and retry with a variation.
    servo(g_hold + EZ * 0.004, R_GRASP, n=20, k=0.3)
    api.wait(0.15)
    f = fit_rod()
    T = tcp_T()
    depth_ok = False
    if f is not None:
        cc = (f["p0"] + f["p1"]) / 2
        if f["ext"] < L - 0.004:        # partly hidden by the fingers: anchor on the far visible end
            x0 = in_tcp(T, f["p0"])[0]; x1 = in_tcp(T, f["p1"])[0]
            cc = f["p0"] + f["a"] * L / 2 if abs(x0) > abs(x1) else f["p1"] - f["a"] * L / 2
        ct = in_tcp(T, cc)
        lifted = cc[2] > rod_z0 + 0.002
        # attempt 0 insists on a pads grip; the raised variations accept any lifted, centred grip
        depth_ok = lifted and abs(ct[1]) < 0.003 and (ct[2] < 0.0055 or attempt > 0)
        print(f"  4 mm up: rod ctr z {cc[2]:.4f} (was {rod_z0:.4f}), in TCP frame {np.round(ct*1000, 1).tolist()}mm -> "
              f"{'ok' if depth_ok else 'not held / not in the pads'}")
    if not depth_ok and attempt < N_ATT - 1:
        servo(g_hold, R_GRASP, n=15, k=0.4)
        servo(g_hold, R_GRASP, n=10, grip=1.0, min_n=10)
        servo(g_hold + EZ * 0.03, R_GRASP, n=30, k=0.4)
        api.wait(0.1)
        f = fit_rod()
        if f is not None and abs(f["a"][2]) < 0.3 and abs(f["ext"] - L) < 0.006:
            rE0, rE1 = rod_ends(f)
            rod_z0 = ((rE0 + rE1) / 2)[2]
        ev = f"attempt {attempt}: rod not seated in the pads"
        continue
    servo(g_hold + EZ * 0.07, R_GRASP, n=50, k=0.25)
    api.wait(0.15)

    T_LIFT = tcp_T()
    FIT1 = fit_rod()
    if FIT1 is None:
        ev = "no mask fit after lift"
        break
    tcp = T_LIFT[:3, 3]
    m0, m1 = rod_ends(FIT1)
    if FIT1["ext"] < L - 0.004:
        # r14 s02: only 17 mm visible (hand occludes the rest) -> the visible extreme farther from the
        # TCP along the rod is a true end; the other end is L away
        x0 = in_tcp(T_LIFT, FIT1["p0"])[0]; x1 = in_tcp(T_LIFT, FIT1["p1"])[0]
        if abs(x0) > abs(x1):
            m0 = FIT1["p0"]; m1 = m0 + FIT1["a"] * L
        else:
            m1 = FIT1["p1"]; m0 = m1 - FIT1["a"] * L
    ctr = (m0 + m1) / 2
    c_t = in_tcp(T_LIFT, ctr)
    ok = ctr[2] > rod_z0 + 0.04 and np.linalg.norm(ctr[:2] - tcp[:2]) < 0.03 and FIT1["n"] > 150
    ev = (f"attempt {attempt}: mask fit rod ctr z {rod_z0:.3f}->{ctr[2]:.3f}, rod ctr in TCP frame "
          f"{np.round(c_t*1000, 1).tolist()}mm, ext {FIT1['ext']*1000:.0f}mm, rms {FIT1['rms']*1000:.2f}mm, width {W_CLOSED:.4f}")
    print(ev)
    if ok:
        mb, mc = (m0, m1) if np.linalg.norm(in_tcp(T_LIFT, m0) - EB_PRED) < np.linalg.norm(in_tcp(T_LIFT, m1) - EB_PRED) else (m1, m0)
        EB_MEAS, EC_MEAS = in_tcp(T_LIFT, mb), in_tcp(T_LIFT, mc)
        print("rod in tcp: pred bottom", EB_PRED, "top", EC_PRED, "\n            meas bottom", EB_MEAS, "top", EC_MEAS)
        break
    servo(np.asarray(api.observe()["tcp_pos"], float), R_GRASP, n=10, grip=1.0, min_n=10)
    api.move_to(G_PT + EZ * 0.12, Q_GRASP, timeout_s=2.0)
    f = fit_rod()
    if f is None or abs(f["a"][2]) > 0.3 or abs(f["ext"] - L) > 0.006:
        print("  re-locate failed")
        break
    rE0, rE1 = rod_ends(f)
    rod_z0 = ((rE0 + rE1) / 2)[2]
    print("  re-located rod ctr", np.round((rE0 + rE1) / 2, 4))
stage_done(ok, ev)
