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
        cost = np.abs(poses["high"][2] - q_seed).sum() + 3 * np.abs(poses["fin"][2] - poses["high"][2]).sum()
        zdir = unit(np.array([R[0, 2], R[1, 2], 0.0]))
        cost += 8.0 * (1.0 - float(zdir @ radial))      # r05: hand between base and fixture
        ang = np.arccos(np.clip((np.trace(R_GRASP_REF.T @ R) - 1) / 2, -1, 1))
        cost += 3.0 * ang                               # r06: smaller hand rotation
        if verbose:
            print(f"  psi={np.degrees(psi):6.1f} cost={cost:.2f} tcp_z_dir={np.round(R[:, 2], 2)}")
        out.append(dict(psi=psi, cost=cost, poses=poses, Eb=Eb, Ec=Ec))
    out.sort(key=lambda d: d["cost"])
    return out


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
        for s in (L / 2, L / 2 - 0.005, L / 2 - 0.010):
            g = eA + unit(eB - eA) * s
            g = np.array([g[0], g[1], BOARD_Z + TCP_ABOVE_BOARD])
            Tg = Tm(g, Rg)
            Eb_t = in_tcp(Tg, eB); Ec_t = in_tcp(Tg, eA)
            Eb_t[1] = 0.0; Ec_t[1] = 0.0
            coll = finger_clearance(g, Rg)
            plans = plan_insert(Eb_t, Ec_t, q_now) if coll < 15 else []
            cost = (plans[0]["cost"] if plans else 1e9) + 60 * (L / 2 - s) + 0.5 * coll
            print(f"grasp cand end{k} s={s:.3f}: finger_pts={coll} plan={'none' if not plans else round(plans[0]['cost'], 2)}")
            cands.append((cost, k, s, g, Eb_t, Ec_t, coll))
    cands.sort(key=lambda c: c[0])
    return Rg, cands[0]


def feel_down_and_close(G, R):
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
    g_hold = np.array([G[0], G[1], float(o["tcp_pos"][2]) + 0.0004])
    print(f"  contact at tcp z {o['tcp_pos'][2]:.4f} Fz {o['tcp_wrench_ext'][2]:.1f} (board {BOARD_Z:.4f})")
    servo(g_hold, R, n=6, k=0.5, min_n=6)
    for tr in range(2):
        servo(g_hold, R, n=20, grip=-1.0, k=0.4, min_n=20)      # close while holding the pose
        w = float(api.observe()["gripper_width"])
        print(f"  close {tr}: width {w:.4f} F {np.round(api.observe()['tcp_wrench_ext'][:3], 1)}")
        if w > 2 * RAD - 0.0015:
            return g_hold, w
        # exp04: s01 closes ended at 3.3-4.6 mm with the rod at the rounded tips (lost on the first
        # rotation step); s00 closes at 8.9 mm held the rod inside the pads. Re-open a little, re-close.
        api.set_gripper(width=2 * RAD + 0.006, hold_s=0.3)
        servo(g_hold, R, n=15, k=0.5)
    return g_hold, w


ok = False
ev = "not attempted"
EB_MEAS = EC_MEAS = None
rE0, rE1 = R0E0, R0E1
rod_z0 = ROD_CTR0[2]
JITTER = [(0.0, 0.0), (0.006, np.radians(8)), (-0.006, np.radians(-8)), (0.0, np.radians(15))]
for attempt in range(4):
    R_GRASP, (_, K_SEL, S_SEL, G_PT, EB_PRED, EC_PRED, COLL) = choose_grasp(rE0, rE1)
    # r11: identical retries gave identical failures -> vary the grasp point along the rod and the yaw
    dsh, dyaw = JITTER[attempt]
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
    g_hold, W_CLOSED = feel_down_and_close(G_PT, R_GRASP)
    # exp04/r11: a grip with the rod centre > ~5.5 mm along TCP z (at the rounded tips) pivots or drops
    # later. Check right after a 4 mm lift; if bad, put it back down and retry with a variation.
    servo(g_hold + EZ * 0.004, R_GRASP, n=20, k=0.3)
    api.wait(0.15)
    f = fit_rod()
    T = tcp_T()
    depth_ok = False
    if f is not None:
        cc = (f["p0"] + f["p1"]) / 2
        ct = in_tcp(T, cc)
        lifted = cc[2] > rod_z0 + 0.002
        depth_ok = lifted and ct[2] < 0.0055 and abs(ct[1]) < 0.003
        print(f"  4 mm up: rod ctr z {cc[2]:.4f} (was {rod_z0:.4f}), in TCP frame {np.round(ct*1000, 1).tolist()}mm -> "
              f"{'pads' if depth_ok else 'NOT in the pads'}")
    if not depth_ok and attempt < 3:
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
