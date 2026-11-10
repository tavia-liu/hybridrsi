# grasp: top-down pinch of the lying rod near the end that will be on top, fingers pre-opened narrow,
# grasp point chosen free of neighbouring obstacles (depth heightmap) and with a reachable insertion.
# The close is done with my own Jacobian servo holding the TCP pose (set_gripper lets the arm ride up
# ~3 mm, exp02), then lift; if the camera fit says the rod was not lifted, re-locate and retry.
TCP_ABOVE_BOARD = 0.0110   # start height of the fingertip feel-down
TIP_REACH = 0.0089         # r01/exp02: fingertips touch the board with the TCP ~8.7-9.3 mm above it
W_PRE = 0.03               # pre-grasp opening
H_PRE = (Z_TOP - GB[2]) + 0.015   # rod bottom 15 mm above the fixture top before the descent
H_HIGH = 0.12
ASCALE = np.asarray(api.constants()["action_scale"], float)


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
    """Cartesian P-servo of the TCP with the Jacobian, one control step at a time.
    grip: action[7] (-1 close, +1 open, 0 keep)."""
    # r07: a pure P step on the measured joints left a steady 1-2 mm offset (gravity) -> the joint
    # target is integrated here (q_t accumulates), which removes the steady-state error.
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
        v = np.concatenate([k * e, k * w])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), v)
        q_m = np.asarray(o["joints"], float)
        q_t = q_t + dq
        q_t = q_m + np.clip(q_t - q_m, -0.6 * ASCALE, 0.6 * ASCALE)     # anti-windup
        a = np.clip((q_t - q_m) / ASCALE, -1, 1)
        api.step(np.concatenate([a, [grip]]))
    return o


def plan_insert(Eb, Ec, q_seed, verbose=False):
    """Eb/Ec = rod bottom/top axis points in the TCP frame. Rod vertical, bottom at GB(+dz),
    yaw about the vertical swept. Returns the cheapest reachable plan or None."""
    Rb = rot_a_to_b(Eb - Ec, -EZ)
    best = None
    lim = np.asarray(api.constants()["joint_limits"], float)
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
        # r05: approach pointing back toward the base put the hand/arm on the far side, low over the
        # board parts (blocked 7 mm above the bore, global camera occluded) -> prefer the hand between
        # the base and the fixture (approach pointing away from the base)
        radial = unit(np.array([GB[0], GB[1], 0.0]))
        zdir = unit(np.array([R[0, 2], R[1, 2], 0.0]))
        cost += 8.0 * (1.0 - float(zdir @ radial))
        # r06: smaller hand rotation from the grasp orientation = gentler on the fingertip pinch
        ang = np.arccos(np.clip((np.trace(R_GRASP_REF.T @ R) - 1) / 2, -1, 1))
        cost += 3.0 * ang
        if verbose:
            print(f"  psi={np.degrees(psi):6.1f} cost={cost:.2f} tcp_z_dir={np.round(R[:, 2], 2)}")
        if best is None or cost < best["cost"]:
            best = dict(psi=psi, cost=cost, poses=poses, Eb=Eb, Ec=Ec)
    return best


def in_tcp(T, p):
    return T[:3, :3].T @ (p - T[:3, 3])


def finger_clearance(g, R):
    """Number of heightmap points (not the rod) inside the two finger footprints, above the board."""
    y = R[:, 1]; x = R[:, 0]
    n = 0
    for sgn in (-1, 1):
        fc = g + sgn * y * (W_PRE / 2 + 0.010)
        d = HM - fc
        inside = (np.abs(d @ x) < 0.014) & (np.abs(d @ y) < 0.012) & (HM[:, 2] > BOARD_Z + 0.003) \
            & (HM[:, 2] < g[2] + 0.06)
        n += int(inside.sum())
    return n


def choose_grasp(rE0, rE1):
    a_h = rE1 - rE0; a_h[2] = 0; a_h = unit(a_h)
    y_ax = unit(np.cross(EZ, a_h))
    if y_ax @ np.array([0, -1.0, 0]) < 0:
        y_ax = -y_ax
    z_ax = -EZ
    Rg = np.column_stack([np.cross(y_ax, z_ax), y_ax, z_ax])
    global R_GRASP_REF
    R_GRASP_REF = Rg
    q_now = np.asarray(api.observe()["joints"], float)
    tcp_z = BOARD_Z + TCP_ABOVE_BOARD
    cands = []
    for k, (eA, eB) in enumerate(((rE0, rE1), (rE1, rE0))):    # eA = end to hold (top), eB = bottom
        for s in (L / 2, L / 2 - 0.004, L / 2 - 0.008):   # r07: off-centre pinch pivots under gravity
            g = eA + unit(eB - eA) * s
            g = np.array([g[0], g[1], tcp_z])
            Tg = Tm(g, Rg)
            Eb_t = in_tcp(Tg, eB); Ec_t = in_tcp(Tg, eA)
            Eb_t[1] = 0.0; Ec_t[1] = 0.0          # rod centred between the fingers
            coll = finger_clearance(g, Rg)
            plan = plan_insert(Eb_t, Ec_t, q_now) if coll < 15 else None
            cost = (plan["cost"] if plan else 1e9) + 60 * (L / 2 - s) + 0.5 * coll
            print(f"grasp cand end{k} s={s:.3f}: g={np.round(g, 4).tolist()} finger_pts={coll} "
                  f"plan={'none' if plan is None else round(plan['cost'], 2)}")
            cands.append((cost, k, s, g, Eb_t, Ec_t, plan, coll))
    cands.sort(key=lambda c: c[0])
    return Rg, cands[0]


ok = False
ev = "not attempted"
EB_MEAS = EC_MEAS = None
rE0, rE1 = R0E0, R0E1
rod_z0 = ROD_CTR0[2]
for attempt in range(3):
    R_GRASP, (_, K_SEL, S_SEL, G_PT, EB_PRED, EC_PRED, PLAN, COLL) = choose_grasp(rE0, rE1)
    Q_GRASP = mat_to_quat(R_GRASP)
    print(f"attempt {attempt}: end {K_SEL} s {S_SEL} finger_pts {COLL} plan feasible {PLAN is not None}")
    api.move_to(G_PT + EZ * 0.08, Q_GRASP, timeout_s=3.0)
    api.set_gripper(width=W_PRE, hold_s=0.3)
    api.move_to(G_PT + EZ * 0.015, Q_GRASP, timeout_s=2.0)
    servo(G_PT, R_GRASP, n=30, k=0.5)
    o = api.observe()
    print("  at grasp tcp", np.round(o["tcp_pos"], 4), "target", np.round(G_PT, 4), "F", np.round(o["tcp_wrench_ext"][:3], 1))
    # feel for the board with the fingertips: lower 0.3 mm per step until an upward contact force appears
    g_hold = G_PT.copy()
    z_floor = BOARD_Z + TIP_REACH - 0.0015
    zc = float(o["tcp_pos"][2])
    touched = False
    for _ in range(40):
        o = api.observe()
        if float(o["tcp_wrench_ext"][2]) > 3.0:
            touched = True
            break
        zc = max(zc - 0.0003, z_floor)
        servo(np.array([G_PT[0], G_PT[1], zc]), R_GRASP, n=1, k=1.0)
        if zc <= z_floor:
            break
    o = api.observe()
    g_hold = np.array([G_PT[0], G_PT[1], float(o["tcp_pos"][2]) + 0.0004])
    print(f"  board contact {touched} at tcp z {o['tcp_pos'][2]:.4f} (board {BOARD_Z:.4f}) Fz {o['tcp_wrench_ext'][2]:.1f}; hold z {g_hold[2]:.4f}")
    servo(g_hold, R_GRASP, n=6, k=0.5, min_n=6)
    o = servo(g_hold, R_GRASP, n=20, grip=-1.0, k=0.4, min_n=20)      # close while holding the pose
    o = api.observe()
    W_CLOSED = float(o["gripper_width"])
    print("  closed: width", round(W_CLOSED, 4), "tcp", np.round(o["tcp_pos"], 4), "F", np.round(o["tcp_wrench_ext"][:3], 1))
    api.move_to(G_PT + EZ * 0.07, Q_GRASP, timeout_s=2.0, gain=0.6)
    api.wait(0.15)

    T_LIFT = tcp_T()
    FIT1 = fit_rod()
    if FIT1 is None:
        ev = "no mask fit after lift"
        break
    tcp = T_LIFT[:3, 3]
    m0, m1 = rod_ends(FIT1, anchor_far_from=tcp)
    ctr = (m0 + m1) / 2
    dxy = np.linalg.norm(ctr[:2] - tcp[:2])
    ok = ctr[2] > rod_z0 + 0.04 and dxy < 0.04 and FIT1["n"] > 150
    ev = (f"attempt {attempt}: mask fit rod ctr z {rod_z0:.3f}->{ctr[2]:.3f}, ctr-tcp xy {dxy*1000:.0f}mm, "
          f"ext {FIT1['ext']*1000:.0f}mm, rms {FIT1['rms']*1000:.2f}mm, width {W_CLOSED:.4f}")
    print(ev)
    if ok:
        # which measured end is the planned bottom: the one nearer the predicted bottom (centre grasp)
        mb, mc = (m0, m1) if np.linalg.norm(in_tcp(T_LIFT, m0) - EB_PRED) < np.linalg.norm(in_tcp(T_LIFT, m1) - EB_PRED) else (m1, m0)
        EB_MEAS, EC_MEAS = in_tcp(T_LIFT, mb), in_tcp(T_LIFT, mc)
        print("rod in tcp: pred bottom", EB_PRED, "top", EC_PRED, "\n            meas bottom", EB_MEAS, "top", EC_MEAS)
        break
    # rod still on the board (or dropped): open, back off, re-locate from a fresh view and retry
    api.set_gripper(open_=True, hold_s=0.3)
    api.move_to(G_PT + EZ * 0.12, Q_GRASP, timeout_s=2.0)
    f = fit_rod()
    if f is None or abs(f["a"][2]) > 0.3 or abs(f["ext"] - L) > 0.006:
        print("  re-locate failed", None if f is None else (f["a"], f["ext"]))
        break
    rE0, rE1 = rod_ends(f)
    rod_z0 = ((rE0 + rE1) / 2)[2]
    print("  re-located rod ctr", np.round((rE0 + rE1) / 2, 4))
stage_done(ok, ev)
