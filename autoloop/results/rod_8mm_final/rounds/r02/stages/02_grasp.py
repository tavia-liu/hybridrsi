# grasp: top-down pinch of the lying rod near the end that will be on top, fingers pre-opened narrow,
# grasp point chosen free of neighbouring obstacles (depth heightmap) and with a reachable insertion; lift.
TIP_BELOW_TCP = 0.009    # r01: fingertips hit the board with the TCP 9 mm above it
TCP_ABOVE_BOARD = 0.0115
W_PRE = 0.03             # pre-grasp opening
H_PRE = 0.035            # rod bottom above the goal bottom before the final descent
H_HIGH = 0.12


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


a_h = R0E1 - R0E0; a_h[2] = 0; a_h = unit(a_h)
y_ax = unit(np.cross(EZ, a_h))
if y_ax @ np.array([0, -1.0, 0]) < 0:
    y_ax = -y_ax
z_ax = -EZ
R_GRASP = np.column_stack([np.cross(y_ax, z_ax), y_ax, z_ax])
Q_GRASP = mat_to_quat(R_GRASP)
q_now = np.asarray(api.observe()["joints"], float)
TCP_Z = BOARD_Z + TCP_ABOVE_BOARD

cands = []
for k, (eA, eB) in enumerate(((R0E0, R0E1), (R0E1, R0E0))):    # eA = end to hold (top), eB = bottom
    for s in (0.011, 0.016, 0.021):
        g = eA + unit(eB - eA) * s
        g = np.array([g[0], g[1], TCP_Z])
        Tg = Tm(g, R_GRASP)
        ax_pt = (eA + eB) / 2
        # rod centred between the fingers: project its axis into the TCP y=0 plane
        Eb_t = in_tcp(Tg, eB); Ec_t = in_tcp(Tg, eA)
        Eb_t[1] = 0.0; Ec_t[1] = 0.0
        coll = finger_clearance(g, R_GRASP)
        plan = plan_insert(Eb_t, Ec_t, q_now) if coll < 15 else None
        cost = (plan["cost"] if plan else 1e9) + 20 * (s - 0.011) + 0.5 * coll
        print(f"grasp cand end{k} s={s:.3f}: g={np.round(g, 4).tolist()} finger_pts={coll} "
              f"plan={'none' if plan is None else round(plan['cost'], 2)}")
        cands.append((cost, k, s, g, Eb_t, Ec_t, plan, coll))
cands.sort(key=lambda c: c[0])
_, K_SEL, S_SEL, G_PT, EB_PRED, EC_PRED, PLAN, COLL = cands[0]
print("selected end", K_SEL, "s", S_SEL, "finger_pts", COLL, "plan feasible", PLAN is not None)

api.move_to(G_PT + EZ * 0.10, Q_GRASP, timeout_s=3.0)
api.set_gripper(width=W_PRE, hold_s=0.3)
api.move_to(G_PT + EZ * 0.02, Q_GRASP, timeout_s=2.0)
api.move_to(G_PT, Q_GRASP, timeout_s=1.5, tol=0.004, gain=0.6)
o = api.observe()
print("at grasp tcp", np.round(o["tcp_pos"], 4), "target", np.round(G_PT, 4), "wrench", np.round(o["tcp_wrench_ext"], 2))
api.set_gripper(open_=False, hold_s=0.6)
W_CLOSED = float(api.observe()["gripper_width"])
api.move_to(G_PT + EZ * 0.07, Q_GRASP, timeout_s=2.0, gain=0.7)
api.wait(0.12)

T_LIFT = tcp_T()
FIT1 = fit_rod()
ok = False
ev = "no mask fit after lift"
EB_MEAS = EC_MEAS = None
if FIT1 is not None:
    tcp = T_LIFT[:3, 3]
    m0, m1 = rod_ends(FIT1, anchor_far_from=tcp)
    mb, mc = (m0, m1) if np.linalg.norm(m0 - tcp) > np.linalg.norm(m1 - tcp) else (m1, m0)
    EB_MEAS, EC_MEAS = in_tcp(T_LIFT, mb), in_tcp(T_LIFT, mc)
    ctr = (m0 + m1) / 2
    dxy = np.linalg.norm(ctr[:2] - tcp[:2])
    ok = ctr[2] > ROD_CTR0[2] + 0.04 and dxy < 0.04 and FIT1["n"] > 150
    ev = (f"mask fit rod ctr z {ROD_CTR0[2]:.3f}->{ctr[2]:.3f}, ctr-tcp xy {dxy*1000:.0f}mm, ext {FIT1['ext']*1000:.0f}mm, "
          f"rms {FIT1['rms']*1000:.2f}mm, width {W_CLOSED:.4f}")
    print("rod in tcp: pred bottom", EB_PRED, "top", EC_PRED, "\n            meas bottom", EB_MEAS, "top", EC_MEAS)
print(ev)
stage_done(ok, ev)
