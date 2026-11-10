# grasp: top-down pinch near one end of the lying rod (end chosen so the insertion pose is reachable), lift.
S_GRASP = 0.011          # grasp point distance from the rod end that will be on top
TCP_BELOW_AXIS = 0.001   # fingertip centre this far below the rod axis
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


def plan_insert(T_rel, q_seed, verbose=False):
    """T_rel = tcp^-1 * part. Rod bottom = the end farther from the TCP. Returns best plan dict or None."""
    Et = T_rel[:3, :3] @ E0 + T_rel[:3, 3]
    Eu = T_rel[:3, :3] @ E1 + T_rel[:3, 3]
    Eb, Ec = (Et, Eu) if np.linalg.norm(Et) > np.linalg.norm(Eu) else (Eu, Et)
    Rb = rot_a_to_b(Eb - Ec, -EZ)          # tcp-frame rod (top->bottom) onto world -z
    best = None
    lim = np.asarray(api.constants()["joint_limits"], float)
    for psi in np.linspace(-np.pi, np.pi, 24, endpoint=False):
        R = rotz(psi) @ Rb
        poses = {}
        good = True
        q_prev = q_seed
        for name, dz in (("high", H_HIGH), ("pre", H_PRE), ("fin", 0.0)):
            pos = GB + EZ * dz - R @ Eb
            q, reach = api.ik(pos, mat_to_quat(R), q_init=q_prev)
            q = np.asarray(q, float)
            if not reach:
                good = False
                break
            margin = np.min(np.minimum(q - lim[:, 0], lim[:, 1] - q))
            if margin < 0.05:
                good = False
                break
            poses[name] = (pos, R, q)
            q_prev = q
        if not good:
            continue
        cost = np.abs(poses["high"][2] - q_seed).sum() + 3 * np.abs(poses["fin"][2] - poses["high"][2]).sum()
        if verbose:
            print(f"  psi={np.degrees(psi):6.1f} cost={cost:.2f} tcp_z_dir={np.round(R[:, 2], 2)}")
        if best is None or cost < best["cost"]:
            best = dict(psi=psi, cost=cost, poses=poses, Eb=Eb, Ec=Ec)
    return best


e0, e1 = ends_world(P_EST, Q_EST)
ctr = (e0 + e1) / 2
a_h = e1 - e0; a_h[2] = 0; a_h = unit(a_h)
y_ax = unit(np.cross(EZ, a_h))
if y_ax @ np.array([0, -1.0, 0]) < 0:
    y_ax = -y_ax
z_ax = -EZ
x_ax = np.cross(y_ax, z_ax)
R_GRASP = np.column_stack([x_ax, y_ax, z_ax])
Q_GRASP = mat_to_quat(R_GRASP)
T_PART0 = Tm(P_EST, quat_to_mat(Q_EST))
q_now = np.asarray(api.observe()["joints"], float)

cands = []
for k, (eA, eB) in enumerate(((e0, e1), (e1, e0))):     # eA = end to hold (top), eB = bottom
    g = eA + unit(eB - eA) * S_GRASP
    g = np.array([g[0], g[1], ctr[2] - TCP_BELOW_AXIS])
    T_rel_pred = Tinv(Tm(g, R_GRASP)) @ T_PART0
    plan = plan_insert(T_rel_pred, q_now)
    print(f"grasp cand {k}: g={np.round(g, 4).tolist()} plan={'none' if plan is None else round(plan['cost'], 2)}")
    cands.append((plan["cost"] if plan else 1e9, k, g, T_rel_pred, plan))
cands.sort(key=lambda c: c[0])
_, K_SEL, G_PT, T_REL_PRED, PLAN = cands[0]
print("selected grasp end", K_SEL, "plan feasible", PLAN is not None)

api.move_to(G_PT + EZ * 0.10, Q_GRASP, timeout_s=3.0)
api.move_to(G_PT + EZ * 0.025, Q_GRASP, timeout_s=2.0)
api.move_to(G_PT, Q_GRASP, timeout_s=2.0, tol=0.005, gain=0.6)
o = api.observe()
print("at grasp tcp", np.round(o["tcp_pos"], 4), "target", np.round(G_PT, 4), "wrench", np.round(o["tcp_wrench_ext"], 2))
api.set_gripper(open_=False, hold_s=0.6)
W_CLOSED = float(api.observe()["gripper_width"])
api.move_to(G_PT + EZ * 0.06, Q_GRASP, timeout_s=2.0, gain=0.7)
api.wait(0.12)

p, q, c = fresh_est()
ok = False
ev = "no estimate after lift"
if p is not None:
    f0, f1 = ends_world(p, q)
    pc = (f0 + f1) / 2
    tcp = np.asarray(api.observe()["tcp_pos"], float)
    T_REL_MEAS = Tinv(tcp_T()) @ Tm(p, quat_to_mat(q))
    cloud = mask_cloud()
    cz = cloud[:, 2].mean() if len(cloud) else 0.0
    ok = pc[2] > ctr[2] + 0.035 and np.linalg.norm(pc[:2] - tcp[:2]) < 0.04 and cz > ctr[2] + 0.03
    ev = (f"rod ctr z {ctr[2]:.3f}->{pc[2]:.3f} (mask z {cz:.3f}), ctr-tcp xy {np.linalg.norm(pc[:2]-tcp[:2])*1000:.0f}mm, "
          f"width {W_CLOSED:.4f}")
    print("T_rel pred\n", T_REL_PRED, "\nT_rel meas\n", T_REL_MEAS)
print(ev)
stage_done(ok, ev)
