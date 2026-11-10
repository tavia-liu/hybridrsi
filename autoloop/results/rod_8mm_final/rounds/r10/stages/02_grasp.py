# grasp: top-down pinch near one END (A) of the lying rod, then PIVOT the rod up against the board:
# the far end B stays on the board while the (top-down, never rotating) hand moves on a circle around B
# until the rod stands vertical under the fingers; lift -> the rod hangs vertically from the pinch.
# Evidence: the fingertip pinch is a near-frictionless pivot about the closing axis (r09 s00 rod kept its
# world orientation while the hand turned 90 deg; r07 s00 rod pinched 11 mm from its end hung vertical).
TCP_ABOVE_BOARD = 0.0110   # start height of the fingertip feel-down
TIP_REACH = 0.0089         # fingertips touch the board with the TCP ~8.8-9.2 mm above it (exp02, exp03)
W_PRE = 0.03               # pre-grasp opening
S_END = 0.010              # pinch this far from end A
ASCALE = np.asarray(api.constants()["action_scale"], float)


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
        v = np.concatenate([k * e, k * w])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), v)
        q_m = np.asarray(o["joints"], float)
        q_t = q_t + dq
        q_t = q_m + np.clip(q_t - q_m, -0.6 * ASCALE, 0.6 * ASCALE)     # anti-windup
        a = np.clip((q_t - q_m) / ASCALE, -1, 1)
        api.step(np.concatenate([a, [grip]]))
    return o


def in_tcp(T, p):
    return T[:3, :3].T @ (p - T[:3, 3])


def clearance(g, R, zmin, zmax, half_x=0.014, half_y=0.012, gap=W_PRE / 2 + 0.010):
    """Heightmap points (not the rod) inside the two finger footprints between zmin and zmax."""
    y = R[:, 1]; x = R[:, 0]
    n = 0
    for sgn in (-1, 1):
        d = HM - (g + sgn * y * gap)
        n += int(((np.abs(d @ x) < half_x) & (np.abs(d @ y) < half_y) & (HM[:, 2] > zmin) & (HM[:, 2] < zmax)).sum())
    return n


def top_down_R(a_h):
    y_ax = unit(np.cross(EZ, a_h))
    if y_ax @ np.array([0, -1.0, 0]) < 0:
        y_ax = -y_ax
    z_ax = -EZ
    return np.column_stack([np.cross(y_ax, z_ax), y_ax, z_ax])


def choose_end(rE0, rE1):
    a_h = rE1 - rE0; a_h[2] = 0; a_h = unit(a_h)
    Rg = top_down_R(a_h)
    best = None
    for k, (eA, eB) in enumerate(((rE0, rE1), (rE1, rE0))):
        u = eB - eA; u[2] = 0; u = unit(u)
        g = eA + u * S_END
        g = np.array([g[0], g[1], BOARD_Z + TCP_ABOVE_BOARD])
        c0 = clearance(g, Rg, BOARD_Z + 0.003, g[2] + 0.06)
        # hand above B at the end of the pivot: fingers span ~[B top - 12 mm, + 60 mm]
        gB = np.array([eB[0], eB[1], BOARD_Z + L])
        c1 = clearance(gB, Rg, BOARD_Z + L - S_END - 0.012, BOARD_Z + L + 0.06)
        print(f"end cand A=end{k}: grasp {np.round(g, 4).tolist()} finger pts grasp {c0} / over B {c1}")
        if best is None or c0 + c1 < best[0]:
            best = (c0 + c1, k, eA, eB, u, g, Rg)
    return best


def feel_down_and_close(G, R):
    servo(G, R, n=30, k=0.5)
    o = api.observe()
    z_floor = BOARD_Z + TIP_REACH - 0.0015
    zc = float(o["tcp_pos"][2])
    touched = False
    for _ in range(40):
        o = api.observe()
        if float(o["tcp_wrench_ext"][2]) > 3.0:
            touched = True
            break
        zc = max(zc - 0.0003, z_floor)
        servo(np.array([G[0], G[1], zc]), R, n=1, k=1.0)
        if zc <= z_floor:
            break
    o = api.observe()
    g_hold = np.array([G[0], G[1], float(o["tcp_pos"][2]) + 0.0004])
    print(f"  board contact {touched} at tcp z {o['tcp_pos'][2]:.4f} (board {BOARD_Z:.4f}); hold z {g_hold[2]:.4f}")
    servo(g_hold, R, n=6, k=0.5, min_n=6)
    servo(g_hold, R, n=20, grip=-1.0, k=0.4, min_n=20)      # close while holding the pose
    o = api.observe()
    print("  closed: width", round(float(o["gripper_width"]), 4), "F", np.round(o["tcp_wrench_ext"][:3], 1))
    return g_hold


ok = False
ev = "not attempted"
rE0, rE1 = R0E0, R0E1
for attempt in range(2):
    _, K_SEL, eA, eB, U_AB, G_PT, R_GRASP = choose_end(rE0, rE1)
    Q_GRASP = mat_to_quat(R_GRASP)
    print(f"attempt {attempt}: hold end{K_SEL} at {np.round(G_PT, 4).tolist()}")
    api.move_to(G_PT + EZ * 0.08, Q_GRASP, timeout_s=3.0)
    api.set_gripper(width=W_PRE, hold_s=0.3)
    api.move_to(G_PT + EZ * 0.015, Q_GRASP, timeout_s=2.0)
    g_hold = feel_down_and_close(G_PT, R_GRASP)

    # pivot about B's lower rim, keeping |pinch - B| constant (0.5 mm short to keep B pressed on the board)
    P0 = eA + U_AB * S_END                 # pinch point on the rod axis
    P0[2] = (eA[2] + eB[2]) / 2
    off = g_hold - P0                      # TCP relative to the pinch (hand orientation is fixed)
    Bax = eB.copy(); Bax[2] = P0[2]
    Q = Bax - RAD * EZ
    d = np.linalg.norm((P0 - Bax)[:2]) - 0.0005
    for phi in np.radians(np.linspace(5, 92, 18)):
        Bp = Q + RAD * (U_AB * np.sin(phi) + EZ * np.cos(phi))
        P = Bp + d * (-U_AB * np.cos(phi) + EZ * np.sin(phi))
        servo(P + off, R_GRASP, n=5, k=0.6)
    o = api.observe()
    print("  after pivot: tcp", np.round(o["tcp_pos"], 4), "F", np.round(o["tcp_wrench_ext"][:3], 1),
          "width", round(float(o["gripper_width"]), 4))
    # lift slowly; the rod hangs from the pinch
    T = tcp_T()
    servo(T[:3, 3] + EZ * 0.06, R_GRASP, n=40, k=0.25)
    api.wait(0.4)

    f = fit_rod()
    T = tcp_T()
    if f is None:
        ev = f"attempt {attempt}: no mask fit after the pivot/lift"
    else:
        lo, hi = (f["p0"], f["p1"]) if f["p0"][2] < f["p1"][2] else (f["p1"], f["p0"])
        tilt = np.degrees(np.arccos(min(1.0, abs(f["a"][2]))))
        top_d = np.linalg.norm(hi[:2] - T[:3, 3][:2])
        ok = tilt < 15 and lo[2] > BOARD_Z + 0.02 and top_d < 0.02 and f["n"] > 150
        ev = (f"attempt {attempt}: rod hanging, tilt {tilt:.1f}deg, bottom z {lo[2]:.4f} (board {BOARD_Z:.4f}), "
              f"top {top_d*1000:.0f}mm from TCP axis, ext {f['ext']*1000:.0f}mm rms {f['rms']*1000:.2f}")
    print(ev)
    if ok:
        break
    # failed: open, back off, re-locate the rod and retry
    servo(np.asarray(api.observe()["tcp_pos"], float), R_GRASP, n=10, grip=1.0, min_n=10)
    api.move_to(np.asarray(api.observe()["tcp_pos"], float) + EZ * 0.08, Q_GRASP, timeout_s=2.0)
    f = fit_rod()
    if f is None or abs(f["a"][2]) > 0.3 or abs(f["ext"] - L) > 0.006:
        print("  re-locate failed")
        break
    rE0, rE1 = rod_ends(f)
    BOARD_Z = ((rE0 + rE1) / 2)[2] - RAD
    print("  re-located rod ctr", np.round((rE0 + rE1) / 2, 4))
stage_done(ok, ev)
