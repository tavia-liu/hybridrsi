# insert: turn the rod vertical over the hole, correct xy with the camera fit of the hanging rod,
# descend, release so it drops to the bottom, back off; verify with the mask fit.
use_meas = (EB_MEAS is not None and np.linalg.norm(EB_MEAS - EB_PRED) < 0.008
            and np.linalg.norm(EC_MEAS - EC_PRED) < 0.012)
EB_USE, EC_USE = (EB_MEAS, EC_MEAS) if use_meas else (EB_PRED, EC_PRED)
print("rod-in-hand model:", "measured" if use_meas else "predicted")
q_now = np.asarray(api.observe()["joints"], float)
plan = plan_insert(EB_USE, EC_USE, q_now, verbose=True)
if plan is None and use_meas:
    EB_USE, EC_USE = EB_PRED, EC_PRED
    plan = plan_insert(EB_USE, EC_USE, q_now, verbose=True)
if plan is None:
    stage_done(False, "no reachable vertical insertion pose")

R_ins = plan["poses"]["high"][1]
Q_INS = mat_to_quat(R_ins)
print("insert psi", np.degrees(plan["psi"]), "tcp z dir", np.round(R_ins[:, 2], 3))
api.move_to_joints(plan["poses"]["high"][2], timeout_s=4.0)
pre = plan["poses"]["pre"][0].copy()
api.move_to(pre, Q_INS, timeout_s=3.0)

# closed-loop xy alignment of the hanging rod's axis over the hole axis
corr = np.zeros(3)
for it in range(3):
    api.wait(0.1)
    f = fit_rod()
    if f is None:
        print("align: no fit"); break
    T = tcp_T()
    m0, m1 = rod_ends(f, anchor_far_from=T[:3, 3])
    mb = m0 if m0[2] < m1[2] else m1
    a = unit(m1 - m0)
    # axis point at the goal-bottom height (extrapolate the line), compare with the goal axis
    lam = (GB[2] - mb[2]) / a[2] if abs(a[2]) > 0.5 else 0.0
    at_goal = mb + lam * a
    err = GB - at_goal
    err[2] = (GB[2] + H_PRE) - mb[2]
    print(f"align {it}: rod bottom {np.round(mb, 4)} axis {np.round(a, 3)} ext {f['ext']*1000:.0f}mm "
          f"rms {f['rms']*1000:.2f} xy err {np.round(err[:2]*1000, 1)}mm z err {err[2]*1000:.1f}mm")
    if np.linalg.norm(err[:2]) < 0.0005 and abs(err[2]) < 0.002:
        break
    if np.linalg.norm(err) > 0.02:
        print("align: correction too large, ignored"); break
    corr += err
    api.move_to(pre + corr, Q_INS, timeout_s=1.5, tol=0.003)

fin = plan["poses"]["fin"][0] + corr
for dz in (0.02, 0.01, 0.004):
    api.move_to(fin + EZ * dz, Q_INS, timeout_s=1.5, tol=0.004, gain=0.5)
    o = api.observe()
    F = np.asarray(o["tcp_wrench_ext"], float)
    print(f"descend dz={dz:.3f} tcp z={o['tcp_pos'][2]:.4f} want {fin[2] + dz:.4f} F={np.round(F[:3], 2)}")
api.set_gripper(open_=True, hold_s=0.4)
o = api.observe()
api.move_to(np.asarray(o["tcp_pos"]) - R_ins[:, 2] * 0.06, Q_INS, timeout_s=2.0)
api.move_to(np.asarray(api.observe()["tcp_pos"]) + EZ * 0.10, Q_INS, timeout_s=2.0)
api.wait(0.2)

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
tr = tracker()
if tr:
    print("tracker ends", np.round(tr[0], 4), np.round(tr[1], 4))
print(ev)
stage_done(ok, ev)
