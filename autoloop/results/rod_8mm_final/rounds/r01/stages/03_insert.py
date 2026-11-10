# insert: turn the rod vertical over the hole, descend, release so it drops to the bottom, back off.
q_now = np.asarray(api.observe()["joints"], float)
plan = plan_insert(T_REL_PRED, q_now, verbose=True)
if plan is None:
    print("no plan from predicted grasp, trying measured")
    plan = plan_insert(T_REL_MEAS, q_now, verbose=True)
if plan is None:
    stage_done(False, "no reachable vertical insertion pose")

pos_h, R_ins, q_h = plan["poses"]["high"]
Q_INS = mat_to_quat(R_ins)
print("insert psi", np.degrees(plan["psi"]), "tcp z dir", np.round(R_ins[:, 2], 3))
api.move_to_joints(q_h, timeout_s=4.0)
api.move_to(plan["poses"]["pre"][0], Q_INS, timeout_s=3.0)
api.wait(0.1)

# vision check of the hanging rod before going in
p, q, c = fresh_est()
if p is not None:
    f0, f1 = ends_world(p, q)
    fb = f0 if f0[2] < f1[2] else f1
    print("pre-insert: tracker rod bottom", np.round(fb, 4), "planned", np.round(GB + EZ * H_PRE, 4),
          "axis", np.round(unit(f1 - f0), 3))

# descend to just above the hole bottom
fin = plan["poses"]["fin"][0]
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

p, q, c = fresh_est()
ok = False
ev = "no estimate"
if p is not None:
    f0, f1 = ends_world(p, q)
    fb, ft = (f0, f1) if f0[2] < f1[2] else (f1, f0)
    eb = np.linalg.norm(fb - GB); et = np.linalg.norm(ft - GT)
    cloud = mask_cloud()
    mz = cloud[:, 2].max() if len(cloud) else 0
    ok = eb < 0.004 and et < 0.004
    ev = f"tracker bottom err {eb*1000:.1f}mm top err {et*1000:.1f}mm, mask top z {mz:.4f} (goal top {GT[2]:.4f})"
print(ev)
stage_done(ok, ev)
