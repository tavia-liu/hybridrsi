# insert: turn the rod vertical over the bore, closed-loop camera alignment of the rod's TILT and its
# actual bottom point, descend in steps (re-align when blocked), release inside the bore, back off; verify.
use_meas = (EB_MEAS is not None and np.linalg.norm(EB_MEAS - EB_PRED) < 0.008
            and np.linalg.norm(EC_MEAS - EC_PRED) < 0.012)
EB_USE, EC_USE = (EB_MEAS, EC_MEAS) if use_meas else (EB_PRED, EC_PRED)
print("rod-in-hand model:", "measured" if use_meas else "predicted")
q_now = np.asarray(api.observe()["joints"], float)
plan = plan_insert(EB_USE, EC_USE, q_now, verbose=False)
if plan is None and use_meas:
    EB_USE, EC_USE = EB_PRED, EC_PRED
    plan = plan_insert(EB_USE, EC_USE, q_now, verbose=False)
if plan is None:
    stage_done(False, "no reachable vertical insertion pose")

R_ins = plan["poses"]["high"][1]
print("insert psi", np.degrees(plan["psi"]), "tcp z dir", np.round(R_ins[:, 2], 3), "cost", round(plan["cost"], 2))
api.move_to_joints(plan["poses"]["high"][2], timeout_s=6.0, gain=0.5)
api.move_to(plan["poses"]["pre"][0], mat_to_quat(R_ins), timeout_s=3.0, gain=0.6)


def measure_hanging():
    """Rod bottom point and downward axis from the camera fit, or None if the rod is not in the hand."""
    f = fit_rod()
    if f is None:
        return None
    T = tcp_T()
    m0, m1 = rod_ends(f, anchor_far_from=T[:3, 3])
    mb, mc = (m0, m1) if np.linalg.norm(m0 - T[:3, 3]) > np.linalg.norm(m1 - T[:3, 3]) else (m1, m0)
    if np.linalg.norm((mb + mc) / 2 - T[:3, 3]) > 0.06:
        return None
    return dict(b=mb, a=unit(mb - mc), T=T, f=f)


def align(z_bottom, iters=4, tag=""):
    """Rotate/translate the TCP so the rod axis is vertical and its bottom is at (goal xy, z_bottom)."""
    m = None
    for it in range(iters):
        api.wait(0.25)
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
        if np.linalg.norm(err) > 0.03:
            print("align: error too large, not trusted")
            return None
        Rc = rot_a_to_b(m["a"], -EZ)
        Rcur = m["T"][:3, :3]; pcur = m["T"][:3, 3]
        b_des = m["b"] + err
        p_new = b_des + Rc @ (pcur - m["b"])
        servo(p_new, Rc @ Rcur, n=30, k=0.4)
    api.wait(0.25)
    m2 = measure_hanging()
    return m2 if m2 is not None else m


m = align(Z_TOP + 0.015, iters=5)
if m is None:
    stage_done(False, "rod not in hand / alignment failed above the fixture")

# step down: bottom to just above the bore, then into it; detect blocking from the TCP not reaching
Z_REL = max(GB[2] + 0.004, Z_TOP - 0.012)        # release height of the rod bottom (inside the bore)
z_targets = [Z_TOP + 0.004] + list(np.arange(Z_TOP - 0.002, Z_REL - 1e-4, -0.003)) + [Z_REL]
retries = 0
i = 0
blocked_final = False
b_now = m["b"].copy()
while i < len(z_targets):
    zb = z_targets[i]
    T = tcp_T()
    # move TCP by the bottom's height change (orientation held)
    tgt = T[:3, 3] + EZ * (zb - b_now[2])
    servo(tgt, T[:3, :3], n=20, k=0.4)
    o = api.observe()
    short = o["tcp_pos"][2] - tgt[2]
    F = np.asarray(o["tcp_wrench_ext"], float)
    print(f"step bottom->{zb:.4f}: tcp short by {short*1000:.1f}mm F={np.round(F[:3], 1)}")
    b_now = b_now + EZ * (zb - b_now[2]) + EZ * short
    if short > 0.0015:
        retries += 1
        if retries > 2:
            blocked_final = True
            break
        # back out above the fixture top, re-align from a fresh view, restart the descent
        T = tcp_T()
        servo(T[:3, 3] + EZ * (Z_TOP + 0.008 - b_now[2]), T[:3, :3], n=25, k=0.4)
        mm = align(Z_TOP + 0.008, iters=3, tag="R")
        if mm is None:
            break
        b_now = mm["b"].copy()
        i = 0
        continue
    i += 1
print("release; blocked" if blocked_final else "release")
T = tcp_T()
servo(T[:3, 3], T[:3, :3], n=12, grip=1.0, k=0.4, min_n=12)     # open while holding the pose
o = api.observe()
api.move_to(np.asarray(o["tcp_pos"]) + EZ * 0.08, np.asarray(o["tcp_quat"]), timeout_s=2.0)
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
