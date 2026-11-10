# insert: the rod hangs vertically from a top-down pinch. Carry it (orientation unchanged) over the bore,
# let it settle, align its measured bottom over the bore axis by translation, step down into the bore
# (back out + re-align if blocked above the fixture top), release near the bottom, lift; verify.
R_TD = tcp_T()[:3, :3]
Q_TD = mat_to_quat(R_TD)


def measure_hanging(settle=True):
    """Rod bottom/top from the camera fit (waits until two fits agree, the rod may swing)."""
    prev = None
    for _ in range(4 if settle else 1):
        api.wait(0.2)
        f = fit_rod()
        if f is None:
            return None
        T = tcp_T()
        lo = f["p0"] if f["p0"][2] < f["p1"][2] else f["p1"]
        m0, m1 = rod_ends(f, anchor_far_from=lo + EZ * 1.0)
        mb, mc = (m0, m1) if m0[2] < m1[2] else (m1, m0)
        if np.linalg.norm((mb + mc) / 2 - T[:3, 3]) > 0.07:
            return None
        cur = dict(b=mb, c=mc, a=unit(mb - mc), T=T, f=f)
        if prev is not None and np.linalg.norm(cur["b"] - prev["b"]) < 0.0004:
            return cur
        prev = cur
    return prev


m = measure_hanging()
if m is None:
    stage_done(False, "rod not seen hanging in the hand")
off_b = m["b"] - m["T"][:3, 3]                       # rod bottom relative to the TCP (world frame)
print("rod bottom rel. TCP (mm)", np.round(off_b * 1000, 1), "tilt", round(np.degrees(np.arccos(min(1, -m['a'][2]))), 1))
Z_PRE = Z_TOP + 0.012
p_cmd = np.array([GB[0], GB[1], Z_PRE]) - off_b
T = tcp_T()
api.move_to(np.array([T[0, 3], T[1, 3], max(T[2, 3], p_cmd[2] + 0.04)]), Q_TD, timeout_s=2.0, gain=0.4)
api.move_to(p_cmd + EZ * 0.04, Q_TD, timeout_s=4.0, gain=0.3)
servo(p_cmd, R_TD, n=40, k=0.3)


def align(z_bottom, iters=5, tag=""):
    """Translate the TCP so the measured rod bottom sits at (goal xy, z_bottom)."""
    global p_cmd
    m = None
    for it in range(iters):
        m = measure_hanging()
        if m is None:
            print(f"align{tag} {it}: rod not seen"); return None
        err = np.array([GB[0], GB[1], z_bottom]) - m["b"]
        tilt = np.degrees(np.arccos(np.clip(-m["a"][2], -1, 1)))
        print(f"align{tag} {it}: bottom {np.round(m['b'], 4)} tilt {tilt:.2f}deg xy err {np.round(err[:2]*1000, 2)}mm "
              f"z err {err[2]*1000:.1f}mm ext {m['f']['ext']*1000:.0f} rms {m['f']['rms']*1000:.2f}")
        if np.linalg.norm(err[:2]) < 0.0004 and abs(err[2]) < 0.0015:
            return m
        if np.linalg.norm(err) > 0.03:
            print("align: error too large, not trusted"); return None
        p_cmd = np.asarray(m["T"][:3, 3], float) + err
        servo(p_cmd, R_TD, n=30, k=0.4)
    return measure_hanging()


m = align(Z_PRE)
if m is None:
    stage_done(False, "alignment over the bore failed")

# step down; blocked = the TCP stays short of its target
Z_REL = GB[2] + 0.003
z_targets = [Z_TOP + 0.004] + list(np.arange(Z_TOP - 0.002, Z_REL - 1e-4, -0.003)) + [Z_REL]
b_now = m["b"].copy()
retries = 0
i = 0
while i < len(z_targets):
    zb = z_targets[i]
    T = tcp_T()
    tgt = T[:3, 3] + EZ * (zb - b_now[2])
    servo(tgt, R_TD, n=20, k=0.4)
    o = api.observe()
    short = o["tcp_pos"][2] - tgt[2]
    F = np.asarray(o["tcp_wrench_ext"], float)
    print(f"step bottom->{zb:.4f}: tcp short by {short*1000:.1f}mm F={np.round(F[:3], 1)}")
    b_now = b_now + EZ * (zb - b_now[2]) + EZ * short
    if short > 0.0015 and zb > Z_TOP - 0.006:
        # tip on the fixture face: back out, re-align, restart
        retries += 1
        if retries > 2:
            break
        T = tcp_T()
        servo(T[:3, 3] + EZ * (Z_TOP + 0.008 - b_now[2]), R_TD, n=25, k=0.4)
        mm = align(Z_TOP + 0.008, iters=3, tag="R")
        if mm is None:
            break
        b_now = mm["b"].copy()
        i = 0
        continue
    if short > 0.0015:
        # inside the bore but stuck (pinch slides / bottom reached): stop pushing
        print("stuck inside the bore -> release here")
        break
    i += 1
print("release at rod bottom ~", round(float(b_now[2]), 4))
T = tcp_T()
servo(T[:3, 3], R_TD, n=12, grip=1.0, k=0.4, min_n=12)      # open while holding the pose
servo(T[:3, 3] + EZ * 0.08, R_TD, n=30, k=0.4)
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
