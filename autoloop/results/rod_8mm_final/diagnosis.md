# Diagnosis notebook

## Round 0 -- experiment 01_scene (s00, s01)
Hypothesis: the R8 rod lies on the board and goes into a vertical hole (goal axis vertical); locate_part
gives a usable pose from the global cameras at home.
Prediction: goal maps the long axis to world z; tracker gives a horizontal axis at board height.
Observed (summary.txt stdout, keyframes k00_t01.96 / k00_t02.20):
- part model: cylinder, bbox [-4,-4,0]..[4,4,50] mm -> 8 mm dia, 50 mm long, frame origin at one end, axis = part z.
- goal_pos [0.3447, 0.0756, 0.0125], goal_quat maps part z -> world +z: rod stands vertical, bottom at z=12.5 mm.
- tracker at home: s00 pos z 0.0128, axis [-0.89,0.46,-0.003]; s01 z 0.0130, axis [0.01,1.0,0.003] -> horizontal
  at board height (board top ~8.8 mm). Uses global2, ~2000 px. Own mask back-projection agrees within a few mm
  (s00 mask mean [0.578,0.002]/tracker ctr ~[0.577,0.003]).
- tcp at home points down (tcp R z = [0.10,0,-0.995]), tcp y ~ -world y. Observe keys: tcp_pos, tcp_quat,
  joints, gripper_width, tcp_wrench_ext, goal_pos, goal_quat.
- images: a blue holder with a round bore on the board is the likely fixture.
Prediction held.

Plan for round 1: locate -> grasp -> insert. Top-down pinch 11 mm from one end, rod becomes perpendicular to
the approach axis; to stand it up the hand turns to a ~horizontal approach (rod hanging below the fingers).
Insertion pose solved by IK over yaw around the vertical rod (rotationally symmetric, flip-symmetric
success), end chosen by feasibility. Release 4 mm above the hole bottom so it drops in.
Predictions/unknowns: whether the horizontal-approach pose is IK reachable near the fixture (grasp prints
plan feasibility), whether TCP 1 mm below the rod axis keeps the fingertips off the board (watch wrench at
grasp), and the lateral error at the hole (tracker vs grasp-prediction printed pre-insert).

## Round 1 -- 0/2 (both TIMEOUT; grasp FAIL by my verifier, insert never ran)
Previous hypotheses: horizontal-approach insertion reachable -> supported (plan costs 4.7 found on both).
Fingertips 1 mm below the rod axis clear the board -> REFUTED (fingertips hit the board).
focus: grasp (the report names insert, but insert never ran: grasp's verifier stopped the policy) -- s00, s01
- s00: trace t=4.12 descent stalls at tcp z 0.0178 (target 0.0117), wrench Fz +39 N; k03 wrist view rod between
  fingers -> fingertips on the board. t=4.72 close, width 0.009; k05 t=5.32 rod clearly lifted in the fingers.
  Verifier FAIL because the tracker estimate was frozen on the board position (stale, hand occludes it)
  while my mask cloud said z 0.063.
- s01: t=2.84 pre-grasp Fx +54 N, hand brushing the gray block next to the rod (k01); t=5.44 descent stalls
  at z 0.0189; t=6.88 rod lifted in hand (k05/k06), tracker estimate went below the board (z -0.002).
Root causes: (1) the verifier (and the planned rod-in-hand transform) used the tracker, which is wrong/stale
when the hand occludes the rod; (2) the TCP target 9 mm too low; (3) fingers fully open (80 mm) sweep into
neighbours.
Change: own vision everywhere -- cylinder fit of the mask depth points (global+global2) for locate, grasp
verification, rod-in-hand measurement, closed-loop xy alignment over the hole and the insert verifier; TCP
11.5 mm above the board; pre-open 30 mm; grasp candidates scored by a heightmap finger-footprint check.
Prediction: grasp PASS on both with no >20 N wrench before closing; insert reaches the pre pose and the
align prints xy errors < 5 mm before correction, < 1 mm after. Unknown: whether the dropped rod seats.

## Round 2 -- 0/2 (both TIMEOUT; insert FAIL)
Previous hypotheses: own mask-cylinder fit is reliable -> supported (locate agrees with tracker 0.4-0.6 mm,
grasp verified both lifts; s01 align correctly saw the rod on the board). TCP 11.5 mm above the board avoids
the board -> supported (no contact force at grasp), but the grip it gives is weak (s01).
focus: insert -- s00, s01
- s00: k12 t=9.36 rod vertical over the blue fixture; stdout align: rod axis tilted ~6 deg ([0.005,-0.104,-0.995])
  and I aligned the axis line *extrapolated to the bore bottom*, so the real tip was ~3.5 mm off the bore at
  the fixture top. t=13.80 descent blocked at tcp z 0.078 (rod tip ~0.037 = fixture top), F ~(-10,7,9) N
  (k14/k15 wrist view: tip on the fixture); opened there, rod fell against the fixture (k19, axis z 0.69).
- s01: rod (pinched 11 mm from its end, pads only on its lower edge) slipped out during the fast
  move_to_joints swing (sheet t=8.0, width -0.0015 at t=10.36); insert then aligned to nothing.
Root cause: (1) no tilt correction + wrong alignment point; (2) weak end-grip + fast swing.
Changes: insert aligns closed-loop with the camera fit -- rotates the TCP about the rod tip so the axis is
vertical and puts the actual tip over the bore (<0.4 mm, <0.6 deg); fixture top height measured from the
depth heightmap around the goal at locate; steps down 3 mm at a time, if the TCP falls short (>1.5 mm)
backs out and re-aligns (2 retries); releases 12 mm inside the bore. Grasp: TCP 10 mm above the board,
prefer 21 mm from the end; slower transport (gain 0.5); insert FAILs if the rod is not seen in hand.
Prediction: s01 keeps the rod through transport; align converges to tilt < 0.6 deg, xy < 0.4 mm on both;
the rod enters the bore (no shortfall at the Z_TOP-0.002 step) and drops to the goal.

## Round 3 -- 0/2 (s00 grasp FAIL, s01 insert FAIL)
Previous hypotheses: TCP 10 mm above board + grasp 21 mm from the end gives a firm grip -> REFUTED.
Slower transport keeps the rod -> REFUTED (s01 dropped it again, zero wrench, open space).
focus: grasp -- s00, s01 (insert's s01 failure is the grasp's weak hold)
- s00: trace t=3.80 close stops at width 0.0089 with F=(-12.6,5.7,-52.6) N; k06/k07 rod still on the board,
  rotated, fit ctr 27 mm from the TCP -> squeezed out from under the fingers.
- s01: t=5.48 close to width 0.0032, rod lifted (k06/k07 and fit z 0.082), then lost during the turn of the
  hand (move_to_joints t=6.36-11.36, width 0.0037 -> -0.0002, sheet t=10.6 rod on board).
Root cause hypothesis: with the fingertips ~9 mm below the TCP the pads only reach the rod's lower half
(axis 6 mm below the TCP), a marginal pinch that either squirts the rod out or lets it slip when turned.
Not sure which heights/points hold -> experiment 02_grip_height: centre grasp at TCP 9.5/11/13 mm above the
board, lift, measure the rod in the TCP frame by camera fit, then a 90 deg hand turn with the best height.

### Experiment 02_grip_height (s00, s01) -- result
Hypothesis: grip quality depends on TCP height; the 90 deg turn makes the rod slip.
Prediction: heights differ clearly; the turn shows slipping.
Observed:
- Fingertips touch the board at TCP ~9.3 mm above it: h=9.5 mm descents stall at tcp z 0.0180 with
  F=(1,65,33) N (s00 t=3.44, k03) / (55,-35,33) N (s01 t=4.24).
- set_gripper rides the arm UP ~3 mm while closing: s00 h=13 tcp z 0.0216 -> 0.0245 (t=12.92 -> 13.72),
  s00 h=9.5 0.0180 -> 0.0194, s01 h=13 0.0218 -> 0.0247. Also set_gripper(width) at t=1.96 moved the TCP 5 mm.
- Held only once: s01 h=9.5 (t=5.04 close, fit ctr z 0.0605, rod axis 7.3 mm below the TCP, centred y -0.2 mm).
  The other trials started from rods displaced by earlier trials and/or landed fingers on the rod (h=11 forces
  ~45-73 N) -> confounded.
- Turn test (s01, h=9.5): rod held through the 90 deg turn, axis stayed 6-7 mm below the TCP, centred (t=27.5-35.7).
Verdict: turning is NOT what drops a good grip (prediction about the turn refuted). Height matters, and the
set_gripper ride-up (3 mm upward while the fingers close) moves the pads off the rod's equator region during
the close -> marginal/failed pinches. 
Change: grasp at TCP 10.5 mm above the board, descend the last 15 mm and close with my own Jacobian servo
(api.step with action[7]=-1) that holds the TCP pose while the fingers close; retry the grasp up to 3x after
re-locating the rod with the camera fit if the lift check fails. Insert uses the same servo for the fine
alignment, the descent steps and the release (open while holding the pose).
Prediction: closed TCP z stays within 0.5 mm of the target; the rod is lifted on the first attempt on both
starts with the axis ~6-7 mm below the TCP; the rod survives the transport.

## Round 4 -- 0/2 (s00 grasp FAIL x3 attempts, s01 insert FAIL: rod lost in transport)
Previous hypotheses: holding the pose during the close fixes the grip -> REFUTED as sufficient. The servo held
z (closed tcp z 0.0196/0.0192 vs target 0.0196) but the servo left the TCP 0.6-0.8 mm low before the close.
focus: grasp -- s00, s01
- s00: 3 attempts, each: at grasp tcp z 0.0188, close -> width 0.0089 with Fz -47/-66/-70 N; after lift
  (k04 t=5.16, k10 t=9.96, k16 t=14.76) rod still on the board, re-located 5-14 mm away -> squeezed out.
- s01: close width 0.0034, F ~0, rod lifted (k04/k05 t=6.56), slipped out during the move_to_joints turn
  (sheet t=8.0, width -0.0011 at t=11.6, k06 rod on the board).
Both grasps were with the rod axis only ~6 mm below the TCP. The only grasp that survived a 90 deg turn
(exp02 s01) had the axis 7.3 mm below the TCP, i.e. fingertips practically on the board.
Root cause hypothesis: the pads hold the 8 mm rod only when the fingertips are within ~0.5 mm of the board;
a fixed height from the fitted board z (+-0.5 mm servo error) lands on either side of that.
Change: feel for the board -- after the servo to TCP 11 mm above the board, lower 0.3 mm per control step
until Fz_ext > 3 N (fingertips on the board) or the floor (board + 7.4 mm); then hold 0.4 mm above that
contact height and close with the pose-holding servo.
Prediction: contact at tcp z ~0.018-0.0185 on both; first attempt lifts the rod with the axis ~7 mm below
the TCP; the rod survives the turn on s01.

## Round 5 -- 0/2 (both insert FAIL)
Previous hypotheses: feel-down grasp -> partly supported: contact at tcp z 0.0179 on both (as predicted);
s01 lifted on attempt 0 (width 0.0034, axis 6.4-7 mm below TCP) and SURVIVED the turn; s00 attempt 0 again
squeezed out (width 0.009, Fz -48 N), attempt 1 lifted (width 0.0037).
focus: insert -- s00, s01
- s01: rod reached the fixture in hand (k07 t=12.80), camera alignment converged (xy <0.7 mm, tilt <0.7 deg,
  stdout align/alignR), but all 3 descents blocked with the rod bottom at ~0.045 = 7.5 mm ABOVE the fixture
  top (0.0376; r02 s00's rod tip did stop at ~0.037, so the top estimate is consistent), F ~(-30,105,165) N.
  Chosen psi -30: approach pointing back toward the base (tcp z [-0.87,0.50,0]) -> hand body and wrist on
  the far (+x) side, low over the board; sheet t=10-40 the global camera is filled by the arm link right at
  the fixture. Most likely the hand/arm hit board parts, not the rod tip.
- s00: rod lifted on attempt 1, lost during the move_to_joints sweep at tcp z 0.09->0.17 over the connector
  block (sheet t=13.3 in hand next to it, k12 t=15.80 on the board beside it).
Root cause: insertion pose choice (approach toward the base) and a low transport sweep.
Change: plan cost prefers the approach pointing away from the base (hand between base and fixture, arm
above/behind it, global cameras see the rod); vertical lift to tcp z 0.20 before the joint move; release
7 mm inside the bore (was 12) so the hand stays 5 mm higher; step prints full wrench + joint torques.
Prediction: chosen tcp z dir has a positive radial component; s00 keeps the rod through transport; the
descent passes the fixture top without a shortfall.

## Round 6 -- 0/2 (both insert FAIL: rod shed during the reorienting move_to_joints)
Previous hypotheses: radial-outward approach chosen -> supported (psi 45 / -45, tcp z dir [0.68,0.73] /
[0.71,-0.70]); "s00 keeps the rod in transport" -> REFUTED; descent not reached (untested).
focus: insert -- s00, s01
- s00: grasp attempt 1 lifted (width 0.0037, k10/k11); k12 t=12.84 rod in hand at z 0.20; sheet t=13.3 rod
  already on the board; k13 t=17.16 jaws empty (width -0.0011).
- s01: grasp attempt 0 lifted (width 0.0034, k05); k06 t=8.44 in hand at z 0.20; during move_to_joints
  (8.44-12.08) flung off the board (sheet t=10.6, k07).
The pinch is a fingertip pinch (rod axis 6-7.5 mm below the TCP = beyond the pads' reach), which is a pivot
about the closing axis; the new psi needs a bigger rotation (yaw + 90 deg) done in one fast joint move,
swinging the rod across the board -> inertial load sheds it. (r05 s01 with a smaller rotation kept it.)
Change: transport gently -- lift, translate with the grasp orientation so the rod centre sits where it will
be at the high pose, then rotate about the rod centre in ~11 deg IK waypoints (move_to_joints gain 0.35);
plan cost adds 3*rotation angle from the grasp orientation.
Prediction: the rod stays in the hand through the rotation on both (in-hand fit at the pre pose); the run
then reaches the descent; time to the pre pose < 28 s.

## Round 7 -- 0/2 (both insert FAIL)
Previous hypotheses: gentle rotation keeps the rod -> supported on s00 (rotated 108 deg, k17 t=20.64 rod
vertical in hand), refuted on s01 (rod pinched 21 mm from its end swung into a tilted hang already during the
translation, k07 t=11.28, and fell during the rotation, k08 t=12.48).
focus: insert -- s00, s01
- s00: align 0-4 (stdout): xy err [-2.0,-3.5] -> [0.4,1.2] -> stuck at [1.36,1.03] mm, tilt 1.03 deg, z err
  2.0 mm for 3 iterations with no change -> my servo did not move the arm for mm-size corrections.
  Descended anyway; step to 0.0356 short by 2.1 mm, Fz +2.2 (tip on the fixture rim; k24 t=30.08 rod
  vertical, tip on top of the blue fixture beside the bore); then time ran out in the retry.
- s01: off-centre pinch pivots under gravity during transport -> dropped.
Root causes: (1) servo = P step on the measured joints -> steady-state offset of 1-2 mm under gravity, so
small corrections are never executed; (2) off-centre fingertip pinch pivots.
Changes: servo integrates its joint target (q_t += dq each step, anti-windup clamp) -> no steady-state
offset; grasp at the rod centre (s = L/2, fallbacks 4/8 mm off) so gravity has no lever on the pinch; the
hanging rod's bottom is now the lower end (a centre grasp makes "farther from the TCP" ambiguous).
Prediction: align converges to < 0.4 mm / < 0.6 deg within 2-3 iterations; the rod survives transport on
both; the tip enters the bore (no shortfall at the Z_TOP-0.002 step).

## Round 8 -- 0/2 (both insert FAIL)
Previous hypotheses: centre grasp survives transport -> supported on s00 (held through 126 deg, wrist view
k11/k16/k19 rod fixed in the fingers), REFUTED on s01 (fell during the plain high translation t=8.56-11.08,
k07 rod across the fixture, width -0.0002). Integral servo: untested (align never ran).
focus: insert -- s00, s01
- s00: rotation waypoints: move_to_joints FAIL x3 at t=14.28 (err 0.044); at the pre pose the camera saw
  the rod 70 deg off vertical, 24/26 mm off in xy (align 0), the rod still fixed in the hand -> the HAND was
  not at the planned orientation. My chain IK waypoints were not checked against the joint limits (the plan's
  were) and the pre move used move_to (own IK branch).
- s01: grasp closed with ~0.6 N (light), rod lifted, then dropped by a plain translation -> grasp quality.
Changes (insert): reject chain waypoints near joint limits / ending in another IK branch, fall back to
joint-space interpolation between the in-limit endpoints; pre pose via move_to_joints(plan q); print the
hand orientation error after the rotation.
Open question for s01: where is the rod relative to the finger pads? Experiment 03_finger_geom (prefix
locate -> grasp): wrist-depth finger points and the held rod in the TCP frame, then the empty fingers at
widths 0.08/0.03/0.008/closed.

### Experiment 03_finger_geom (s00, s01) -- result
Hypothesis: the held rod sits at/below the fingertips (tip-edge hold) because the pads cannot reach the
rod's equator over the board. Prediction: fingertip ~9 mm along +z from the TCP, held rod axis >= ~7 mm.
Observed (stdout; the "holding" part of the log was truncated, so the held-rod numbers are lost):
- Empty fingers at width 0.008 / 0.03 (both starts identical): inner faces |y| = width/2 over z -10..+6 mm,
  only a few points at z 6..8.4 mm (rounded tip), tip max z +8.1..8.5 mm. Closed: max z ~5 mm visible.
- With the TCP held 9.2 mm above the board (contact z 0.0179 + 0.4), the lying rod's axis is at z +5.2 mm:
  inside the flat pad face but only ~0.8 mm from its end. Round 5-8 in-hand fits put the axis at +6.3..7.5
  after lifting -> the rod slides into the rounded tip region during the lift.
- Grasps that ended with width ~0.009 (s00 r08) held through a 126 deg turn; ~0.0034 (s01) never survived.
Verdict: partly supported -- the pinch is at the very edge of the pad; a close ending far below the rod
diameter means the rod went under the pad edge (tip hold). (Held-rod position itself not measured.)
Change (grasp): after the close, if width < 2*RAD-1.5 mm: open holding the pose, rise 12 mm, swap the
fingers (yaw 180 deg), feel down and close again; keep whichever. Insert changes from round 8 entry stand.
Prediction: s01 gets width ~0.008-0.009 on the 2nd yaw (or the 1st) and keeps the rod in transport; s00
reaches the pre pose with orientation error < 2 deg and the align converges.

## Round 9 -- 0/2 (s00 insert FAIL, s01 grasp never finished)
Previous hypotheses: hand reaches the planned orientation -> supported (s00 "off the plan by 0.8 deg");
yaw-swap re-close fixes s01 -> REFUTED (both closes empty, rod against a tall post, later attempts collided
with the post at 47-192 N, k11 t=22.28, k14 t=28.00; timeout inside grasp).
focus: grasp -- s01 (and the s00 insert cause, which is the grasp's pivot)
- s00: hand at the planned pose within 0.8 deg, yet the rod was 88 deg off vertical (align 0) and k18-k21
  show it sticking straight out of the fingertips along the approach axis. With a CENTRE pinch the rod kept
  its world orientation (horizontal) while the hand turned 90 deg -> the fingertip pinch is a nearly free
  pivot about the closing axis. Consistent with r07 s00 (pinch 11 mm from the end -> rod hung vertical after
  the turn, tilt 1-4 deg) and r07 s01 (swung into a tilted hang during transport).
Root cause: my whole insertion relied on the rod being rigid in the hand; it is a pendulum on a pivot.
Change (strategy): use the pivot. Pinch 10 mm from end A top-down, feel down, close; then pivot the rod up
against the board -- end B stays on the board while the never-rotating top-down hand moves on a circle
around B's rim until the rod is vertical; lift: it hangs vertically (pendulum, centre of mass below the
pivot). Insert carries it top-down over the bore (no hand rotation, hand stays high), waits for the swing to
settle, aligns the measured bottom by translation, steps down, releases 3 mm above the bore bottom. End A is
chosen by finger-footprint obstacle points at the grasp and over B (avoids s01's post).
Prediction: after the lift the fit shows tilt < 5 deg on both; align converges < 0.4 mm; the descent goes
past the fixture top; the rod is released inside the bore.

## Round 10 -- 0/2 (both grasp FAIL)
Previous hypothesis "the pinch is a free pivot" -> REFUTED: s01 attempt 0 (stdout) after the board-pivot the
rod hung 64 deg off vertical (friction holds it), and the hand had dragged it 4.6 cm (tcp y 0.201->0.155)
instead of pivoting it; s00 closed to 4.5 mm with F (-19,37,-54) and lost the rod during the pivot (k04
t=11.48 rod on the board, moved 4.4 cm); the verifier's "hanging" fit (ext 35 mm, rms 0.56) was spurious.
s01 attempt 1 hit the post (contact at tcp z 0.035, k07 t=15.60 F -96 N).
Re-reading r09 s00: the 90 deg rod-in-hand change happened with the JOINT-SPACE interpolation fallback (TCP
path not about the rod centre, likely contacts), whereas r08 s00 with IK-chain waypoints kept the rod
fixed in the wrist view. So the pinch is semi-rigid with friction, not a pivot.
focus: grasp -- s00, s01. Back to the rigid-hold strategy (round 8 version), but I need to know (a) what
the close looks like on s01 (why it ends at 3.3-3.4 mm) and (b) whether a slow IK-chain rotation about the
rod centre keeps a centre-pinched rod rigid.
Experiment 04_air_rehang (prefix locate): centre grasp with feel-down, wait() keyframes before/after the
close and 3 mm up, then a slow 90 deg rotation about the closing axis around the rod centre in 15 deg
IK steps with a camera fit of the rod (axis in the TCP frame + world tilt) after every step.

### Experiment 04_air_rehang (s00, s01) -- result
Hypothesis: centre pinch is semi-rigid in a slow rotation; s01's failure is in the close.
Prediction: rod fixed in the TCP frame during rotation (s00); s01 rod ends at the tips after the close.
Observed (stdout; wait() produced no keyframes, so only fits/trace):
- s00: close width 0.0089, F (-15,13,-46); rod centre in TCP frame z +3.8 mm (3 mm up) -> +1.1 mm (60 mm
  up): pushed UP into the pads -- a good grip. Rotation 15/30/45 deg: world tilt 84.7/73.4/48.7 deg, i.e. the
  rod turned with the hand (lagging ~10 deg at first, axis z-component in TCP 0.05->0.20->0.04). At 60 deg
  the hand hit something at tcp z 0.076 (F (-62,-212,258), t=13.96 move_to_joints FAIL) -- rotation done too
  low next to the peg plate.
- s01: close width 0.0046, F ~0; rod centre at z +5.9 (3 mm up) -> +8.1 mm (60 mm up) = at the rounded
  tips; lost at the first 15 deg step (width -0.0002 afterwards, fits show it lying).
Verdict: prediction held: semi-rigid when the close ends ~rod diameter; s01-type closes end with the rod at
the tips. Cause of the s01 close not visible (no keyframe).
Change: back to the rigid strategy (round 8 code paths, rebuilt): centre grasp; after the close, if width
< 6.5 mm re-open to 14 mm and close again (2 tries); lift to 20 cm before anything rotates; IK chain
(translate the rod centre, then rotate about it) checked for every yaw candidate in cost order, the first
valid chain is used -- no joint-space interpolation fallback (r09 rotated the rod in the hand).
Prediction: s00 reaches the pre pose with the rod vertical (align tilt < 5 deg at iteration 0) and the align
converges; s01 either gets a width > 6.5 mm on the re-close and survives, or fails at grasp/transport.

## Round 11 -- 0/2 (grasp FAIL by the auditor on s00, grasp FAIL on s01)
Previous predictions: s00 rod vertical at the pre pose -> REFUTED; s01 re-close helps -> REFUTED.
focus: grasp -- s00, s01
- s00: closed once to 9.0 mm (no re-close needed), lifted (k04 t=7.04), but my printed rod centre in the TCP
  frame was z +7.9 mm (at the tips) and 6.9 mm along the axis. During the 136 deg chain rotation the rod
  pivoted in the fingertip pinch until it pointed along the fingers (k10 starting, k15 t=22.84); at the pre
  pose tilt 51 deg (k19), align refused.
- s01: both closes 3.4-3.5 mm, ~0 N, re-open to 14 mm and re-close the same; second attempt identical
  (contact again tcp z 0.0179); rod never moved (k12 t=18.68). Wrist views (k03-k05) show the rod between
  the fingers; why the pads pass it is still not visible. Identical retries -> identical failures.
Root cause: the grasp outcome (rod in the pads at z ~1-4 mm vs at the tips ~6-8 mm vs missed) varies with
small details I cannot see; a tip grip later pivots.
Change (grasp): check the grip 4 mm above the board with the camera fit: rod lifted AND its centre < 5.5 mm
along TCP z AND centred (|y| < 3 mm); otherwise put it back, open, re-fit and retry with a variation
(shift 6 mm along the rod / yaw +-8 deg / yaw 15 deg), up to 4 attempts.
Prediction: s00 gets a "pads" grip within 2 attempts and the rod stays vertical through the rotation
(align tilt < 15 deg); s01 gets a pads grip on one of the varied attempts or fails with all 4 reported.

## Round 12 -- 0/2 (s00 insert FAIL, s01 grasp never finished)
Previous predictions: s00 pads grip + vertical rod -> SUPPORTED (close 9.0 mm, 4 mm lift raised the rod
13.1->19.6 mm, rod 2.9 mm deep in the pads; k21 t=27.92 rod hanging vertically above the fixture; align 0
tilt 9 deg -> 1.5 deg). s01 varied retries -> REFUTED (all 4 closes 3.4-4.5 mm, rod untouched).
focus: grasp (s01) and insert (s00)
- s00: at the pre pose the rod bottom stayed at z 0.067 (z err -14.7 mm for every align pass) and the xy
  error grew (1,-2.5) -> (-7,-11) -> (-11,-17) mm; the step down stopped 25 mm short with F/T (-55,-109,145)
  N while the rod tip was 28 mm above the fixture top -> the HAND (horizontal, body ~6 cm wide vertically)
  was sitting on the tall grey part next to the fixture (k22/final), and the arm was being deflected
  sideways. The 16-step chain also used 13 s (t=13-26).
- s01: k03 the hand body right over the tall grey post beside the rod; contact "Fz 35 at 0.0179" and the
  pads closing on nothing are consistent with the hand body (which extends ~3 cm along the rod direction
  and ~10 cm along the closing axis) resting on the post top, the fingers never reaching the rod's sides
  (r12 k11 later: 209 N jam against the post). My obstacle check only covered the finger footprints.
Root cause: no collision model of the hand body, neither at the grasp nor at the insertion pose.
Change: coarse hand model (two fingers + a 8 x 21 x 11.5 cm body box behind them) checked against the
depth heightmap (now 25 cm around the rod and 25 cm around the fixture): grasp candidates (also 15 mm off
centre) add the hits to their cost; insertion yaw candidates are penalised by the hits at the release
pose. Chain steps 15 deg, faster joint moves.
Prediction: s01 picks a grasp point away from the post and closes ~8-9 mm; s00 picks a yaw whose hand does
not cover the grey part; the align reaches z err < 2 mm at the pre pose and xy converges.

## Round 13 -- 1/2 (s00 SUCCESS 27.4 s; s01 grasp never finished)
Previous predictions: s00 yaw avoiding the grey part + converging align -> SUPPORTED (psi -45 with the fewest
body hits that had a valid chain; align 0 tilt 6.9 deg, xy (1.1,2.6) -> align 2 (0.25,0.27) mm, tilt 0.17;
all steps down with < 0.2 mm shortfall; release at bottom 0.0308 -> SUCCESS). The insert verifier shows '-'
because the episode ends (EpisodeOver) at the moment the released rod reaches the goal, before my
post-release check runs -- nothing to fix there.
s01 grasp -> REFUTED again: 4 attempts at s=10/20 mm from end0 (body-hit check moved the grasp away from
the post), every close 3.4-4.5 mm, rod untouched; attempt 0 felt NO contact down to tcp z 0.0163 (fingertips
~1 mm below the rod's support height!) yet still closed past the rod; later attempts collided with the
grey block (t=22.16 236 N).
focus: grasp -- s01. I cannot explain the s01 close from the evidence I have (wrist RGB shows the rod between
the fingers). Experiment 05_s01_close (s00 for comparison): wrist depth + mask in the TCP frame and the hand
orientation error/roll at 20 mm above, at contact, at width 14 mm, 9 mm and closed.

### Experiment 05_s01_close (s00, s01) -- result
Hypothesis: on s01 the fingers do not straddle the rod at pad height (offset or roll).
Prediction: s01 rod outside |y|<15 mm / centre z > 6 mm / fingertip heights differ > 3 mm.
Observed (stdout): hand orientation error <= 0.2 deg and closing axis level (z-comp <= 0.002) on BOTH starts;
rod centre in the TCP frame at contact s00 (0.1,-0.5,5.2) mm, s01 (0.2,0.3,5.4) mm -- identical geometry.
s00: at width 9 mm the rod is pushed (y -2.7 mm) with F (-26,17,-30) N, closed width 9.1 mm (contact).
s01: at width 9 mm F = 0; closing on to width 4.8 mm puts both inner pad faces 2.5 mm from the rod axis while
the rod stays exactly where it was (wrist mask y [-4.1,3.9]) and F ~0 -> the pads pass THROUGH the rod
(no contact) on this start. Prediction REFUTED: the geometry is the same; the contact is missing.
Earlier s01 runs did lift the rod by the fingertips when the hold height was 1-2.5 mm higher (r04-r06, r08:
width ~3.4, rod at the tips) -> the tip region still collides there.
Change (grasp): 3 attempts -- (1) contact height, needs a pads grip; (2) same point, hold 2.5 mm higher;
(3) shifted 6 mm, yaw 8 deg, hold 1.2 mm higher; (2)/(3) accept any lifted, centred grip. Single close per
attempt (the re-open/re-close never helped). Insert unchanged (r13 s00 SUCCESS path).
Prediction: s00 unchanged (attempt 0, SUCCESS ~27 s); s01 lifts the rod on attempt 2 or 3 with the rod
centre ~6-8 mm along TCP z; then whether the tip grip survives the rotation decides s01.

## Round 14 -- 3/5 (s00 27.4 s, s01 28.6 s, s04 34.8 s SUCCESS; s02 insert FAIL; s03 grasp FAIL)
Previous predictions: s00 unchanged -> SUPPORTED; s01 lifted by a raised variation and inserted -> SUPPORTED.
focus: grasp (s03), insert (s02)
- s02: attempt 2 lifted the rod (k11 t=18.16, k12 rod hanging); the fit saw only 17 mm of it (ext 17) and
  rod_ends() centred the 50 mm rod on the visible piece -> "measured" ends +40/-10 mm vs predicted +24/-26,
  rejected -> predicted used; then all 8 yaw candidates "chain bad" -> FAIL at t=20.56 (stdout). My chain
  check also rejected chains that end in another IK branch than the plan's.
- s03: the rod lies against the blue fixture; every grasp candidate > 1000 obstacle points (stdout); all 3
  attempts stopped at tcp z ~0.046 with a finger on the fixture (k03 t=4.12, k09 t=20.80; [-52,49,34] N).
- s04 SUCCESS but my verifier (run 0.3 s after release) saw the rod still 18 mm high -> it was sliding.
Changes:
  grasp: partly occluded in-hand fit -> anchor on the visible end farther from the TCP (true end), other
  end L away (both in the 4 mm check and the final measurement). If the best grasp candidate has > 40
  obstacle points: push_clear -- close the fingers, press the fingertips on the rod top and drag/roll it
  40 mm away from the nearby obstacle points, re-locate, then grasp.
  insert: chain no longer rejected for ending in another IK branch -- pre/fin re-solved from the chain end
  (and checked); 12 candidates; the reason of each rejection printed. Verifier waits 1.0 s.
Prediction: s02 measured model consistent with predicted, a chain found, insertion like s00; s03 rod pushed
> 2 cm from the fixture and grasped on attempt 0-2; s00/s01/s04 unchanged.

## Round 15 -- 2/5 (s00 26.6 s, s01 30.4 s SUCCESS; s02, s03, s04 insert FAIL)
Previous predictions: s02 chain found -> SUPPORTED (144 deg chain ran) but the rod dropped; s03 push ->
SUPPORTED (pushed to [0.363,0.018], lifted on attempt 1, rod over the fixture) but TIMEOUT at the pre pose;
s04 unchanged -> REFUTED (released, toppled).
focus: insert -- s02, s03, s04
- s02: fingertip grip (rod centre z 6.8 mm in the TCP frame); rod in hand at k05 t=10.76, separating k06
  t=11.20, on the board k07 t=11.72 -> slipped early in the rotation chain (no contact).
- s04: aligned (tilt 0.1 deg, xy 0.4 mm), 4 steps down with ~0 shortfall and |F| 2-3 N, released at an
  estimated bottom 0.0308; k16 t=29.64 rod leaning 27 deg at the fixture mouth, then fell (final). Fingertip
  grip (z 7.0 mm) -> the tip most likely sat on the rim while the rod slid UP in the fingers, which hides
  the block from my "TCP short" test.
- s03: push 6 s + two grasp attempts -> 20 s; first five yaw candidates "chain step 2 unreachable", the one
  used rotated 168 deg slowly (t=25.6-37.3); timeout at the pre pose (align 1: tilt 1.35, xy 1.5 mm).
Changes (insert): slip detection during the descent -- the highest visible rod point relative to the TCP is
recorded before the descent and re-measured at every step below the fixture top; > 1.5 mm rise = blocked
-> back out, re-align with a 0.5 mm xy offset pattern, retry (3 retries). Chain joint moves at gain 0.28
for fingertip grips (rod centre > 5.5 mm along TCP z), 0.45 for pad grips.
Prediction: s04 detects the slip (if it recurs) and retries instead of releasing on the rim; s02 keeps the
rod through the slower chain; s03 still likely times out (time budget).

## Round 16 -- 2/5 (s01 28.8 s, s04 33.1 s SUCCESS; s00, s02, s03 insert FAIL)
Previous predictions: slip detection works -> SUPPORTED (s00 stdout "rod slid up 2.1 mm, F 14 N" at every
block); s04 succeeded; s02 kept the rod -> REFUTED (dropped during the lift/transport, k05 t=10.76, wrench
0, fingertip grip 6.8 mm); s03 timeout -> as predicted (168 deg chain stalled, t=37.32).
focus: insert -- s00 (regression)
- s00: aligned xy (0.35,0.03)->(0.11,-0.03) mm, tilt 0.35 deg (k15 t=21.76 rod vertical over the bore);
  every descent blocked at rod bottom ~Z_TOP (step to Z_TOP-2 mm: slip 2.1-2.3 mm, F ~15 N); 3 back-out +
  re-align (0.5 mm offset pattern) gave the same block; released on the rim, rod toppled (k16 t=39.16).
  This yaw (psi -30 was unreachable, a different one used) differs from the earlier successful s00 runs ->
  the camera-measured tip position has a view-dependent bias larger than the bore clearance.
Change (insert): on the first block below the fixture top, undo the slip, then a spiral search around the
current xy (rings 0.4/0.8/1.2 mm, 24 points), pressing lightly at each point; entry = TCP drop > 2 mm minus
the rod's slip in the grip > 1.5 mm; then continue the descent to the release height from the measured
bottom. Falls back to back-out + re-align if not found.
Prediction: s00 enters the bore within the 1.2 mm spiral and succeeds; s01/s04 unchanged (no block).

## Round 17 -- 2/5 (s00 27.0 s, s01 31.2 s SUCCESS; s02, s03, s04 insert FAIL)
Previous predictions: s00 enters via the spiral -> SUPPORTED (stdout "spiral r 0.4mm ang 23: tcp drop 2.9mm
slip 0.1mm, entered", then steps with slip 0.5 mm, SUCCESS 27.0 s). s04 unchanged -> REFUTED.
focus: insert -- s04 (s02: fingertip grip dropped during the chain again, k05 t=10.76 -> k06 t=11.40;
s03: time + a chain that stalled 12x in open space)
- s04: fingertip grip (7.0 mm); aligned (tilt 0.11, xy 0.44/0.23 mm); step to Z_TOP-2 mm: slid up 2.8 mm with
  F ~0; spiral's FIRST point claimed entry (drop 3.6, slip "-0.0"), next step slid 3.4 mm -> false entry;
  back-out + re-align twice, then top_rel() returned garbage ("slid up -70 mm", F 0) and the loop walked the
  estimated bottom down to the release height and released on the rim (sheet t=29-37 rod at the bore mouth,
  final: toppled across the fixture).
Root cause: a single spiral pass that stops at the first (false) entry, and unguarded slip readings.
Change (insert): the spiral is resumable -- a later block continues with the next spiral points around the
same centre (up to 4 calls, 24 points), instead of back-out/re-align; top_rel() returns None for fits with
< 20 mm visible or an implausible value (so no garbage slip).
Prediction: s04 keeps searching after the false entry and enters within the 24 points (or ends blocked
without the garbage walk-down); s00/s01 unchanged.

## Round 18 -- 2/5 (s00 27.0 s, s01 31.2 s SUCCESS; s02, s03, s04 insert FAIL)
Previous predictions: s00/s01 unchanged -> SUPPORTED; s04 search continues -> it did, but every spiral
point "entered" falsely (4x "entered, rod bottom ~0.0380" = fixture top; alignR z err -8.6 mm = the rod had
slid 8.6 mm up in the fingers); final.png rod standing on the fixture top. s02 dropped again during the
chain (k05 t=10.76 -> k06 t=11.40); s03 timed out (20 s grasp + 12 s rotation).
focus: insert -- s04
Root cause: with a fingertip grip the rod's top end sits inside the fingers' shadow, so "highest visible
point above the TCP" does not move when the rod slides up -> slip reads ~0 in the spiral -> TCP drop is
taken as entry. Geometrically, from the cameras a rod sliding up in the fingers and a rod sinking into the
bore look the same (bottom hidden by the fixture, top hidden by the fingers).
Change (insert): force-based entry test -- at a spiral point with a TCP drop, push the hand +-1.2 mm along
the closing axis and read the lateral force change: a tip inside the bore resists (> 4 N), a tip resting on
the rim slides along. Fingertip grips (rod centre > 5.5 mm along TCP z) use only this test; pad grips use
slip-corrected drop OR the lateral test. After a confirmed entry, a later block releases there (the rod is in
the bore) instead of searching again.
Prediction: s04 spiral reports lateral forces ~1-2 N on the rim and > 4 N at the true entry, enters, SUCCESS;
s00 unchanged (lateral F printed at its entry, expected > 4 N).

## Round 19 -- 2/5 (s00 27.6 s, s01 28.8 s SUCCESS; s02 grasp/insert, s03, s04 insert FAIL)
Previous prediction for s04 -> REFUTED: 11 spiral points each "tcp drop 3.5 mm, slip 0, lateral F 0.1 N";
sheet t=26-40 rod vertical on the fixture top, never in. s00 unchanged -> SUPPORTED.
focus: insert (s04); grasp (s02: fingertip grip 6.8 mm deep, dropped in the chain again, k05/k06)
- A fingertip pinch lets the rod slide along its axis AND pivot about the closing axis with ~0 force, so
  pressing on the rim only moves the rod in the fingers (TCP drops 3.5 mm with no slip of the visible top =
  pivot), and neither slip nor lateral force can sense the bore. Note s01 (also a fingertip grip, 7.7 mm)
  went straight in at a final alignment error of (0.08,0.35) mm without any search; s04 blocked at
  (0.44,0.23) -> for fingertip grips the only lever is alignment accuracy, not force.
Change (insert): fingertip grips (rod centre > 5.5 mm along TCP z) skip the spiral; on a block they back out,
re-align with the tip 2 mm above the rim (align tolerance now 0.25 mm / 0.5 deg, 6 iterations at the pre
pose), descend again; after 2 blocked tries they release the rod from 2 mm above the bore (drop). A time
guard releases when < 3 s are left.
Prediction: s04 either enters on a re-aligned descent or is dropped into the bore from 2 mm (success if the
alignment is within the clearance); s00/s01 unchanged.

## Round 20 -- 3/5 (s00 27.6, s01 33.3, s04 34.7 s SUCCESS; s02, s03 insert FAIL)
Previous predictions: s04 -> SUPPORTED (SUCCESS 34.7 s); s00/s01 unchanged -> SUPPORTED.
focus: insert -- s02 (s03 is the time budget: push + 2 grasp attempts 20 s, then the only reachable
candidate rotates 168 deg and stalls; not addressed)
- s02: fingertip grip (6.8 mm deep, 4.6 mm off-centre, 22 mm visible), held at k04 t=6.24; between the lift to
  20 cm and the translation toward the fixture the rod fell (k05 t=10.76 falling over the fixture, k06
  t=11.40 on the board), before the rotation proper. Same in r17-r19.
Change (insert): for fingertip grips the lift is slower (k 0.1) and only to 17 cm, and the translation to the
chain start runs at gain 0.16 (was 0.3); chain gain stays 0.28 (s01/s04 already need 33-35 s).
Prediction: s02 keeps the rod through the lift/translation (rod seen in hand at align 0); s01/s04 still
succeed but 1-3 s slower.

## Round 21 -- 3/5 (s00 27.5, s01 34.3, s04 38.1 s SUCCESS; s02, s03 insert FAIL)
Previous predictions: s02 keeps the rod -> REFUTED (sheet t=8.0 held, t=10-13.3 / k05 t=13.84 back on the
board, width -0.0002); s01/s04 slower -> SUPPORTED (34.3 / 38.1 s, s04 within 2 s of the limit). s03 timed
out mid-rotation (k09-k20) as before.
focus: insert/grasp -- s02
The slow transport did not help and costs time -> reverted to the round-20 motion. Remaining difference of
s02 from the fingertip grips that survive (s01, s04): it is gripped 5 mm off-centre (candidate s=0.020 won by
plan cost 17.5 vs 21.6 for the centre); the centre-weight term (60 per m) was negligible.
Change (grasp): centre weight 1000 per m (5 mm off-centre = +5 cost) so the centre pinch wins unless the
insertion plan is much worse; transport reverted.
Prediction: s02 grasps at s=0.025 and (if still a fingertip grip) keeps the rod with no off-centre lever;
s00/s01/s04 times back to r20 values.

## Round 22 -- 2/5 (s00 27.6, s04 34.7 s SUCCESS; s01, s02, s03 FAIL)
Previous predictions: s02 centred -> grasped centred (rod centre -0.3 mm, 6.3 mm deep, k03/k04) but still
dropped early in the lift/rotation (sheet t=8.0, k05 t=10.80) -> REFUTED (off-centre lever was not the
cause). s01 REGRESSED: the centre candidate failed twice (rod not lifted), attempt 2 pinched 9.4 mm off
centre, rod swung and fell during the chain (k11 t=27.56 -> k12 t=28.44). s04 back to 34.7 s (revert of the
slow transport worked); s03 unchanged (timeout in the rotation stall).
focus: insert/grasp -- revert
Change: centre weight back to 60 per m (the r20/r21 configuration that passed s00/s01/s04). No other change.
Prediction: back to 3/5 (s00, s01, s04) with times ~27/33/35 s; s02 and s03 still fail (s02: fingertip grip
drops in the first second of the insert motion regardless of centring/speed; s03: time budget).

## Round 23 -- 2/5 (s00 27.6, s04 34.7 s SUCCESS; s01, s02, s03 FAIL)
Previous prediction "back to 3/5" -> REFUTED for s01: this time attempt 0 was the s=0.020 candidate (r21:
s=0.010), it lifted the rod (z 0.0131->0.0157) at 6.3 mm depth and my attempt-0 rule (< 5.5 mm) rejected
it; attempt 1 missed; attempt 2 lifted at 6.8 mm (k10 t=21.80). Chain 93 deg with 7 slow moves; pre pose
only at t=37.3 (k18); align 2-4 stuck at xy ~0.3 mm (tolerance 0.25) while the clock ran out -> TIMEOUT
with the rod 14 mm above the bore (final.png).
focus: insert/grasp -- s01 time
Changes: attempt 0 accepts a lifted, centred grip up to 7 mm deep (fingertip grips like s01/s04 do insert);
align returns when the xy error is < 0.5 mm, tilt < 0.6 deg and it stopped improving (< 0.1 mm change).
Prediction: s01 takes attempt 0, reaches the pre pose ~15 s earlier, SUCCESS; s00/s04 unchanged.

## Round 24 -- 3/5 (s00 27.6, s01 32.7, s04 33.0 s SUCCESS; s02, s03 FAIL)
Previous prediction -> SUPPORTED (s01 SUCCESS 32.7 s; s00/s04 unchanged, s04 1.7 s faster).
Remaining failures, both known:
- s02: fingertip grip (6.8 mm deep, width 3.3 mm) loses the rod in the first seconds of the lift/rotation
  (sheet t=8.0 held, k05 t=10.76 width -0.0002, rod at the fixture). Centring and slower motion (r21, r22)
  did not help; a re-grasp after a detected drop would not fit in the remaining ~30 s with a second chain.
- s03: push + 2 grasp attempts take 20 s, the only reachable insertion yaw needs a 168 deg rotation that
  takes until t=37 s -> time budget.
Decision for the last round: no change. The current code is the best configuration seen (3/5 on the debug
set, all successes 27-33 s); any further edit would go unvalidated.
