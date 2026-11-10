# Findings -- R8 rod (8 mm x 50 mm) into the vertical bore, API level 'final'

Final policy: locate -> grasp -> insert (plan.md). Best result: 3/5 on the 5-start debug set
(rounds 14, 20, 21, 24, 25; s00 27.6 s, s01 32.7 s, s04 33.0 s in r24/r25, mean 31.1 s). The debug set never
grew past 5 starts. Rounds 24/25 are the best: same 3 successes with the shortest times. Round 25 is the
final code.

## Causes found and fixes (with evidence)

1. **The tracker is unusable once the rod is in the hand** (r01: estimate frozen on the board / below the
   board while the rod was visibly lifted). Fix: own vision everywhere. A cylinder fit (known radius,
   PCA axis + circle fit) on the depth points of the part mask from the two global cameras. Locate agrees
   with the tracker to 0.4-0.6 mm (all rounds). The fit verifies the lift and measures the rod in the hand,
   the hanging rod's tilt and tip position, and the final pose.

2. **Fingertips reach ~8.4-9 mm below the TCP; the 8 mm rod's axis is only ~5 mm below the TCP when the
   tips touch the board** (r01 board contact at tcp z 0.0178; exp02/exp03 wrist-depth finger geometry:
   flat pad face to +6 mm, rounded tip to +8.4 mm). Fix: feel down to board contact (Fz > 3 N) and close
   0.4 mm above it.

3. **set_gripper lets the arm ride up ~3 mm while closing; move_to stops within 0.01 rad (several mm)**
   (exp02). Fix: own Jacobian servo through api.step that holds the TCP pose while closing. In r07 the
   align stalled at 1-2 mm. Fix: the servo integrates its joint target, which removed the steady-state
   offset (r13 align converged to 0.25 mm).

4. **Grip quality varies by start.** On s00 the pads close on the rod (width ~9 mm, rod 1-3 mm deep, rigid).
   On the other starts the flat pads pass through the rod without contact and only the rounded tips catch
   it (exp05: inner faces 2.5 mm from the rod axis, rod unmoved, F~0). Fix (partial): check the grip
   4 mm up with the camera. If the rod is not held, retry at +2.5 mm, then shifted 6 mm / yaw 8 deg /
   +1.2 mm (r14: s01 started succeeding).

5. **The rod in hand is not a free pivot but slips** (r09/r10 pivot strategies failed; exp04 showed that a
   pads grip follows a slow hand rotation). Fix: rotate the hand about the rod centre with a checked IK
   chain (joint limits, branch).
   - Pre/fin poses are re-solved from the chain end (r14 s02).
   - The chain runs only after a lift to 20 cm (exp04: the hand hit the peg plate when rotating low).

6. **Hand body collisions:**
   - At the grasp, the hand sat on a tall post (s01 r12).
   - At insertion, the horizontal hand sat on the part beside the fixture (s00 r12).
   - Fix: a coarse hand model (fingers + body box) checked against the depth heightmap, used in the grasp
     candidate cost and the insertion yaw cost. Result: r13 first SUCCESS (s00).

7. **Rod against the fixture** (s03): every grasp had > 1000 obstacle points. Fix: push the rod clear with
   the closed fingertips pressed on its top (r15: moved ~3 cm, then grasped).

8. **Camera tip position has a bias of ~0.3-0.5 mm, larger than the bore clearance** (r16 s00 blocked at
   the rim 4x at 0.1-0.3 mm measured error).
   - Fix: on a block, a spiral search of 0.4/0.8/1.2 mm rings with a light press. Entry = TCP drop minus
     the rod's slip in the grip (r17: s00 entered at the first ring point).
   - Fix: blocks are detected by the rod sliding up in the fingers (highest visible point vs the TCP),
     because a fingertip grip never makes the TCP fall short (r15 s04).

9. **Fingertip grips cannot be force-inserted.** Pressing just slides or pivots the rod in the fingers; slip
   and lateral-force probes read ~0 (r18/r19 s04). Fix: fingertip grips skip the search. They re-align
   tightly (0.25 mm / 0.5 deg), re-descend, and as a last resort drop the rod from 2 mm above the bore.
   r20: s04 SUCCESS.

10. **Time budget.**
    - Attempt 0 accepts a lifted, centred grip up to 7 mm deep.
    - Align stops when it no longer improves (< 0.5 mm, < 0.1 mm change).
    - r24: s01 SUCCESS 32.7 s instead of a timeout.
    - Reverted, because they cost time without helping: slower transport (r21) and a strong centre-grasp
      preference (r22, broke s01).

## What still fails and why
- **s02:** the fingertip grip (rod 6.8 mm deep, width 3.3 mm) loses the rod in the first seconds of the
  lift/rotation (r17-r25, e.g. r25 k05 t=10.76). Centring the grasp (r22) and slower motion (r21) did not
  help. Nothing re-grasps after a drop, and a full second pick + chain would not fit in 40 s. The drop is
  also detected late (only at align), because nothing checks the rod during the chain.
- **s03:** the rod starts next to the fixture. The push and 2 grasp attempts take ~20 s. The only reachable
  insertion yaw needs a 168 deg rotation, which stalls in open air until ~37 s (r15-r25), so it times out
  with the rod in hand.
- **Underlying limitation:** whether the pads grip the lying rod at all varies with the start (exp05). Only
  fingertip contact is reliable there, and fingertip grips are weak and can slip out.
- **Insert verdict:** the insert verifier shows '-' on successes. The episode ends (EpisodeOver) the moment
  the released rod reaches the goal, before the post-release check runs. This is not a motion error.

## Experiments
- 01_scene: geometry, goal, cameras.
- 02_grip_height: fingertip reach, set_gripper ride-up, rod held through a 90 deg turn.
- 03_finger_geom: pad and tip geometry in the TCP frame.
- 04_air_rehang: a pads grip is semi-rigid in a slow rotation; a tip grip is lost.
- 05_s01_close: on s01 the pads pass through the rod (no contact), while the geometry matches s00.
