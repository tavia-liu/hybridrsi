# Round 21 of 25: 3/5 starts succeeded, mean time 33.3 s

## Verdicts per start (the auditor's for failed starts, yours for successes)
s00 (SUCCESS 27.5 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s01 (SUCCESS 34.3 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s02 (TIMEOUT 40.0 s): locate:PASS  grasp:PASS  insert:FAIL
    insert: The rod dropped out of the fingers during the lift/144° rotation between t=8 (sheet: still held) and t≈10–13.8 (sheet t=10/13.3, k05: rod back on the board, gripper fully closed at -0.0002). Every later move_to_joints was done with an empty hand. The policy's FAIL at t=24.56 ('rod not in hand') is correct. final.png: the rod is on the board, not in the bore.
    note: t=0: the rod is lying on the board, untouched, and the locate fit agrees with the tracker to 0.4 mm. t=1.56–6.24 (k01–k04): the fingers come down beside the rod centre (candidate end0 s=0.020), close to a width of 3.4 mm and lift 7 cm. In k04 (t=6.24) the rod is clearly between the fingers and off the board in both views. It is held shallowly, near the fingertips: the rod centre is 6.8 mm deep in the TCP frame, and the plan says ~8 mm means 'at the tips'. Insert then runs the lift and a 144° rotation chain (psi 60) inside the first move_to_joints, with no keyframe between t=6.24 and t=13.84. The sheet still shows the rod in the fingers at t=8.0. By sheet t=10.0/13.3 and k05 (t=13.84) the rod is lying on the board again, near where it started, and the gripper width is -0.0002, so the fingers are closed on nothing. The rod slipped out of the shallow tip grip while the hand was being re-oriented. Insert then makes 10+2 move_to_joints near the fixture with an empty hand (t=13.84–24.36; one FAIL at 22.00 with err 22 mm, in open space with no wrench) and reports FAIL at 24.56 with 'rod not seen in hand'. That reason is correct. Nothing happens from then until the 40 s timeout. final.png: the rod is on the board and the empty hand is beside the blue fixture. Note: the tracker's part estimate in the trace is frozen at [0.6042, 0.0627, 0.0131] for the whole episode and cannot be used.
s03 (TIMEOUT 40.0 s): locate:PASS  grasp:PASS  insert:FAIL
    insert: It never got past the rotation chain. k09–k20 (t=29.8–39.4): the hand re-orients in the air at z≈0.16 with the rod still held and tilted. The last move_to_joints FAILs (err 0.15) and the 40 s timeout hits before the pre-pose, alignment or descent. final.png: the rod is still in the fingers, far above the bore and not vertical. There was no collision; it ran out of time. The insert stage never reported a verdict.
    note: t=0: the rod is lying right next to the blue fixture. Locate: ctr [0.354, 0.042], rms 0.23 mm, 0.5 mm from the tracker. Grasp: every candidate was blocked by the nearby fixture (finger_pts > 1000). The policy pushed the rod sideways to [0.363, 0.018] (sheet t=2.7–8.0, the rod moved away from the fixture). Attempt 0 at t≈8.3–14.7: the fingers closed at contact height. In the 4 mm-up check the rod was not in the pads (TCP frame [-4.8, -3.4, -8.5] mm), so the policy reopened. Attempt 1 at t=15.0–20.4: closed 2.5 mm higher (width 4.5 mm) and lifted. In k08 (t=20.40) the rod is in the fingers, off the board, in both views. It is held about 9.6 mm off centre and 8.4 mm deep (at the tips). Insert: the plan's top candidates all fail (chain unreachable). It picked psi -15 (cost 112.9, 2976 hand-hit points) with a 168° rotation. k09 t=29.76: the hand is lifted to z≈0.16 above and beside the fixture, and the rod still hangs tilted from the fingertips. k10–k20 (t=30.6–39.4): eleven slow move_to_joints steps re-orient the hand almost in place (TCP stays near [0.356, 0.077, 0.162]). The rod stays in the hand at a changing tilt (k14 t=34.16: rod tilted, wrist camera view rolled). The last call ends FAIL with err 0.15 at t=39.44. The episode times out at 40 s in the middle of this rotation chain. final.png: the arm is contorted, the wrist camera points up at the arm's own links, and the rod is still in the fingers well above the fixture. There is no contact with the fixture (wrench < 2 N). The approach, alignment and descent never happened. The time budget ran out: 20 s spent on the push plus two grasp attempts, about 9 s on lift/planning and about 10 s on the slow rotation chain.
s04 (SUCCESS 38.1 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.

Artifacts: attempts/r21/<id>/ (trace.json, summary.txt, keyframes/, sheet.png, episode.mp4)

Focus stage: insert
Rounds left: 4.

Append this round's entry to diagnosis.md first. Then either revise the focus stage's file (or the stage the
evidence points to) and end your reply with READY, or write experiments/NN_<slug>/ and end with EXPERIMENT.
