# Round 20 of 25: 3/5 starts succeeded, mean time 31.9 s

## Verdicts per start (the auditor's for failed starts, yours for successes)
s00 (SUCCESS 27.6 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s01 (SUCCESS 33.3 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s02 (TIMEOUT 40.0 s): locate:PASS  grasp:PASS  insert:FAIL
    insert: The rod slipped out of the fingers during the lift and rotation chain, between t=6.24 and t=10.76: k05 shows it falling over the fixture with the hand far away, and k06 shows it lying on the board beside the fixture. The rest of the chain and the descent (t=10.8–21.2) ran with an empty hand (width about 0). Nothing was inserted, and the bore is empty at the end.
    note: Locate passed at t=0.00. Grasp: the hand came down at about the rod centre (t=1.56–2.32), closed, and at k04 t=6.24 the rod is lifted and held vertically between the fingertips in the wrist view, while the global view shows it raised off the board. The grip was shallow, at the fingertips: rod centre 6.8 mm deep in the TCP frame and only 22 mm of the rod visible to the fit. Insert: on the lift and rotation chain the rod fell out. At k05 t=10.76 the global view shows the rod in mid-air, tilted, over the blue fixture, with the hand high and far to the upper left; it is not attached to the hand. By k06 t=11.40 the rod lies flat on the board just left of the fixture, and k07–k09 (t=12.12–13.52) show it still there while the wrist view keeps tilting. Gripper width reads -0.0002 from t=16.88 on, so the jaws are fully closed on nothing. The policy kept running the rotation chain with an empty hand: ten move_to_joints calls, one FAIL at t=18.88 with err 0.022 in open air (no wrench), then a descent at t=21.24 to TCP z 0.082 beside the fixture. At t=21.44 'align 0: rod not seen in hand' made insert FAIL. That verdict is correct, but the reason is that the rod was dropped about 4 s earlier during the reorientation, not that it was off-vertical. The final frame (k17 t=21.44, and the sheet through t=40) shows the rod lying on the board or fixture edge at an angle in the wrist view, the fixture bore empty, and the episode timing out at 40 s.
s03 (TIMEOUT 40.0 s): locate:PASS  grasp:PASS  insert:FAIL
    insert: The stage never finished: TIMEOUT at 40 s with no stage_done. The chosen plan (psi -15, 168° rotation, 2976 hand hits) drove the arm into a contorted pose, and the joint moves stalled in free space (move_to_joints FAIL err 0.057, low wrench, t=37.32). The only alignment reading was a 13.8° tilt and 11.5 mm xy error. At the end the rod is still in the hand at TCP z 0.088 and the bore is empty (k21, k22).
    note: Locate passed at t=0.00, with the rod lying close to the fixture. Grasp: every candidate was blocked by obstacles, so the policy pushed the rod sideways. k02 t=8.32 shows the rod moved away from the fixture onto open board. Attempt 0 (t=8.6–15.0) was judged 'not held / not in the pads'. Attempt 1, 2.5 mm higher (t=15.0–15.3), held. At k08 t=20.40 the rod hangs from the fingertips, lifted (z 0.013 -> 0.082), gripped about 9.6 mm from its centre and 8.4 mm deep, i.e. near one end at the tips. Insert: all the cheap yaw candidates were unreachable or jumped IK branch, so the policy chose psi=-15: cost 112.9, 2976 hand-obstacle hits, and a 168° rotation. k09 t=25.64 and k13 t=29.16 show the rod still in the hand, tilting toward vertical high above the fixture. From about t=30 the arm folds into an awkward pose: the global camera is blocked by the arm, and the wrist camera ends up looking back at the robot's own links (k17 t=32.68, k21 t=37.32). Twelve move_to_joints calls ending at t=37.32 report FAIL with err 0.057. The wrench is small (|F| < 3.1 N), so this is a stall in open air from reach or IK limits, not a collision. The only alignment reading (bottom z 0.052, tilt 13.8°, xy error 11.5/-3.2 mm) shows the rod was neither vertical nor over the bore. At t=38.76 the TCP is at z 0.088 with the rod still in the fingers (wrist view). The 40 s timeout hit before insert reached its verifier. The global view at t=37.32 shows the fixture bore empty.
s04 (SUCCESS 34.7 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.

Artifacts: attempts/r20/<id>/ (trace.json, summary.txt, keyframes/, sheet.png, episode.mp4)

Focus stage: insert
Rounds left: 5.

Append this round's entry to diagnosis.md first. Then either revise the focus stage's file (or the stage the
evidence points to) and end your reply with READY, or write experiments/NN_<slug>/ and end with EXPERIMENT.
