# Round 14 of 25: 3/5 starts succeeded, mean time 30.3 s

Every start passed, so the debug set grew to 5 starts and the new ones ran in this round too (ids in starts.json).

## Verdicts per start (the auditor's for failed starts, yours for successes)
s00 (SUCCESS 27.4 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s01 (SUCCESS 28.6 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said '-'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.
s02 (TIMEOUT 40.0 s): locate:PASS  grasp:PASS  insert:FAIL
    insert: Stage FAIL at t=20.56 with no motion toward the fixture: 'no reachable vertical insertion pose with a valid rotation chain', and all 8 psi candidates were chain bad. The planner used the predicted rod-in-hand model, not the measured +40/-10 mm one. After the FAIL the sheet shows the hand still moving (t=21–34), and by t=37.3 the rod has dropped flat onto the board near the fixture. At the end nothing is in the bore.
    note: t=0 locate: the camera fit matches the start frame (orange rod lying flat on the board at about [0.580,0.059,0.013], 49 mm long). Attempt 0 (t≈1.6–7.7): the fingers felt down to the board (contact at TCP z 0.018) and closed, but 4 mm up the rod sat 18.6 mm from the TCP along its axis, so the policy rejected it. Attempt 1 (t≈7.7–13.8) closed to 3.7 mm, but the mask fit at the end still put the rod at z 0.013, so it never lifted. Before attempt 2 the rod's centre had shifted about 1.5 mm in x, so it was nudged. Attempt 2 (t≈14.0–18.2, shifted 6 mm, yaw 8°, 1.2 mm higher) closed at 4.5 mm and lifted. Keyframe k11 (t=18.16) shows the orange rod rising with the fingers in the global view, and in the wrist view the rod hangs between the pads. In k12 (t=20.56, TCP z 0.20) the rod is clearly hanging from the gripper. It is held off-centre, close to one end: the measured rod-in-TCP centre is 15.2 mm along the axis, with measured ends at +40 mm and -10 mm, while the prediction was +24 / -26 mm. The camera saw only 17 mm of the rod. The tracker estimate (part z 0.005 → 0.003) stays at board height and is wrong; the images show the rod lifted. insert (t=20.56): the planner used the 'predicted' rod-in-hand model instead of the measured offset of about 16 mm. All 8 yaw candidates came back 'chain bad'; the best, psi -30, also had 12 hand-hit points. It reported FAIL without moving toward the fixture. In the sheet the arm keeps moving after t=20.56 with no trace entries (t=21–34). By t=37.3 and t=40 the rod has dropped and lies flat on the board near the fixture plate, not in the bore. So the episode ended with the part neither held nor inserted.
s03 (TIMEOUT 40.0 s): locate:PASS  grasp:FAIL  insert:FAIL
    grasp: The rod was never picked up. All 3 attempts collided with the blue fixture beside the rod (k03 t=4.12, k06 t=12.28, k09 t=20.80): servo_timeout at TCP z ≈0.046 with a wrench of about [-52,49,34] N, and the 'contact' height of 0.046 is the fixture, not the board at 0.009. The fingers closed to -0.9 mm on empty air about 3 cm above the rod, and the rod stayed at z 0.013 in the same place. The policy's FAIL is correct, but its stated reason (rod not lifted) hides the cause: it went ahead with a grasp even though every candidate had plan=none and more than 1000 obstacle points. The rod lies against the fixture and needs to be pushed or rolled clear, or grasped from another approach.
    insert: Never ran, because the grasp FAIL stopped the policy. The rod is still lying on the board beside the fixture at t=40.
    note: t=0 locate: the rod lies flat on the board right beside the blue fixture block, about 3.5 cm from the goal bore. The fit agrees with the start frame. Grasp scoring: every candidate had over 1000 finger-footprint obstacle points and plan=none, but the policy went ahead with the least-bad one anyway (end0 s=0.010). In all three attempts (t=1.8–4.1, 10.0–12.3, 18.8–20.8) the descent stopped at TCP z ≈0.046, about 3.7 cm above the board (0.009). The move_to calls ended in servo_timeout with a large wrench of about [-52, 49, 34] N. Keyframes k03 (t=4.12) and k09 (t=20.80) show one finger landing on and pressing against the blue fixture next to the rod. In the wrist view the jaws straddle the rod from far above, with the fixture under the left pad. The policy's 'contact' was the fixture top, not the board. The gripper closed fully (width -0.0009) on nothing each time. The rod never moved: the tracker and the fit stay at [0.354,0.042,0.013] throughout, and the sheet frames t=24–40 show it lying in the same place. Grasp FAIL at t=26.60, and insert never ran.
s04 (SUCCESS 34.8 s): locate:PASS  grasp:PASS  insert:PASS
    insert: your verifier said 'FAIL: mask fit: axis z 0.984, top z 0.0804 vs goal 0.0625 (+18.0mm), axis xy off 4.9mm'
    note: SUCCESS although your verifier failed insert: the part reached the goal, so fix the verifier, not the motion.

Artifacts: attempts/r14/<id>/ (trace.json, summary.txt, keyframes/, sheet.png, episode.mp4)

Focus stage: grasp
Rounds left: 11.

Append this round's entry to diagnosis.md first. Then either revise the focus stage's file (or the stage the
evidence points to) and end your reply with READY, or write experiments/NN_<slug>/ and end with EXPERIMENT.
