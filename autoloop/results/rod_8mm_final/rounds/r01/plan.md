Stages: locate -> grasp -> insert

## locate
Read the rod pose from locate_part() (fresh, stale==0) and the CAD model (long axis, length L, radius).
Part state: rod lying on the board, untouched; its two axis end points known in the world.
Verifier: fresh tracker estimate exists, rod axis near horizontal, centre below 4 cm, and the
segmentation-mask point cloud of the two global cameras (own depth back-projection) agrees with the
tracker centre within 15 mm in xy.

## grasp
Top-down pinch with the closing axis perpendicular to the rod, 11 mm from the end that will be on top.
The end is chosen by planning the vertical insertion for both choices (IK sweep over yaw about the
vertical rod axis -- the rod is rotationally symmetric and either way up counts) and taking the cheaper
feasible one. Then lift 6 cm.
Part state: rod held in the fingers, lifted off the board.
Verifier: fresh tracker estimate has the rod centre >= 3.5 cm higher than on the board and within 4 cm of
the TCP in xy; the global-camera mask cloud's mean height rose by > 3 cm.

## insert
Using the grasp geometry (TCP->rod transform), move to the planned pose where the rod is vertical above
the goal (high, then 3.5 cm above), descend to 4 mm above the hole bottom, open the gripper so the rod
drops in, back off along the approach axis and up.
Part state: rod standing in the fixture hole at the goal pose.
Verifier: fresh tracker estimate's lower and upper axis end points within 4 mm of the goal's.
