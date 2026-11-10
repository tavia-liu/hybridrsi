Stages: locate -> grasp -> insert

## locate
Own vision: back-project the part-mask pixels (interior only) of the two global cameras with depth, fit a
cylinder of the CAD radius (PCA axis + circle fit in the cross-section) -> rod axis and both end points.
Also a heightmap of everything else around the rod (finger obstacle check) and the fixture-top height
around the goal bore. The tracker is only printed for comparison.
Part state: rod lying on the board, untouched, its axis and ends known in the world.
Verifier: the fit has > 300 points, residual < 1.2 mm, visible length within 6 mm of the CAD length, axis
near horizontal, centre below 4 cm.

## grasp
Top-down pinch at (or within 10 mm of) the rod centre, closing axis perpendicular to the rod, candidates
scored by finger-footprint obstacle points and the cost of a reachable vertical-insertion plan. Fingers
pre-opened 30 mm, feel down until the fingertips touch the board, close with a pose-holding Jacobian servo;
if the close ends below 6.5 mm (rod under the pad edge) re-open to 14 mm and close again. Lift 7 cm.
Retry once after re-locating.
Part state: rod held (semi-rigidly) in the fingers, lifted off the board.
Verifier: fresh mask fit -- rod centre >= 4 cm above its board height and within 3 cm of the TCP in xy;
prints the rod centre in the TCP frame (inside the pads ~1-4 mm, at the tips ~8 mm).

## insert
Lift to 20 cm; plan the vertical rod pose over the bore (yaw swept, cost prefers the hand between base and
fixture and a small rotation); for the cheapest candidates build an IK chain that first translates the rod
centre to its high-pose position and then rotates about it in <= 11 deg steps (rejecting chains near joint
limits or ending in another IK branch); execute slowly. Pre pose: rod bottom 15 mm above the fixture top.
Closed-loop camera alignment (rotate about the tip to vertical, move the tip over the bore axis), step down
3 mm at a time (back out + re-align if blocked, 2 retries), release 7 mm inside the bore, lift.
Part state: rod standing in the fixture bore at the goal pose.
Verifier: fresh mask fit -- axis vertical (|a_z| > 0.98), top end within 2.5 mm of the goal top height and
the axis within 2 mm of the goal axis in xy.
