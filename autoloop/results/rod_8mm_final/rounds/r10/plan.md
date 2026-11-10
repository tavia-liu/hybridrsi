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
Top-down pinch 10 mm from one end A (end chosen by finger-footprint obstacle points at the grasp and over
the other end B), fingers pre-opened 30 mm, feel down until the fingertips touch the board, close with a
pose-holding Jacobian servo. Then pivot the rod up: B stays on the board while the never-rotating hand
moves on a circle around B's rim until the rod is vertical under the fingers (the fingertip pinch is a
pivot about the closing axis). Lift 6 cm: the rod hangs vertically. Retry once after re-locating.
Part state: rod hanging vertically below the top-down fingers, held near its top end.
Verifier: fresh mask fit -- axis within 15 deg of vertical, bottom > 2 cm above the board, top within
2 cm of the TCP axis.

## insert
Carry the hanging rod (orientation unchanged) above the bore, wait until two fits agree (swing settled),
translate so the measured bottom is over the bore axis 12 mm above the fixture top (< 0.4 mm), step down
3 mm at a time; if blocked above the fixture top, back out, re-align, retry (2x). Release with the bottom
3 mm above the bore bottom, open while holding the pose, lift.
Part state: rod standing in the fixture bore at the goal pose.
Verifier: fresh mask fit -- axis vertical (|a_z| > 0.98), top end within 2.5 mm of the goal top height and
the axis within 2 mm of the goal axis in xy.
