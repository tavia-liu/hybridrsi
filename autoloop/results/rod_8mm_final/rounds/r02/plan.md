Stages: locate -> grasp -> insert

## locate
Own vision: back-project the part-mask pixels (interior only) of the two global cameras with depth, fit a
cylinder of the CAD radius (PCA axis + circle fit in the cross-section) -> rod axis and both end points.
Also builds a heightmap of everything else around the rod (obstacle check for the fingers). The tracker is
only printed for comparison.
Part state: rod lying on the board, untouched, its axis and ends known in the world.
Verifier: the fit has > 300 points, residual < 1.2 mm, visible length within 6 mm of the CAD length, axis
near horizontal, centre below 4 cm.

## grasp
Top-down pinch, closing axis perpendicular to the rod, TCP 11.5 mm above the board (fingertips reach 9 mm
below the TCP), fingers pre-opened to 30 mm. Candidates: either end on top, 11/16/21 mm from that end;
each scored by the finger footprints' obstacle points in the heightmap and by the IK plan of the vertical
insertion (yaw swept around the vertical rod). Close, lift 7 cm.
Part state: rod held in the fingers, off the board.
Verifier: fresh mask cylinder fit -- rod centre >= 4 cm above its board height and within 4 cm of the TCP
in xy. Also gives the rod's measured end points in the TCP frame.

## insert
Plan the vertical pose (measured rod-in-hand if consistent with the prediction, else predicted), go high,
then rod bottom 3.5 cm above the hole bottom; up to 3 camera-fit corrections of the rod axis xy (and
bottom height) against the goal axis; descend to 4 mm above the hole bottom; open (rod drops in); back
off along the approach axis and up.
Part state: rod standing in the fixture at the goal pose.
Verifier: fresh mask fit -- axis vertical (|a_z| > 0.98), top end within 2.5 mm of the goal top height
and the axis within 2 mm of the goal axis in xy.
