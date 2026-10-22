# Scanner tools

This folder holds the printed aids for a scanning session: the calibration target the session checks the scanner against, and the jigs that hold each card at a set angle on the bed. Both are US Letter PDFs, printed at actual size.

## Calibration target

`calibration_target.pdf` is an 80 mm square with rulers, slanted edges, line pairs and tone steps, and the page carries its own instructions. Cut out the square and scan it twice at the card scans' settings, once with its arrow toward the far edge of the glass (scan A) and once turned 90 degrees clockwise in the same spot (scan B). The session reads the two scans with `--cal-a` and `--cal-b`, measures scale, x and y anisotropy and edge width from them, and stops when they fall outside the gate. `python -m glyphstack.calibration cal_a.png cal_b.png` runs the check alone.

## Card jigs

`jigs.pdf` puts four jigs on one sheet, for 0, 22.5, 45 and 67.5 degrees. Two thin cuts through the middle of the sheet make four quarters, each keeping a factory corner of the paper, which goes into the scanner bed's corner. Each quarter then needs its two thick notch cuts along the lines of the card's edges. The card slides into the notch from the open side, and every jig puts the card's centre at the same place from the bed corner. Keep the printed side up on the glass.

Nothing is printed within 15 mm of the sheet's edge, so printers that cannot print to the edge still print every line. The thin cuts and some notch lines stop at that border, and those cuts carry on straight to the paper's edge.

**Each jig gives four of the sixteen angles.** Used in each of the bed's four corners, a jig turns the card by its own angle plus a quarter turn per corner, so the four jigs together cover the full circle in 22.5 degree steps. Turning the card end over end in the same notch adds a half turn without moving the jig.

The angle on each jig is counterclockwise as printed. The card lies face down, so in the scan the turn can come out mirrored. If the first session warns that every off-axis scan is about twice its angle away from the scan table, write each angle as 360 minus the angle, such as 337.5 in place of 22.5. The README at the top of the repository describes the scan table.

`make_jigs.py` writes `jigs.pdf`. It searches for the card position and the angle for each corner that give each notch the most contact with the card inside the printable area, then the fewest cuts that run on past their printed line. It checks that no piece covers the card and that nothing is drawn within the margin.

```
.venv/bin/python scanner/make_jigs.py
.venv/bin/python scanner/make_jigs.py --angles 0 22.5 45 67.5 --card 63 88 --margin 10
```

`--angles` takes up to four angles, one for each corner of the sheet, `--card` sets the card size in mm, and `--margin` the unprinted border in mm. The default card is a 2.5 by 3.5 inch playing card, which a Magic card fits with about half a millimetre of play.
