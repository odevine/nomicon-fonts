# Scanner tools

This folder holds the printed aids for a scanning session: the calibration target the session checks the scanner against, and the jigs that hold each card at a set angle on the bed. Both are US Letter PDFs, printed at actual size.

## Calibration target

`calibration_target.pdf` is an 80 mm square with rulers, slanted edges, line pairs and tone steps, and the page carries its own instructions. Cut out the square and scan it twice at the card scans' settings, once with its arrow toward the far edge of the glass (scan A) and once turned 90 degrees clockwise in the same spot (scan B). The session reads the two scans with `--cal-a` and `--cal-b`, measures scale, x and y anisotropy and edge width from them, and stops when they fall outside the gate. `python -m glyphstack.calibration cal_a.png cal_b.png` runs the check alone.

## Card jigs

`jigs.pdf` puts four jigs on one sheet, for 0, 15, 30 and 45 degrees. Two thin cuts through the middle of the sheet make four quarters, each keeping a factory corner of the paper, which goes into the scanner bed's corner. Each quarter then needs its two thick notch cuts along the lines of the card's edges. The card slides into the notch from the open side, and every jig puts the card's centre at the same place on the bed, which the sheet's notes give. Keep the printed side up on the glass.

The angle on each jig is counterclockwise as printed. The card lies face down, so in the scan the turn can come out mirrored. If the first session warns that every off-axis scan is about twice its angle away from the scan table, write the angles as 345, 330 and 315 in place of 15, 30 and 45. The README at the top of the repository describes the scan table.

`make_jigs.py` writes `jigs.pdf`. It searches for the card position and the angle for each corner that let every notch touch the full length of both card edges, and checks that no piece covers the card.

```
.venv/bin/python scanner/make_jigs.py
.venv/bin/python scanner/make_jigs.py --angles 0 15 30 45 --card 63 88
```

`--angles` takes up to four angles, one for each corner of the sheet, and `--card` sets the card size in mm. The default is a 2.5 by 3.5 inch playing card, which a Magic card fits with about half a millimetre of play.
