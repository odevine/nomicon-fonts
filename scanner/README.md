# Scanner tools

This folder holds the printed aids for a scanning session: the calibration target the session checks the scanner against, and the jigs that hold each card at a set angle on the bed. Both are US Letter PDFs, printed at actual size.

## Calibration target

`calibration_target.pdf` is an 80 mm square with rulers, slanted edges, line pairs and tone steps, and the page carries its own instructions. Cut out the square and scan it twice at the card scans' settings, once with its arrow toward the far edge of the glass (scan A) and once turned 90 degrees clockwise in the same spot (scan B). The session reads the two scans with `--cal-a` and `--cal-b`, measures scale, x and y anisotropy and edge width from them, and stops when they fall outside the gate. `python -m glyphstack.calibration cal_a.png cal_b.png` runs the check alone.

## Card jigs

`jigs.pdf` holds the 0 degree jig in one quarter of a sheet. Two thin cuts through the middle of the sheet free that quarter, which keeps a factory corner of the paper that goes into the scanner bed's corner. The quarter then needs its two thick notch cuts along the lines of the card's edges. The card slides into the notch from the open side, and every jig puts the card's centre at the same place from the bed corner. Keep the printed side up on the glass.

Nothing is printed within 15 mm of the sheet's edge, so printers that cannot print to the edge still print every line. The thin cuts and some notch lines stop at that border, and those cuts carry on straight to the paper's edge.

**Each card is scanned twice, at 0 and 90 degrees.** Used in a corner of the bed and then in the next corner along, the jig turns the card by a quarter turn, which gives the 90 degree scan from the same notch. More angles add no detail at this scanner's blur, since every printed glyph already lands at its own subpixel offset, and they multiply the data to process.

The angle on each jig is counterclockwise as printed. The card lies face down, so in the scan the turn can come out mirrored. If the first session warns that every turned scan is about twice its angle away from the scan table, write each angle as 360 minus the angle, such as 270 in place of 90. The README at the top of the repository describes the scan table.

`make_jigs.py` writes `jigs.pdf`. It can lay out up to four jigs at other angles, one for each corner of the sheet. It searches for the card position and the corner for each angle that give each notch the most contact with the card inside the printable area, then the fewest cuts that run on past their printed line. It checks that no piece covers the card and that nothing is drawn within the margin.

```
.venv/bin/python scanner/make_jigs.py
.venv/bin/python scanner/make_jigs.py --angles 0 --card 63 88 --margin 10
```

`--angles` takes up to four angles, one for each corner of the sheet, `--card` sets the card size in mm, and `--margin` the unprinted border in mm. The default card is a 2.5 by 3.5 inch playing card, which a Magic card fits with about half a millimetre of play.
