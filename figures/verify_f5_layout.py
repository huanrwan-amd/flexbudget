"""Geometric layout audit for f5_mechanism.

Why this exists
---------------
f5 is hand-placed in a 0-100 data-coordinate space, so nothing stops a string
from running through a box edge or off the canvas. Three separate spills
reached a rendered poster and were each caught by a human reading the image,
not by any check:

  * "most recent    most attended    best match" measured 49.0 x-units against
    47.5 of inner box width and crossed the score-vocabulary box's right edge.
  * "Measured on GPU at this budget: 47.2% dropped" ran to x=103.3 on a canvas
    that ends at 100, and clipped the callout box's right edge on the way.
  * The section headers rendered as continuations of the line above them.

Eyeballing a 925x524 pt figure does not reliably catch a 1.5-unit overlap.
This does, by measuring every text artist's rendered extent.

Run:
    python3 figures/verify_f5_layout.py
Exit 0 = clean. Exit 1 = at least one spill.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.transforms as mt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
TOL = 0.15          # x-units of slack; ~1.5 pt at this figure scale
CANVAS = (0.0, 100.0)


def main() -> int:
    sys.path.insert(0, str(HERE))
    ns = runpy.run_path(str(HERE / "make_f5_mechanism.py"), run_name="__layout_audit__")
    fig, ax = ns["fig"], ns["ax"]

    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    def bbox(artist):
        b = artist.get_window_extent(renderer=renderer)
        p0, p1 = inv.transform((b.x0, b.y0)), inv.transform((b.x1, b.y1))
        return mt.Bbox([[min(p0[0], p1[0]), min(p0[1], p1[1])],
                        [max(p0[0], p1[0]), max(p0[1], p1[1])]])

    frames = [bbox(p) for p in ax.patches if isinstance(p, FancyBboxPatch)]
    fails = []

    # 1. no text may cross a box frame it is not fully inside
    for tx in ax.texts:
        tb, s = bbox(tx), tx.get_text().replace("\n", " / ")
        for fb in frames:
            if not tb.overlaps(fb):
                continue
            if (tb.x0 >= fb.x0 - TOL and tb.x1 <= fb.x1 + TOL
                    and tb.y0 >= fb.y0 - TOL and tb.y1 <= fb.y1 + TOL):
                continue
            over = []
            if fb.x0 - tb.x0 > TOL:
                over.append(f"left {fb.x0 - tb.x0:.2f}")
            if tb.x1 - fb.x1 > TOL:
                over.append(f"right {tb.x1 - fb.x1:.2f}")
            if fb.y0 - tb.y0 > TOL:
                over.append(f"below {fb.y0 - tb.y0:.2f}")
            if tb.y1 - fb.y1 > TOL:
                over.append(f"above {tb.y1 - fb.y1:.2f}")
            fails.append(f"SPILL through box edge: {s[:52]!r} -> {', '.join(over)}")

    # 2. no text may leave the 0-100 canvas
    lo, hi = CANVAS
    for tx in ax.texts:
        b, s = bbox(tx), tx.get_text().replace("\n", " / ")
        if b.x1 > hi + 0.2 or b.x0 < lo - 0.2 or b.y1 > hi + 0.2 or b.y0 < lo - 0.2:
            fails.append(f"OFF-CANVAS: {s[:52]!r} "
                         f"x[{b.x0:.1f},{b.x1:.1f}] y[{b.y0:.1f},{b.y1:.1f}]")

    print("=" * 70)
    print(f"f5 layout audit: {len(ax.texts)} text artists, {len(frames)} box frames")
    for f in fails:
        print("  FAIL  " + f)
    if not fails:
        print("  PASS  no text crosses a box edge or leaves the canvas")
    print("=" * 70)
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
