#!/usr/bin/env python3
"""F5 -- FlexBudget mechanism diagram (poster asset, schematic).

Renders writing/figures/f5_mechanism.{pdf,png}.

The admission grid drawn here is NOT hand-drawn: it is produced by calling the
real ``exp/code/working_set_mask.py`` (ScoreVocabulary.attn_mass ->
build_block_importance) so the schematic cannot drift from the implementation.
Geometry (block_size=128, seq_len=1024 -> 8 blocks, budget_fraction=0.25,
n_sink_blocks=1, local_window_blocks=1) matches the E0 parity gate.

Run:
    .venv/bin/python3 writing/figures/make_f5_mechanism.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent       # repository root
sys.path.insert(0, str(ROOT / "flexbudget"))

import torch  # noqa: E402  (only used to build the admitted map via the real code)

import working_set_mask as wsm  # noqa: E402

# ---------------------------------------------------------------------------
# 1. Build the admission map with the REAL implementation
# ---------------------------------------------------------------------------
BLOCK_SIZE = wsm.DEFAULT_BLOCK_SIZE          # 128
SEQ_LEN = 1024                               # E0 parity-gate seq_len
N_BLK = wsm.n_blocks(SEQ_LEN, BLOCK_SIZE)    # 8
BUDGET = 0.25                                # E0 "budget25" setting
N_SINK = 1                                   # build_block_importance default
N_LOCAL = 1                                  # build_block_importance default
E0_MEASURED_SKIP = 0.4722                    # e0_parity_gate.json budget25_skip_rate

# Illustrative per-KV-block attention mass (schematic input, NOT measured):
# two "hot" mid-context blocks so score-selected admissions are visible.
ILLUSTRATIVE_ATTN_MASS = torch.tensor([0.05, 0.12, 0.90, 0.20, 0.08, 0.75, 0.30, 0.10])

stats = wsm.BlockStats(
    n_q_blocks=N_BLK, n_kv_blocks=N_BLK, block_size=BLOCK_SIZE,
    attn_mass=ILLUSTRATIVE_ATTN_MASS,
)
scores = wsm.SCORERS["attn_mass"](stats)
admitted = wsm.build_block_importance(
    scores, budget_fraction=BUDGET, n_sink_blocks=N_SINK, local_window_blocks=N_LOCAL
).numpy()

qi = torch.arange(N_BLK).unsqueeze(1)
kj = torch.arange(N_BLK).unsqueeze(0)
causal = (kj <= qi).numpy()
sink = ((kj < N_SINK) & (kj <= qi)).numpy()
local = ((kj > (qi - N_LOCAL)) & (kj <= qi)).numpy()
n_causal_cells = int(causal.sum())
n_admitted_cells = int((admitted & causal).sum())
SKIP_ILLUS = 1.0 - n_admitted_cells / n_causal_cells

Q_ROW = 6                                    # highlighted query row
Q_NCAUSAL = Q_ROW + 1
Q_B = int(torch.ceil(torch.tensor(BUDGET * Q_NCAUSAL)).item())
Q_TOPB = [k for k in range(N_BLK)
          if admitted[Q_ROW, k] and not sink[Q_ROW, k] and not local[Q_ROW, k]]

# ---------------------------------------------------------------------------
# 2. Palette (Okabe-Ito; CMYK-safe, CVD-safe). Never colour-only: every fill
#    also carries a glyph (S / L / B) or a hatch, and every stage a word badge.
# ---------------------------------------------------------------------------
C_ANCHOR = "#E69F00"   # orange -- always-on anchors (sinks + local window)
C_TOPB = "#0072B2"     # blue   -- admitted by score under budget
C_SKIP = "#E4E4E4"     # grey   -- causal but evicted
C_FORBID = "#FFFFFF"   # white  -- non-causal, never admitted
C_SWAP = "#009E73"     # green  -- the one swappable stage
INK = "#111111"
INK2 = "#4A4A4A"
INK3 = "#767676"
EDGE = "#9A9A9A"
BOXFILL = "#F4F4F4"

FS_TITLE = 26
FS_SEC = 22
FS = 18

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig = plt.figure(figsize=(11, 7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
ASPECT = 11 / 7  # x-unit : y-unit page-size ratio (for square cells / round corners)


def box(x, y, w, h, fc=BOXFILL, ec=EDGE, lw=1.6, r=1.0, z=1):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
        mutation_aspect=1 / ASPECT,
    ))


def arrow(x0, y0, x1, y1, lw=2.4, color=INK2, z=3):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=20,
        linewidth=lw, color=color, zorder=z, shrinkA=0, shrinkB=0,
    ))


# ---------------------------------------------------------------------------
# Column grid. Every box, header underline and text block snaps to these, so
# the two halves line up and there is a real gutter between them. Previously
# the left callout ended at x=47.5 and the right boxes STARTED at 47.5 -- they
# touched, which is why the panel had no visible boundary.
# ---------------------------------------------------------------------------
LCOL_X, LCOL_W = 2.5, 40.0          # left column   2.5 .. 42.5
GUTTER         = 4.0                #               42.5 .. 46.5
RCOL_X, RCOL_W = 46.5, 50.5         # right column 46.5 .. 97.0
PAD            = 1.6                # text inset inside any box

# Bottom callouts share one top and one bottom so the halves align.
CALLOUT_Y, CALLOUT_H = 20.2, 16.6   # 20.2 .. 36.8

# ---------------------------------------------------------------------------
# 3. Title
# ---------------------------------------------------------------------------
ax.text(LCOL_X, 96.8, "A compiled KV working set",
        fontsize=FS_TITLE, fontweight="bold", color=INK, va="center")

# Deliberately NOT the legend fills (#E69F00 anchors, #0072B2 top-B): a header
# tinted like a cell would read as a key to that cell. Both are Okabe-Ito
# derived, CVD-safe and CMYK-printable.
C_SEC_L = "#00795C"   # teal-green -- "What each query keeps"
C_SEC_R = "#882E72"   # deep purple -- "How to read the grid"



def section(x, y, label, width, color=INK):
    """Section header with its own underline, so it cannot read as a
    continuation of the line above it."""
    ax.text(x, y, label, fontsize=FS_SEC, fontweight="bold", color=color,
            va="center")
    ax.add_patch(Rectangle((x, y - 2.6), width, 0.35, facecolor=color,
                           edgecolor="none", zorder=4))

# ---------------------------------------------------------------------------
# 4. Admission grid (left)
# ---------------------------------------------------------------------------
CW = 3.45                 # x-units  (= 0.38 in)
CH = CW * ASPECT          # y-units  (square on the page)
GX0, GY_TOP = 9.0, 80.5
GY_BOT = GY_TOP - N_BLK * CH
GX1 = GX0 + N_BLK * CW

section(LCOL_X, 91.6, "What each query keeps", LCOL_W, color=C_SEC_L)

for q in range(N_BLK):
    for k in range(N_BLK):
        x = GX0 + k * CW
        y = GY_TOP - (q + 1) * CH
        cell_ec = "#FFFFFF"
        if not causal[q, k]:
            fc, hatch, glyph, gc = C_FORBID, "///", "", INK3
            cell_ec = "#C9C9C9"
        elif sink[q, k]:
            fc, hatch, glyph, gc = C_ANCHOR, "", "S", "#3A2600"
        elif local[q, k]:
            fc, hatch, glyph, gc = C_ANCHOR, "", "L", "#3A2600"
        elif admitted[q, k]:
            fc, hatch, glyph, gc = C_TOPB, "", "B", "#FFFFFF"
        else:
            fc, hatch, glyph, gc = C_SKIP, "", "", INK3
        ax.add_patch(Rectangle((x, y), CW, CH, facecolor=fc,
                               edgecolor=cell_ec, linewidth=1.4, hatch=hatch,
                               zorder=2))
        if hatch:
            ax.add_patch(Rectangle((x, y), CW, CH, facecolor="none",
                                   edgecolor="#D2D2D2", linewidth=0.9, zorder=3))
        if glyph:
            ax.text(x + CW / 2, y + CH / 2, glyph, fontsize=FS, fontweight="bold",
                    color=gc, ha="center", va="center", zorder=4)

for k in range(N_BLK):
    ax.text(GX0 + (k + 0.5) * CW, GY_TOP + 1.2, str(k), fontsize=FS,
            color=INK2, ha="center", va="bottom")
for q in range(N_BLK):
    ax.text(GX0 - 1.2, GY_TOP - (q + 0.5) * CH, str(q), fontsize=FS,
            color=INK2, ha="right", va="center")
ax.text(GX0, GY_TOP + 5.9, "KV block (128 tokens)", fontsize=FS,
        color=INK, ha="left", va="center")
ax.text(GX0 - 6.0, (GY_TOP + GY_BOT) / 2, "query block", fontsize=FS,
        color=INK, ha="center", va="center", rotation=90)

# highlighted query row
ry = GY_TOP - (Q_ROW + 1) * CH
ax.add_patch(Rectangle((GX0, ry), N_BLK * CW, CH, facecolor="none",
                       edgecolor=INK, linewidth=3.0, zorder=6))

# per-query budget callout under the grid
CB_Y, CB_H = CALLOUT_Y, CALLOUT_H
box(LCOL_X, CB_Y, LCOL_W, CB_H, fc="#FFFFFF", ec=INK, lw=2.6)
# Plain-language walkthrough of the boxed row. The earlier version read
# "B = ceil(0.25 x 7) = 2 blocks / admits S0 + L6 + B2 + B5", which is exact but
# asks a poster viewer to decode four symbols before it means anything.
lines = [
    (f"Reading row {Q_ROW} (boxed)", True, INK),
    (f"Sees {Q_NCAUSAL} blocks. 25% allows {Q_B}.", False, INK2),
    (f"Keeps {Q_B + N_SINK + N_LOCAL}: first, newest, +{Q_B}.", False, INK2),
]
_cbc = CB_Y + CB_H / 2.0          # centred, so it mirrors the right-hand box
for _i, (txt, bold, col) in enumerate(lines):
    ax.text(LCOL_X + PAD, _cbc + 4.2 - _i * 4.2, txt, fontsize=FS, color=col,
            fontweight="bold" if bold else "normal", va="center")

# ---------------------------------------------------------------------------
# 5. Legend + score vocabulary + key message (right)
# ---------------------------------------------------------------------------
RX = RCOL_X + PAD
section(RCOL_X, 91.6, "How to read the grid", RCOL_W, color=C_SEC_R)

SW = 2.9
SH = SW * ASPECT
legend_rows = [
    (C_ANCHOR, "", "S", "first block -- always kept"),
    (C_ANCHOR, "", "L", "most recent block -- always kept"),
    (C_TOPB, "", "B", "kept because it scored highest"),
    (C_SKIP, "", "", "dropped to stay inside the budget"),
    (C_FORBID, "///", "", "in the future -- cannot be used"),
]
ly = 80.0
for fc, hatch, glyph, label in legend_rows:
    ax.add_patch(Rectangle((RX, ly - SH / 2), SW, SH, facecolor=fc,
                           edgecolor="#C4C4C4", linewidth=1.1, hatch=hatch, zorder=2))
    if glyph:
        ax.text(RX + SW / 2, ly, glyph, fontsize=FS, fontweight="bold",
                color="#3A2600" if fc == C_ANCHOR else "#FFFFFF",
                ha="center", va="center", zorder=3)
    ax.text(RX + SW + 2.0, ly, label, fontsize=FS, color=INK2, va="center")
    ly -= 5.0

# score_mod vocabulary. Three short lines, not two long ones: at FS=18 the
# four-gap line "most recent    most attended    best match" measured 49.0
# x-units against 47.5 of inner box width and spilled through the right edge.
SV_Y, SV_H = 38.9, 18.0
box(RCOL_X, SV_Y, RCOL_W, SV_H, fc="#EAF7F2", ec=C_SWAP, lw=2.6)
ax.text(RX, SV_Y + SV_H - 3.4, "The score is the only swappable part",
        fontsize=FS, fontweight="bold", color="#00543E", va="center")
for _i, _line in enumerate(["most recent    most attended",
                            "best match    both combined",
                            "position-corrected"]):
    ax.text(RX, SV_Y + SV_H - 7.6 - _i * 4.0, _line,
            fontsize=FS, color=INK, va="center")

# The "Measured on GPU at this budget: 47.2% dropped" note lived here. It ran
# 3.3 x-units past the canvas and clipped the callout box's right edge, and the
# poster caption under this figure already states the same number. Removed.

# key message
KM_Y, KM_H = CALLOUT_Y, CALLOUT_H
box(RCOL_X, KM_Y, RCOL_W, KM_H, fc="#FFFFFF", ec=INK, lw=2.6)
_kmc = KM_Y + KM_H / 2.0          # centre the 3 lines in the shared height
for _i, (_txt, _b, _c) in enumerate([
        ("Budget and kernel stay FIXED.", True, INK),
        ("Only the score swaps, so it is", False, INK2),
        ("a controlled variable.", False, INK2)]):
    ax.text(RX, _kmc + 4.2 - _i * 4.2, _txt, fontsize=FS,
            fontweight="bold" if _b else "normal", color=_c, va="center")

# ---------------------------------------------------------------------------
# 6. Pipeline strip (bottom, left -> right)
# ---------------------------------------------------------------------------
PY, PH = 0.8, 12.0
PX0, PX1 = LCOL_X, RCOL_X + RCOL_W
GAP = 2.4
stages = [
    ("KV cache", "split into blocks\nof 128 tokens", None),
    ("the score", "ranks the blocks\nscore_mod", "SWAPS"),
    ("the budget", "keeps the top few\nplus anchors", "FIXED"),
    ("the kernel", "one compiled pass\nflex_attention", "FIXED"),
]
NB = len(stages)
BW = (PX1 - PX0 - GAP * (NB - 1)) / NB

for i, (head, body, badge) in enumerate(stages):
    x = PX0 + i * (BW + GAP)
    swap = badge == "SWAPS"
    box(x, PY, BW, PH,
        fc="#EAF7F2" if swap else BOXFILL,
        ec=C_SWAP if swap else EDGE,
        lw=2.6 if swap else 1.8)
    ax.text(x + BW / 2, PY + PH - 3.0, head, fontsize=FS, fontweight="bold",
            color="#00543E" if swap else INK, ha="center", va="center")
    ax.text(x + BW / 2, PY + 3.6, body, fontsize=FS, color=INK2,
            ha="center", va="center", linespacing=1.15)
    if badge:
        bw = 11.5
        box(x + BW / 2 - bw / 2, PY + PH + 0.7, bw, 4.4,
            fc=C_SWAP if swap else "#DCDCDC",
            ec=C_SWAP if swap else "#B4B4B4", lw=1.2, r=0.8)
        ax.text(x + BW / 2, PY + PH + 2.9, badge, fontsize=FS, fontweight="bold",
                color="#FFFFFF" if swap else INK2, ha="center", va="center")
    if i < NB - 1:
        arrow(x + BW + 0.3, PY + PH / 2, x + BW + GAP - 0.3, PY + PH / 2)

fig.savefig(HERE / "f5_mechanism.pdf", bbox_inches="tight")
fig.savefig(HERE / "f5_mechanism.png", dpi=150, bbox_inches="tight")
print(f"causal={n_causal_cells} admitted={n_admitted_cells} skip={SKIP_ILLUS:.4f} "
      f"| row{Q_ROW}: B={Q_B} topB={Q_TOPB} | E0 ref skip={E0_MEASURED_SKIP}")
print("wrote f5_mechanism.pdf / .png")
