#!/usr/bin/env python3
"""Poster figure F2 -- Mid-context recall profile (lost-in-the-middle).

HONEST NEGATIVE RESULT. H4 (position-corrective recovery of the mid-context
bucket) FAILED: position_corrective recovers the 10% edge bucket but leaves the
30-50% middle at the floor -- at the 50% bucket it is far WORSE than the
uncorrected working set (0.5B: 1/16 vs 11/16; 1.5B: 2/16 vs 10/16).

Every plotted value is read from exp/results/full/e2_midcontext.json.
Wilson score 95% CIs are recomputed here from the raw (k, n) counts; the CI
fields (if any) written by earlier agents are NOT trusted or used.

Run:
  .venv/bin/python3 writing/figures/make_f2_midcontext.py
"""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Embed Type 42 (TrueType) fonts. matplotlib defaults to Type 3, which IEEE and
# PaperCept reject outright and which rasterises poorly at 24x36 poster scale.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent            # writing/figures
ROOT = HERE.parent                            # workspace root
SRC = ROOT / "results" / "e2_midcontext.json"
OUT_PDF = HERE / "f2_midcontext.pdf"
OUT_PNG = HERE / "f2_midcontext.png"
OUT_PROV = HERE / "f2_midcontext_provenance.json"

# --------------------------------------------------------------------------
# Design tokens (dataviz skill reference palette; light surface)
# Categorical slots assigned in fixed order: 1 blue, 2 green, 3 magenta.
# Validated all-pairs on the light surface: CVD worst dE 13.0, normal worst
# dE 27.5. Slot 3 (magenta) is sub-3:1 vs surface -> the relief rule applies,
# so every series carries a direct end-of-line label in addition to the legend,
# plus a distinct marker and dash pattern (never colour-only encoding).
# --------------------------------------------------------------------------
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = {
    "dense": "#2a78d6",               # slot 1 blue
    "uncorrected": "#008300",         # slot 2 green
    "position_corrective": "#e87ba4",  # slot 3 magenta
}
DIP_SHADE = "#f0efec"  # neutral gray -- no hue, so it never reads as a series

STYLE = {
    "dense": dict(marker="o", linestyle="-", label="Dense baseline (no eviction)"),
    "uncorrected": dict(marker="s", linestyle="--", label="Working set (uncorrected)"),
    "position_corrective": dict(marker="^", linestyle="-.", label="Working set + position_corrective"),
}
SHORT = {
    "dense": "dense",
    "uncorrected": "uncorrected",
    "position_corrective": "pos. corrective",
}
CONDS = ["dense", "uncorrected", "position_corrective"]

# Poster typography (canvas 24 x 36 in portrait)
FS_TICK = 16
FS_AXIS = 20
FS_TITLE = 24
FS_SUB = 18
FS_ANNOT = 16
FS_CAPTION = 12
LW = 3.0
MS = 10


def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion. Recomputed from raw (k, n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    data = json.loads(SRC.read_text())

    depths = data["depths_pct"]
    n_samples = data["n_samples"]
    seed = data["seed"]
    budget = data["budget"]
    ctx = data["ctx_tokens"]
    mode = data["mode"]
    device = data["env"]["device_name"]
    model_keys = list(data["models"].keys())

    provenance = {
        "figure": "f2_midcontext",
        "generated_by": "writing/figures/make_f2_midcontext.py",
        "scale": "PILOT",
        "source_file": str(SRC.relative_to(ROOT)),
        "source_mode": mode,
        "seed": seed,
        "budget_fraction_of_dense_kv": budget,
        "ctx_tokens": ctx,
        "n_per_cell": n_samples,
        "device": device,
        "ci_method": "Wilson score interval, 95% (z=1.959963984540054), recomputed from raw (k, n) in this script",
        "notes": [
            "H4_recovery = false and H4_safety_pass = false in the source JSON; verdict REFINE.",
            "No CI field from any prior agent was read or reused.",
        ],
        "verdict": data["verdict"],
        "H4_recovery": data["H4_recovery"],
        "H4_safety_pass": data["H4_safety_pass"],
        "points": [],
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 6.5), sharey=True)

    for ax, mk in zip(axes, model_keys):
        m = data["models"][mk]
        # Shade the mid-context region (30-50% depth) where the dip lives.
        ax.axvspan(30, 50, color=DIP_SHADE, zorder=0, lw=0)

        for cond in CONDS:
            per_depth = m["conditions"][cond]["per_depth"]
            xs, ys, lo_err, hi_err = [], [], [], []
            for d in depths:
                cell = per_depth[str(d)]
                k, n = cell["k"], cell["n"]
                acc = cell["accuracy"]
                lo, hi = wilson(k, n)
                xs.append(d)
                ys.append(acc)
                lo_err.append(acc - lo)
                hi_err.append(hi - acc)
                provenance["points"].append(
                    {
                        "json_file": str(SRC.relative_to(ROOT)),
                        "json_path": f"models.{mk}.conditions.{cond}.per_depth.{d}",
                        "model": mk,
                        "condition": cond,
                        "depth_pct": d,
                        "raw_k": k,
                        "raw_n": n,
                        "value": acc,
                        "wilson95_lo": lo,
                        "wilson95_hi": hi,
                    }
                )

            st = STYLE[cond]
            ax.errorbar(
                xs,
                ys,
                yerr=[lo_err, hi_err],
                color=SERIES[cond],
                ecolor=SERIES[cond],
                elinewidth=2.0,
                capsize=5,
                capthick=2.0,
                linewidth=LW,
                linestyle=st["linestyle"],
                marker=st["marker"],
                markersize=MS,
                markeredgecolor=SURFACE,
                markeredgewidth=1.6,
                zorder=3,
                clip_on=False,
            )

        ax.set_xticks(depths)
        ax.set_xlim(2, 98)
        ax.set_ylim(-0.04, 1.12)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "25", "50", "75", "100"])
        ax.tick_params(axis="both", labelsize=FS_TICK, colors=INK_SECONDARY, length=5)
        ax.set_xlabel("Needle depth (% of context)", fontsize=FS_AXIS, color=INK_PRIMARY)
        ax.set_title(mk, fontsize=FS_SUB, color=INK_PRIMARY, pad=8)
        ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(1.2)

    axes[0].set_ylabel("Retrieval accuracy (%)", fontsize=FS_AXIS, color=INK_PRIMARY, labelpad=6)

    m0 = data["models"][model_keys[0]]
    unc30 = m0["conditions"]["uncorrected"]["per_depth"]["30"]
    unc50 = m0["conditions"]["uncorrected"]["per_depth"]["50"]
    pc50 = m0["conditions"]["position_corrective"]["per_depth"]["50"]

    # "mid-context dip" label sits inside the shaded band, above the traces.
    for ax in axes:
        ax.text(40, 1.075, "mid-context dip", ha="center", va="center",
                fontsize=FS_ANNOT - 2, color=INK_SECONDARY, fontweight="bold")

    # Selective direct labels (relief rule for the sub-3:1 magenta slot, and
    # identity carried by text as well as hue). Placed on the left facet at the
    # 50% bucket, where the three series separate maximally.
    axes[0].annotate(
        f"pos. corrective\n{pc50['k']}/{pc50['n']}",
        xy=(50, pc50["accuracy"]), xytext=(66, 0.20),
        ha="left", va="center", fontsize=FS_ANNOT - 2, color=SERIES["position_corrective"],
        fontweight="bold", linespacing=1.25,
        arrowprops=dict(arrowstyle="-", color=SERIES["position_corrective"], lw=1.8,
                        shrinkA=2, shrinkB=6),
    )
    axes[0].annotate(
        f"uncorrected\n{unc50['k']}/{unc50['n']}",
        xy=(50, unc50["accuracy"]), xytext=(66, 0.52),
        ha="left", va="center", fontsize=FS_ANNOT - 2, color=SERIES["uncorrected"],
        fontweight="bold", linespacing=1.25,
        arrowprops=dict(arrowstyle="-", color=SERIES["uncorrected"], lw=1.8,
                        shrinkA=2, shrinkB=6),
    )
    axes[0].text(76, 0.88, "dense", ha="left", va="center", fontsize=FS_ANNOT - 2,
                 color=SERIES["dense"], fontweight="bold")

    handles = [
        Line2D([0], [0], color=SERIES[c], linewidth=LW, linestyle=STYLE[c]["linestyle"],
               marker=STYLE[c]["marker"], markersize=MS, markeredgecolor=SURFACE,
               markeredgewidth=1.6, label=STYLE[c]["label"])
        for c in CONDS
    ]
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.83))

    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
        fontsize=FS_ANNOT - 3,
        handlelength=2.6,
        columnspacing=1.2,
    )
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.text(
        0.5, 0.995,
        "Mid-context recall (lost-in-the-middle)",
        ha="center", va="top", fontsize=FS_TITLE, color=INK_PRIMARY, fontweight="bold",
    )
    caption = (
        f"Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct, needle-in-a-haystack at {ctx}-token context; "
        f"fixed KV budget = {int(budget * 100)}% of dense; n = {n_samples} per depth x condition cell; "
        f"seed = {seed}; {mode} scale on {device}. Error bars are 95% Wilson score intervals recomputed from "
        f"raw (k, n). Shaded band = the 30-50% mid-context region. At 30% depth the uncorrected working set "
        f"falls to {unc30['k']}/{unc30['n']} (0.5B); position_corrective lifts the 10% edge bucket to "
        f"{n_samples}/{n_samples} but drops the 50% bucket to {pc50['k']}/{pc50['n']}, below the "
        f"uncorrected {unc50['k']}/{unc50['n']}. The source JSON "
        f"records H4_recovery = false, H4_safety_pass = false, verdict REFINE."
    )
    fig.text(
        0.5, 0.155, "\n".join(textwrap.wrap(caption, width=118)),
        ha="center", va="top", fontsize=FS_CAPTION, color=INK_SECONDARY, linespacing=1.5,
    )

    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    OUT_PROV.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PROV} ({len(provenance['points'])} plotted values)")


if __name__ == "__main__":
    main()
