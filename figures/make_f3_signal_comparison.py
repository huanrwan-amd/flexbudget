#!/usr/bin/env python3
"""Poster figure F3 (HEADLINE) -- interchangeable importance signals at equal
budget on the same compiled kernel.

The controlled variable is the *importance signal*. Budget (fraction of dense
KV retained), block size, sequence length, task mix, sample count and seed are
all held fixed, and every arm runs through the same compiled eviction kernel.

Every plotted value is read from exp/results/full/e4_signal_comparison.json.
Per-signal (k, n) are pooled across the three tasks (niah, multihop,
summ_extract), 16 samples each -> n = 48 per bar. Wilson score 95% CIs are
recomputed here from the raw pooled (k, n); the per-row `wilson_ci` fields in
the JSON are read only for a consistency cross-check and are never plotted.

Only qwen2.5-0.5b is plotted. The qwen2.5-1.5b harness collapsed to near zero
*including at the dense ceiling* (3/48), which is a prompt/padding harness bug,
not a mechanism failure -- so its accuracies are not shown as valid results.
They are recorded in the provenance JSON and flagged on the figure as an
excluded harness artifact.

Run:
  .venv/bin/python3 writing/figures/make_f3_signal_comparison.py
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
SRC = ROOT / "results" / "e4_signal_comparison.json"
OUT_PDF = HERE / "f3_signal_comparison.pdf"
OUT_PNG = HERE / "f3_signal_comparison.png"
OUT_PROV = HERE / "f3_signal_comparison_provenance.json"

# --------------------------------------------------------------------------
# Design tokens (dataviz skill reference palette; light surface).
# Budget is an ORDINAL variable, so the two budget arms use a single-hue
# ordinal blue ramp (steps 300 and 600), not two categorical hues.
# Validated with the skill's validate_palette.js --ordinal path on the light
# surface: lightness monotone PASS, adjacent dL PASS, light-end contrast
# 2.50:1 PASS (>= 2:1 ordinal floor), single hue (spread 3 deg) PASS.
# Every bar additionally carries a direct value label, so identity is never
# colour-alone.
# --------------------------------------------------------------------------
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BUDGET_COLOR = {0.25: "#6da7ec", 0.5: "#184f95"}   # ordinal blue 300 / 600
DENSE_INK = "#52514e"       # neutral -- the ceiling is a reference, not a series
DENSE_BAND = "#f0efec"      # neutral midpoint gray

MODEL = "qwen2.5-0.5b"
EXCLUDED_MODEL = "qwen2.5-1.5b"
BUDGETS = [0.25, 0.5]

# Display labels: name the mechanism each signal stands in for.
SIGNAL_LABEL = {
    "recency": "keep the most recent\n(sliding window)",
    "streamingllm": "keep first + recent\n(StreamingLLM)",
    "qk_affinity": "keep the best query match\n(SnapKV-style)",
    "combined": "both combined\n(recent x attended)",
    "attn_mass": "keep the most attended-to\n(H2O-style)",
}

# Poster typography (canvas 24 x 36 in portrait)
FS_TICK = 16
FS_AXIS = 20
FS_TITLE = 24
FS_SUB = 18
FS_VALUE = 16
FS_ANNOT = 15
FS_CAPTION = 13


def wilson(k: int, n: int, z: float = 1.959963984540054):
    """Wilson score interval for a binomial proportion, from raw (k, n)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    data = json.loads(SRC.read_text())
    rows = data["rows"]

    seed = data["seed"]
    n_per_cell = data["pilot_samples"]
    block_size = data["block_size"]
    seq_len = data["seq_len"]
    mode = data["mode"]
    tasks = data["tasks"]
    device = data["env"]["device_name"]
    torch_ver = data["env"]["torch"]

    # ---- pool (k, n) across tasks for each (model, budget, signal) ---------
    pooled = defaultdict(lambda: {"k": 0, "n": 0, "rows": []})
    for i, r in enumerate(rows):
        key = (r["model"], r["budget"], r["signal"])
        cell = pooled[key]
        cell["k"] += r["k"]
        cell["n"] += r["n"]
        cell["rows"].append({"index": i, "task": r["task"], "k": r["k"], "n": r["n"]})

    # ---- consistency cross-check: JSON per-row wilson_ci vs recomputed -----
    ci_mismatch = []
    for i, r in enumerate(rows):
        lo, hi = wilson(r["k"], r["n"])
        jlo, jhi = r["wilson_ci"]
        if abs(lo - jlo) > 5e-4 or abs(hi - jhi) > 5e-4:
            ci_mismatch.append(
                {"row_index": i, "json_ci": [jlo, jhi], "recomputed": [round(lo, 4), round(hi, 4)]}
            )

    signals = [s for s in data["signals"] if s != "dense"]

    # ---- ordering: pooled across both budgets, descending -----------------
    pooled_both = {}
    for s in signals:
        k = sum(pooled[(MODEL, b, s)]["k"] for b in BUDGETS)
        n = sum(pooled[(MODEL, b, s)]["n"] for b in BUDGETS)
        pooled_both[s] = (k, n, k / n)
    order = sorted(signals, key=lambda s: pooled_both[s][2], reverse=True)

    dense = pooled[(MODEL, 1.0, "dense")]
    dense_acc = dense["k"] / dense["n"]
    dense_lo, dense_hi = wilson(dense["k"], dense["n"])

    # ---- provenance --------------------------------------------------------
    provenance = {
        "figure": "f3_signal_comparison",
        "generated_by": "writing/figures/make_f3_signal_comparison.py",
        "scale": "PILOT",
        "pilot_note": (
            f"PILOT scale: {n_per_cell} samples per (signal x task x budget) cell; "
            f"bars pool the {len(tasks)} tasks -> n = {n_per_cell * len(tasks)} per bar. "
            "CIs are wide; the ordering is indicative, not statistically separated."
        ),
        "source_file": str(SRC.relative_to(ROOT)),
        "source_mode": mode,
        "source_verdict": data["verdict"],
        "model_plotted": MODEL,
        "model_excluded": EXCLUDED_MODEL,
        "seed": seed,
        "block_size": block_size,
        "seq_len": seq_len,
        "tasks_pooled": tasks,
        "n_per_cell": n_per_cell,
        "budgets_plotted": BUDGETS,
        "device": device,
        "torch": torch_ver,
        "ci_method": (
            "Wilson score interval, 95% (z=1.959963984540054), recomputed in this "
            "script from the pooled raw (k, n). The per-row wilson_ci fields in the "
            "source JSON are cross-checked but never plotted."
        ),
        "per_row_ci_crosscheck": {
            "rows_checked": len(rows),
            "mismatches_over_5e-4": ci_mismatch,
        },
        "ordering_rule": "descending pooled accuracy over both budgets (0.25 and 0.50)",
        "bars": [],
        "reference_line": None,
        "excluded": [],
        "unsourced": [],
    }

    # ---- figure ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 6.0))  # wide+short: poster panels are height-limited

    bar_h = 0.34
    gap = 0.03
    y_centres = []
    for row_i, s in enumerate(order):
        y0 = len(order) - 1 - row_i
        y_centres.append(y0)

    # dense ceiling band + line (drawn first, behind the bars)
    ax.axvspan(dense_lo, dense_hi, color=DENSE_BAND, zorder=0, lw=0)
    ax.axvline(dense_acc, color=DENSE_INK, linestyle="--", linewidth=2.5, zorder=2)

    for row_i, s in enumerate(order):
        y0 = y_centres[row_i]
        for j, b in enumerate(BUDGETS):
            cell = pooled[(MODEL, b, s)]
            k, n = cell["k"], cell["n"]
            acc = k / n
            lo, hi = wilson(k, n)
            # budget 0.25 on top of the pair, 0.50 below
            y = y0 + (bar_h + gap) / 2 * (1 if j == 0 else -1)
            ax.barh(
                y,
                acc,
                height=bar_h,
                color=BUDGET_COLOR[b],
                edgecolor=SURFACE,
                linewidth=2.0,   # 2px surface gap between adjacent fills
                zorder=3,
            )
            ax.errorbar(
                acc,
                y,
                xerr=[[acc - lo], [hi - acc]],
                fmt="none",
                ecolor=INK_SECONDARY,
                elinewidth=2.0,
                capsize=5,
                capthick=2.0,
                zorder=4,
            )
            # Value labels live in a clean column right of every CI whisker and
            # right of the dense band, so no label ever sits on the marks.
            ax.text(
                0.98,
                y,
                f"{acc:.2f}  ({k}/{n})",
                va="center",
                ha="left",
                fontsize=FS_VALUE,
                color=INK_SECONDARY,
                zorder=5,
            )
            provenance["bars"].append(
                {
                    "json_file": str(SRC.relative_to(ROOT)),
                    "json_path": (
                        f"rows[* where model={MODEL}, signal={s}, budget={b}] "
                        f"pooled over tasks {tasks}"
                    ),
                    "source_row_indices": [r["index"] for r in cell["rows"]],
                    "per_task_rows": cell["rows"],
                    "model": MODEL,
                    "signal": s,
                    "budget": b,
                    "raw_k": k,
                    "raw_n": n,
                    "accuracy": acc,
                    "wilson_ci_recomputed": [round(lo, 4), round(hi, 4)],
                }
            )

    provenance["reference_line"] = {
        "json_file": str(SRC.relative_to(ROOT)),
        "json_path": f"rows[* where model={MODEL}, signal=dense, budget=1.0] pooled over tasks",
        "source_row_indices": [r["index"] for r in dense["rows"]],
        "per_task_rows": dense["rows"],
        "model": MODEL,
        "signal": "dense",
        "budget": 1.0,
        "raw_k": dense["k"],
        "raw_n": dense["n"],
        "accuracy": dense_acc,
        "wilson_ci_recomputed": [round(dense_lo, 4), round(dense_hi, 4)],
        "role": "dense ceiling reference line; shaded band is its 95% Wilson CI",
    }

    # excluded 1.5B cells -> recorded, not plotted
    for b in BUDGETS + [1.0]:
        for s in (["dense"] if b == 1.0 else signals):
            cell = pooled.get((EXCLUDED_MODEL, b, s))
            if not cell:
                continue
            k, n = cell["k"], cell["n"]
            lo, hi = wilson(k, n)
            provenance["excluded"].append(
                {
                    "json_file": str(SRC.relative_to(ROOT)),
                    "json_path": (
                        f"rows[* where model={EXCLUDED_MODEL}, signal={s}, budget={b}]"
                    ),
                    "source_row_indices": [r["index"] for r in cell["rows"]],
                    "model": EXCLUDED_MODEL,
                    "signal": s,
                    "budget": b,
                    "raw_k": k,
                    "raw_n": n,
                    "accuracy": k / n,
                    "wilson_ci_recomputed": [round(lo, 4), round(hi, 4)],
                    "exclusion_reason": (
                        "KNOWN HARNESS ARTIFACT: the 1.5B arm collapses to near zero even "
                        "at the dense ceiling (no eviction applied), so the harness -- not "
                        "the eviction mechanism -- is what failed. Not plotted as a valid "
                        "accuracy."
                    ),
                }
            )

    ax.set_yticks(y_centres)
    ax.set_yticklabels([SIGNAL_LABEL[s] for s in order], fontsize=FS_TICK - 1)
    ax.set_ylim(-0.62, len(order) - 0.42)
    ax.set_xlim(0.0, 1.30)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"])
    ax.tick_params(axis="both", labelsize=FS_TICK, colors=INK_SECONDARY, length=5)
    ax.set_xlabel("Task accuracy (%), pooled over 3 tasks", fontsize=FS_AXIS, color=INK_PRIMARY)
    ax.grid(axis="x", color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.2)

    # dense ceiling label
    ax.text(
        dense_acc,
        len(order) - 0.62,
        f"dense ceiling {dense_acc:.2f} ({dense['k']}/{dense['n']})",
        ha="center",
        va="bottom",
        fontsize=FS_ANNOT,
        color=DENSE_INK,
        fontweight="bold",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0),
        zorder=6,
    )

    handles = [
        Patch(facecolor=BUDGET_COLOR[0.25], edgecolor=SURFACE, label="budget = 25% of dense KV"),
        Patch(facecolor=BUDGET_COLOR[0.5], edgecolor=SURFACE, label="budget = 50% of dense KV"),
    ]
    leg = fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        fontsize=FS_ANNOT,
        ncol=2,
        handlelength=1.8,
        columnspacing=2.4,
    )
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)

    fig.suptitle(
        "Interchangeable importance signals at equal budget,\nsame compiled kernel",
        fontsize=FS_TITLE,
        color=INK_PRIMARY,
        fontweight="bold",
        y=1.155,
        linespacing=1.2,
    )
    best, worst = order[0], order[-1]
    bk, bn, bacc = pooled_both[best]
    wk, wn, wacc = pooled_both[worst]
    caption = (
        f"Qwen2.5-0.5B-Instruct, {seq_len}-token context, block size {block_size}, "
        f"tasks {{{', '.join(tasks)}}}, n = {n_per_cell} per signal x task x budget cell "
        f"(pooled n = {n_per_cell * len(tasks)} per bar), seed = {seed}; {mode} scale, "
        f"torch {torch_ver} on {device}. Error bars are 95% Wilson score intervals recomputed "
        f"from the raw pooled (k, n). Dashed line = dense ceiling ({dense['k']}/{dense['n']} = "
        f"{dense_acc:.2f}); gray band = its 95% Wilson CI. Pooled over both budgets the ordering is "
        f"{best.split(chr(10))[0]} {bacc:.2f} ({bk}/{bn}) down to {worst} {wacc:.2f} ({wk}/{wn}). "
        f"PILOT scale: the CIs overlap, so this is an indicative ordering, not a significant "
        f"separation -- the claim is that the signal is a swappable input to one fixed kernel. "
        f"Qwen2.5-1.5B cells are EXCLUDED: that harness collapsed to "
        f"{pooled[(EXCLUDED_MODEL, 1.0, 'dense')]['k']}/"
        f"{pooled[(EXCLUDED_MODEL, 1.0, 'dense')]['n']} even at the dense ceiling, a prompt/padding "
        f"harness bug rather than a mechanism failure; raw counts are kept in the provenance JSON."
    )
    cap = fig.text(0.5, -0.055, caption, ha="center", va="top", fontsize=FS_CAPTION,
                   color=INK_SECONDARY, wrap=True, linespacing=1.4)
    # matplotlib's wrap uses the figure width; force an explicit width in points.
    # Must NOT exceed the figure width (10 in = 720 pt). A wider value (this was 1050)
    # makes bbox_inches="tight" expand the canvas to ~14.8 in to fit the caption, which
    # leaves dead margins and shrinks the plot to ~50% when placed on the poster, taking
    # the tick labels down to ~5 pt. Wrapping inside the figure keeps the plot full-width.
    cap._get_wrap_line_width = lambda: 870

    fig.tight_layout()
    fig.subplots_adjust(top=0.935)

    fig.savefig(OUT_PDF, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    OUT_PROV.write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PROV} ({len(provenance['bars'])} bars + 1 reference line)")
    print(f"per-row CI cross-check mismatches: {len(ci_mismatch)}")
    print("ordering (pooled over both budgets):")
    for s in order:
        k, n, a = pooled_both[s]
        print(f"  {s:14s} {a:.4f}  k={k} n={n}")


if __name__ == "__main__":
    main()
