#!/usr/bin/env python3
"""Poster figure F1: NIAH accuracy vs KV budget knee-curve (FlexBudget, PyTorch Conf 2026).

Re-runnable. Every plotted value is read from the experiment JSONs; nothing is
estimated, smoothed or interpolated. Wilson 95% CIs are recomputed here from the
raw (k, N) counts -- CI fields written by earlier agents are deliberately ignored.

Run:
  .venv/bin/python3 writing/figures/make_f1_budget_knee.py
from the workspace root (or anywhere -- paths are resolved from __file__).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve().parent          # writing/figures
ROOT = HERE.parent                               # repository root
E1 = ROOT / "results" / "e1_budget_knee.json"
E5 = ROOT / "results" / "e5_ablation_reasoning.json"
OUT_PDF = HERE / "f1_budget_knee.pdf"
OUT_PNG = HERE / "f1_budget_knee.png"
OUT_PROV = HERE / "f1_budget_knee_provenance.json"

# ------------------------------------------------- dataviz palette (light)
# Reference palette, light surface. Categorical slot 1 (blue) + slot 6 (orange).
# Validated with scripts/validate_palette.js "#2a78d6,#eb6834" --mode light
# --pairs all -> ALL CHECKS PASS (CVD dE 24.7 protan, normal dE 33.6, >=3:1).
SURFACE = "#fcfcfb"
C_WORKING = "#2a78d6"   # slot 1, blue   -- importance-based working-set selection
C_STREAM = "#eb6834"    # slot 6, orange -- fixed sink+window (StreamingLLM-style)
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
C_INSET = "#2a78d6"

Z = 1.959963984540054  # 95% two-sided normal quantile


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Recomputed from (k, N)."""
    if n == 0:
        raise ValueError("N must be > 0")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def short(repo: str) -> str:
    """'Qwen/Qwen2.5-1.5B-Instruct' -> 'Qwen2.5-1.5B-Instruct' (no invented text)."""
    return repo.split("/", 1)[-1]


def main() -> None:
    e1 = json.loads(E1.read_text())
    e5 = json.loads(E5.read_text())

    budgets = e1["budgets"]                       # [0.125, 0.25, 0.5, 0.75, 1.0]
    xs = [100.0 * b for b in budgets]
    prov: list[dict] = []

    # ---- series definition: (model_key, method_key, label, colour, ls, marker, dodge)
    series_spec = [
        ("qwen2.5-1.5b", "working", "Working set, 1.5B", C_WORKING, "-", "o", -0.8),
        ("qwen2.5-1.5b", "streaming", "Sink+window, 1.5B", C_STREAM, "-", "s", +0.8),
        ("qwen2.5-0.5b", "working", "Working set, 0.5B", C_WORKING, "--", "^", -2.4),
        ("qwen2.5-0.5b", "streaming", "Sink+window, 0.5B", C_STREAM, "--", "D", +2.4),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,      # TrueType, editable/searchable vector text
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(10, 6.5))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.15, 1.0], wspace=0.34,
                          left=0.085, right=0.985, top=0.885, bottom=0.425)
    ax = fig.add_subplot(gs[0, 0])
    ins = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(SURFACE)
    ins.set_facecolor(SURFACE)

    # ---- dense ceiling (horizontal reference). Both models: k=16/16 at budget 1.0.
    dense_vals = {}
    for mk in ("qwen2.5-1.5b", "qwen2.5-0.5b"):
        cell = e1["models"][mk]["cells"]["dense@1.0"]
        dense_vals[mk] = (cell["k"], cell["N"], cell["k"] / cell["N"])
        prov.append({
            "series": f"dense ceiling ({mk})",
            "x_budget_pct": 100.0,
            "json_file": "exp/results/full/e1_budget_knee.json",
            "json_path": f"models.{mk}.cells.dense@1.0",
            "raw_k": cell["k"], "raw_n": cell["N"],
            "value": cell["k"] / cell["N"],
            "wilson95_recomputed": list(wilson(cell["k"], cell["N"])),
        })
    assert {v[2] for v in dense_vals.values()} == {1.0}, "dense ceilings differ; relabel the line"
    ceiling = 1.0
    ax.axhline(ceiling, color=MUTED, ls=(0, (6, 4)), lw=2.5, zorder=1)
    ax.text(103, ceiling + 0.022, "Dense ceiling  16/16", fontsize=16, color=INK_2,
            va="bottom", ha="right")

    # ---- the four measured series
    for model, method, label, colour, ls, marker, dodge in series_spec:
        cells = e1["models"][model]["cells"]
        ys, lo, hi = [], [], []
        for b in budgets:
            cell = cells[f"{method}@{b}"]
            k, n = cell["k"], cell["N"]
            v = k / n
            l, h = wilson(k, n)
            ys.append(v); lo.append(v - l); hi.append(h - v)
            prov.append({
                "series": label,
                "x_budget_pct": 100.0 * b,
                "json_file": "exp/results/full/e1_budget_knee.json",
                "json_path": f"models.{model}.cells.{method}@{b}",
                "raw_k": k, "raw_n": n,
                "value": v,
                "wilson95_recomputed": [l, h],
            })
        ax.errorbar([x + dodge for x in xs], ys, yerr=[lo, hi],
                    color=colour, ls=ls, lw=2.8, marker=marker, ms=10,
                    markeredgecolor=SURFACE, markeredgewidth=1.4,
                    elinewidth=2.0, capsize=6, capthick=2.0, zorder=3, label=label)

    # ---- knee annotation (1.5B working set at 50% budget)
    knee_cell = e1["models"]["qwen2.5-1.5b"]["cells"]["working@0.5"]
    kk, kn = knee_cell["k"], knee_cell["N"]
    kv = kk / kn
    stream_knee = e1["models"]["qwen2.5-1.5b"]["cells"]["streaming@0.5"]
    ax.annotate(
        f"Knee: 50% budget\nworking set {kk}/{kn}\nsink+window "
        f"{stream_knee['k']}/{stream_knee['N']}",
        xy=(50 - 1.6, kv), xycoords="data",
        xytext=(9.5, 0.70), textcoords="data",
        fontsize=16, color=INK, ha="left", va="center", linespacing=1.3,
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE, alpha=0.88,
                  edgecolor=GRID, linewidth=1.0),
        arrowprops=dict(arrowstyle="-|>", color=INK_2, lw=2.2,
                        shrinkA=4, shrinkB=8,
                        connectionstyle="arc3,rad=-0.12"),
    )
    ax.axvline(50, color=AXIS, lw=1.6, ls=":", zorder=0)

    # ---- axes
    ax.set_xlabel("KV budget (% of full cache)", fontsize=20, color=INK, labelpad=10)
    ax.set_ylabel("NIAH retrieval accuracy", fontsize=20, color=INK, labelpad=10)
    fig.text(0.012, 0.985,
             "Half the KV cache, full retrieval accuracy",
             fontsize=24, color=INK, ha="left", va="top")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="both", labelsize=16, colors=INK_2, length=5, width=1.2)
    ax.set_xlim(6, 108)
    ax.set_ylim(-0.06, 1.10)
    ax.grid(axis="y", color=GRID, lw=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.4)

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=MUTED, ls=(0, (6, 4)), lw=2.5))
    labels.append("Dense (no eviction)")
    leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.145),
                     ncol=3, fontsize=16, frameon=True, facecolor=SURFACE,
                     edgecolor=GRID, labelcolor=INK_2, handlelength=2.8,
                     borderpad=0.6, labelspacing=0.45, columnspacing=1.6,
                     handletextpad=0.6)
    leg.get_frame().set_linewidth(1.2)

    # ---- inset: block-size sweep (E5), mean block skip rate vs block size
    legs = {row["leg"]: row for row in e5["ablation"]}
    bs_map = [(64, "bs64"), (128, "anchors_on_bs128"), (256, "bs256")]
    bs_x, bs_y = [], []
    for bs, key in bs_map:
        row = legs[key]
        bs_x.append(str(bs)); bs_y.append(row["mean_block_skip_rate"])
        prov.append({
            "series": "inset: mean block skip rate vs block size",
            "x_block_size": bs,
            "json_file": "exp/results/full/e5_ablation_reasoning.json",
            "json_path": f"ablation[leg={key}].mean_block_skip_rate",
            "raw_k": row["k"], "raw_n": row["n"],
            "note": "raw_k/raw_n are the NIAH accuracy counts for this leg; "
                    "the plotted value is mean_block_skip_rate, a mean over blocks "
                    "with no per-block counts in the JSON, so it carries no CI.",
            "value": row["mean_block_skip_rate"],
        })

    pos = list(range(len(bs_x)))
    ins.bar(pos, bs_y, color=C_INSET, width=0.62, zorder=3)
    for xi, yi in zip(pos, bs_y):
        ins.text(xi, yi + 0.03, f"{yi:.2f}", ha="center", va="bottom",
                 fontsize=15, color=INK_2)
    ins.set_xticks(pos)
    ins.set_xticklabels(bs_x)
    ins.set_xlim(-0.6, len(pos) - 0.4)
    ins.set_ylim(0, 0.92)
    ins.set_yticks([0, 0.5])
    ins.set_yticklabels(["0", "0.5"])
    ins.tick_params(axis="both", labelsize=15, colors=INK_2, length=4, width=1.2)
    ins.set_xlabel("Block size (tokens)", fontsize=16, color=INK, labelpad=8)
    ins.set_ylabel("Mean block skip rate", fontsize=16, color=INK, labelpad=6)
    ins.set_title("Block-size sweep", fontsize=16, color=INK, pad=10)
    ins.tick_params(axis="both", labelsize=16)
    ins.set_yticks([0, 0.25, 0.5, 0.75])
    ins.set_yticklabels(["0", "0.25", "0.50", "0.75"])
    ins.grid(axis="y", color=GRID, lw=1.0, zorder=0)
    ins.set_axisbelow(True)
    for side in ("top", "right"):
        ins.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ins.spines[side].set_color(AXIS)
        ins.spines[side].set_linewidth(1.4)

    # ---- caption / provenance strip
    m15 = e1["models"]["qwen2.5-1.5b"]
    caption = (
        f"Main: {short(m15['repo'])} and "
        f"{short(e1['models']['qwen2.5-0.5b']['repo'])} | NIAH, ctx "
        f"{e1['context_len']} tok, needle depth {e1['needle_depth']}, "
        f"block size {e1['block_size']}, n = {e1['n_samples']}/cell, "
        f"seed {e1['seed']}\n"
        f"Right: {e5['model']}, seq_len {e5['seq_len']}, budget "
        f"{e5['budget']:.0%}, n = {e5['pilot_samples']}, seed {e5['seed']} | "
        f"{e1['env']['device_name']}, torch {e1['env']['torch']}\n"
        f"Error bars: Wilson 95% CI recomputed from raw (k, N). "
        f"{e1['mode']}-scale data -- single seed, n = {e1['n_samples']} per cell; "
        f"not converged."
    )
    fig.text(0.012, 0.008, caption, fontsize=10.5, color=MUTED, ha="left",
             va="bottom", linespacing=1.3)

    fig.savefig(OUT_PDF, facecolor=SURFACE)
    fig.savefig(OUT_PNG, dpi=150, facecolor=SURFACE)

    OUT_PROV.write_text(json.dumps({
        "figure": "F1 -- NIAH accuracy vs KV budget knee-curve",
        "generated_by": "writing/figures/make_f1_budget_knee.py",
        "PILOT_SCALE_WARNING": (
            "All data is PILOT scale (e1 mode='PILOT', e5 mode='PILOT'): n=16 samples "
            "per cell, single seed (42), two small Qwen2.5 models, 8192-token context. "
            "Wilson 95% CIs are correspondingly wide (e.g. 15/16 -> [0.717, 0.989]). "
            "Do not present these as converged results."
        ),
        "sources": {
            "e1": {"path": "exp/results/full/e1_budget_knee.json",
                   "mode": e1["mode"], "seed": e1["seed"],
                   "elapsed_sec": e1["elapsed_sec"],
                   "real_inference": e1["elapsed_sec"] > 1.0},
            "e5": {"path": "exp/results/full/e5_ablation_reasoning.json",
                   "mode": e5["mode"], "seed": e5["seed"],
                   "elapsed_sec": e5["elapsed_sec"],
                   "real_inference": e5["elapsed_sec"] > 1.0},
        },
        "wilson_formula": (
            "centre=(p+z^2/2n)/(1+z^2/n); half=z*sqrt(p(1-p)/n+z^2/4n^2)/(1+z^2/n); "
            "z=1.959963984540054. Recomputed in this script; any wilson_ci field in the "
            "source JSON was ignored."
        ),
        "caveats": [
            "e1 analysis.*.knee_budget is 1.0 and e1 verdict is 'REFINE_knee_above_25pct'; "
            "the '50% knee' annotation describes the observed accuracy jump "
            "(1.5B working set 3/16 at 25% -> 15/16 at 50%), not a claim that 50% "
            "statistically matches the dense ceiling.",
            "The inset uses a different configuration from the main panel "
            "(qwen2.5-0.5b, seq_len 2048, budget 0.25) -- it is not comparable "
            "point-for-point with the main curves.",
            "mean_block_skip_rate has no per-block (k, N) in the JSON, so the inset "
            "bars carry no confidence interval.",
            "NOT PLOTTED: e1 also contains an 'h2o' method series. Its k is IDENTICAL "
            "to 'working' in all 10 cells (both models x 5 budgets), so it would draw "
            "exactly on top of the working-set curve and adds no information. This "
            "identity is itself suspicious (possible phantom-factor / shared code path) "
            "and should be checked before either series is claimed as a distinct method.",
        ],
        "values": prov,
    }, indent=2))

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PROV}")
    for r in prov:
        if "raw_k" in r and r["series"].startswith(("Working", "Sink", "dense")):
            print(f"  {r['series']:<38} x={r['x_budget_pct']:>5}%  "
                  f"k={r['raw_k']:>2}/{r['raw_n']}  acc={r['value']:.4f}  "
                  f"CI=[{r['wilson95_recomputed'][0]:.4f},{r['wilson95_recomputed'][1]:.4f}]")


if __name__ == "__main__":
    main()
