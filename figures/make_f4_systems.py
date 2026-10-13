#!/usr/bin/env python3
"""Poster figure F4 -- FlexBudget systems payoff on a single data-center GPU.

Every plotted number is read from exp/results/full/e3_systems.json at runtime.
Nothing is hard-coded except axis furniture and honesty labels.

Run:
    <repo>/.venv/bin/python3 writing/figures/make_f4_systems.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # workspaces/pytorch_2026_context_management
SRC = ROOT / "results" / "e3_systems.json"
OUT_PDF = HERE / "f4_systems.pdf"
OUT_PNG = HERE / "f4_systems.png"
OUT_PROV = HERE / "f4_systems_provenance.json"

# ---------------------------------------------------------------- palette
# dataviz reference palette, light surface (#fcfcfb).
# Validated: node validate_palette.js "#2a78d6,#008300" --mode light -> ALL PASS
# (band, chroma, CVD dE 26.5, normal-vision dE 29.0, contrast >= 3:1).
BLUE = "#2a78d6"     # categorical slot 1 -- ratio panels (lower is better)
GREEN = "#008300"    # categorical slot 2 -- speedup panel (higher is better)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"

# ------------------------------------------------------------- typography
FS_TICK = 16
FS_AXLABEL = 20
FS_TITLE = 22
FS_VALUE = 15
FS_NOTE = 13

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,          # embed TrueType, keep text as text (vector)
    "ps.fonttype": 42,
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})


def load() -> dict:
    if not SRC.exists():
        sys.exit(f"FATAL: source data missing: {SRC}")
    with SRC.open() as fh:
        return json.load(fh)


def integrity_check(d: dict) -> list[str]:
    """Hard gates. Returns human-readable confirmations; exits on failure."""
    env = d.get("env", {})
    notes = []
    if not env.get("cuda_available"):
        sys.exit("FATAL: env.cuda_available is not true -- refusing to plot.")
    dev = env.get("device_name", "")
    if "MI300X" not in dev:
        sys.exit(f"FATAL: unexpected device_name {dev!r} -- expected an MI300X (gfx942).")
    elapsed = d.get("elapsed_sec", 0)
    if elapsed <= 1:
        sys.exit(f"FATAL: elapsed_sec={elapsed} <= 1s -- this is a mock run, refusing to plot.")
    if d.get("mock_flag") is not False:
        sys.exit("FATAL: mock_flag is not False -- refusing to plot.")
    notes.append(f"cuda_available=True, device_name={dev} (gfx942), elapsed_sec={elapsed}")
    return notes


def main() -> None:
    d = load()
    integrity_check(d)

    env = d["env"]
    sweep = d["length_sweep"]
    lens = ["8192", "16384", "32768"]
    xlabels = ["8K", "16K", "32K"]

    # -------------------------------------------------- values (from JSON only)
    kv_ratio = [sweep[L]["kv_mem_ratio"] for L in lens]
    kv_save = [sweep[L]["kv_mem_saving_pct"] for L in lens]
    ttft_ratio = [sweep[L]["ttft_ratio"] for L in lens]
    dec_ws = [sweep[L]["decode_tok_s_ws"] for L in lens]
    dec_dense = [sweep[L]["decode_tok_s_dense"] for L in lens]
    speedup = [w / s for w, s in zip(dec_ws, dec_dense)]

    x = list(range(len(lens)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 7.0))

    def dress(ax, ylabel, title, subtitle, ymax):
        ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=72, loc="left")
        ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=FS_NOTE,
                color=INK2, ha="left", va="bottom", linespacing=1.55)
        ax.set_ylabel(ylabel, fontsize=FS_AXLABEL, color=INK2, labelpad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=FS_TICK, color=INK)
        ax.set_xlabel("context length", fontsize=FS_AXLABEL, color=INK2, labelpad=4)
        ax.tick_params(axis="y", labelsize=FS_TICK, colors=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.62, len(x) - 0.38)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.spines["bottom"].set_linewidth(1.2)
        for lab in ax.get_yticklabels():
            lab.set_color(MUTED)

    def bars(ax, vals, color):
        return ax.bar(x, vals, width=0.56, color=color, zorder=3,
                      edgecolor=SURFACE, linewidth=2.0)

    def dense_line(ax, text, ymax):
        ax.axhline(1.0, color=INK2, linestyle=(0, (5, 4)), linewidth=2.0, zorder=4)
        ax.text(-0.58, 1.0 + 0.02 * ymax, text, fontsize=FS_VALUE, color=INK2,
                va="bottom", ha="left", zorder=5)

    def label(ax, vals, fmt, extra=None, ymax=1.0):
        for xi, v in zip(x, vals):
            top = v + 0.022 * ymax
            if extra is not None:
                ax.text(xi, top, extra[xi], ha="center", va="bottom",
                        fontsize=FS_NOTE, color=INK2, zorder=5)
                top += 0.055 * ymax
            ax.text(xi, top, fmt.format(v), ha="center", va="bottom",
                    fontsize=FS_VALUE, color=INK, zorder=5)

    # ---------------------------------------------------------------- Panel A
    ymA = 1.34
    axA = axes[0]
    bars(axA, kv_ratio, BLUE)
    dense_line(axA, "dense = 1.0", ymA)
    dress(axA, "ratio vs dense", "A  KV working set",
          "resident per-query; lower is better\n"
          "dense:      %s GB\nFlexBudget: %s GB" % (
              " / ".join(f"{sweep[L]['dense_kv_gb']:.2f}" for L in lens),
              " / ".join(f"{sweep[L]['ws_kv_gb']:.2f}" for L in lens)), ymA)
    label(axA, kv_ratio, "{:.2f}x",
          extra=[f"-{v:.1f}%" for v in kv_save], ymax=ymA)
    axA.yaxis.set_major_locator(MultipleLocator(0.25))

    # ---------------------------------------------------------------- Panel B
    ymB = 1.34
    axB = axes[1]
    bars(axB, ttft_ratio, BLUE)
    dense_line(axB, "dense = 1.0", ymB)
    dress(axB, "ratio vs dense", "B  Time to first token",
          "block-sparse prefill; lower is better\n"
          "dense:      %s ms\nFlexBudget: %s ms" % (
              " / ".join(f"{sweep[L]['ttft_dense_ms']:.2f}" for L in lens),
              " / ".join(f"{sweep[L]['ttft_ws_ms']:.2f}" for L in lens)), ymB)
    label(axB, ttft_ratio, "{:.2f}x", ymax=ymB)
    axB.yaxis.set_major_locator(MultipleLocator(0.25))

    # ---------------------------------------------------------------- Panel C
    ymC = 4.6
    axC = axes[2]
    bars(axC, speedup, GREEN)
    axC.set_xlim(-1.5, len(x) - 0.38)
    axC.axhline(1.0, color=INK2, linestyle=(0, (5, 4)), linewidth=2.0, zorder=4)
    axC.text(-1.42, 1.0 + 0.025 * ymC, "dense 1.0x", fontsize=FS_VALUE,
             color=INK2, va="bottom", ha="left", zorder=5,
             bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.0))
    dress(axC, "speedup vs dense", "C  Decode throughput",
          "block-sparse KV gather; higher is better\n"
          "dense:      %s tok/s\nFlexBudget: %s tok/s" % (
              " / ".join(f"{v:.0f}" for v in dec_dense),
              " / ".join(f"{v:.0f}" for v in dec_ws)), ymC)
    axC.set_xlim(-1.5, len(x) - 0.38)
    label(axC, speedup, "{:.2f}x", ymax=ymC)
    axC.yaxis.set_major_locator(MultipleLocator(1.0))

    # ----------------------------------------------------------- figure text
    rmf = d.get("real_model_forward", {}).get("8192", {})
    header = f"Measured on one {env['device_name']}"
    # The full geometry (Q/KV heads, layers, head_dim) used to be spelled out
    # here in 6 pt type nobody reads at poster scale, and the poster caption
    # already states the model and budget. Keep only what frames the panels.
    subheader = (f"Qwen2.5-7B, budget {d['budget']:.2f}, block size {d['block_size']}")
    fig.text(0.005, 1.10, header, fontsize=FS_TITLE, color=INK, ha="left", va="top")
    fig.text(0.005, 1.035, subheader, fontsize=FS_VALUE, color=INK2, ha="left", va="top")

    foot_lines = []
    foot_lines.append(
        "NOTE ON PANEL A: this is the RESIDENT PER-QUERY KV working set -- the bytes a paging / "
        "eviction backend would keep hot -- NOT a shrink of the statically allocated KV tensor. "
        "For this single FlexAttention forward the same JSON honestly records peak_alloc_ratio = "
        "1.00 and union_frac = 1.00 (every block is touched by some query): peak allocator memory "
        "did NOT drop. The union bound is the labelled diagnostic upper bound; the per-query mean "
        "is the paging-backend target."
    )
    if rmf.get("real_forward_ok"):
        foot_lines.append(
            f"Real {rmf['model']} forward at {rmf['seq_len'] // 1024}K on the same GPU: "
            f"peak {rmf['real_peak_alloc_gb']:.2f} GB, prefill TTFT {rmf['real_prefill_ttft_ms']:.2f} ms."
        )
    foot_lines.append(
        f"Source: exp/results/full/e3_systems.json (mode {d['mode']}, seed {d['seed']}, "
        f"torch {env['torch']}, HIP {env['hip']}, cuda_available=true, "
        f"elapsed {d['elapsed_sec']:.2f} s, mock_flag=false). Every bar is provenance-tracked in "
        "f4_systems_provenance.json."
    )
    foot = "\n".join(textwrap.fill(t, width=132) for t in foot_lines)
    fig.text(0.005, -0.02, foot, fontsize=FS_NOTE, color=INK2,
             ha="left", va="top", linespacing=1.45)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.subplots_adjust(wspace=0.27)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------ provenance
    prov = {
        "figure": "f4_systems",
        "generated_by": str(HERE / "make_f4_systems.py"),
        "source_file": str(SRC.relative_to(ROOT)),
        "integrity": {
            "env.cuda_available": env["cuda_available"],
            "env.device_name": env["device_name"],
            "device_arch_note": "MI300X == gfx942 (GRES tag gfx942 in exp/code/task_4_e3_systems_job.sh)",
            "elapsed_sec": d["elapsed_sec"],
            "elapsed_gt_1s": d["elapsed_sec"] > 1,
            "mock_flag": d["mock_flag"],
            "torch": env["torch"],
            "hip": env["hip"],
        },
        "honesty_note": (
            "Panel A plots length_sweep[L].kv_mem_ratio = mean per-query kept fraction of the KV "
            "cache (resident working set for a paging/eviction backend). The same records report "
            "peak_alloc_ratio = 1.0 and union_frac = 1.0 for this single FlexAttention forward: "
            "peak allocator memory is unchanged. Do not read Panel A as an allocator-memory saving."
        ),
        "bars": [],
    }
    for i, L in enumerate(lens):
        prov["bars"].append({
            "panel": "A", "panel_title": "resident per-query KV working set",
            "context": xlabels[i], "json_file": str(SRC.relative_to(ROOT)),
            "json_path": f"length_sweep.{L}.kv_mem_ratio",
            "raw_value": sweep[L]["kv_mem_ratio"],
            "derived_ratio": sweep[L]["kv_mem_ratio"],
            "supporting": {
                "length_sweep.%s.mean_perquery_kept_frac" % L: sweep[L]["mean_perquery_kept_frac"],
                "length_sweep.%s.kv_mem_saving_pct" % L: sweep[L]["kv_mem_saving_pct"],
                "length_sweep.%s.dense_kv_gb" % L: sweep[L]["dense_kv_gb"],
                "length_sweep.%s.ws_kv_gb" % L: sweep[L]["ws_kv_gb"],
                "length_sweep.%s.peak_alloc_ratio" % L: sweep[L]["peak_alloc_ratio"],
                "length_sweep.%s.union_frac" % L: sweep[L]["union_frac"],
            },
        })
    for i, L in enumerate(lens):
        prov["bars"].append({
            "panel": "B", "panel_title": "time to first token",
            "context": xlabels[i], "json_file": str(SRC.relative_to(ROOT)),
            "json_path": f"length_sweep.{L}.ttft_ratio",
            "raw_value": sweep[L]["ttft_ratio"],
            "derived_ratio": sweep[L]["ttft_ratio"],
            "supporting": {
                "length_sweep.%s.ttft_ws_ms" % L: sweep[L]["ttft_ws_ms"],
                "length_sweep.%s.ttft_dense_ms" % L: sweep[L]["ttft_dense_ms"],
            },
        })
    for i, L in enumerate(lens):
        prov["bars"].append({
            "panel": "C", "panel_title": "decode throughput speedup",
            "context": xlabels[i], "json_file": str(SRC.relative_to(ROOT)),
            "json_path": (f"length_sweep.{L}.decode_tok_s_ws / "
                          f"length_sweep.{L}.decode_tok_s_dense"),
            "raw_value": {"decode_tok_s_ws": dec_ws[i], "decode_tok_s_dense": dec_dense[i]},
            "derived_ratio": round(speedup[i], 4),
        })
    if rmf.get("real_forward_ok"):
        prov["annotations"] = [{
            "text": "real Qwen2.5-7B forward at 8K",
            "json_file": str(SRC.relative_to(ROOT)),
            "json_path": "real_model_forward.8192",
            "raw_value": rmf,
        }]
    with OUT_PROV.open("w") as fh:
        json.dump(prov, fh, indent=2)

    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PROV}")
    print("A kv_mem_ratio  :", kv_ratio)
    print("B ttft_ratio    :", ttft_ratio)
    print("C decode speedup:", [round(s, 4) for s in speedup])


if __name__ == "__main__":
    main()
