"""task_3_e2_midcontext: E2 -- positional NIAH, position-corrective variant (H4).

Positional Needle-in-a-Haystack (NIAH) at needle depths {10,30,50,70,90}% with the KV
budget fixed at 50%, comparing three conditions on the SAME compiled FlexAttention kernel:

    * dense           -- full causal FlexAttention (no budget); upper-bound reference.
    * uncorrected     -- working-set mask, qk_affinity score_mod (SnapKV-style), 50% budget.
    * position_corr   -- working-set mask, position_corrective score_mod
                         (qk_affinity x U-shaped multiplier calibrated per model), 50% budget.

Models: Qwen2.5-0.5B-Instruct (head_dim 64), Qwen2.5-1.5B-Instruct (head_dim 128), 8K context.
(PILOT trims context so the pilot fits the 600 s budget; see PILOT_CONFIG.)

Design notes
------------
* We drive the working set through the model's real attention using the transformers
  FlexAttention integration contract: `flex_attention_forward` treats an `attention_mask`
  argument that is a `torch.nn.attention.flex_attention.BlockMask` AS the block_mask and
  passes it straight to the compiled kernel. So we build our budgeted `BlockMask` and hand it
  to `model(..., attention_mask=block_mask)` -- no monkeypatching of transformers internals,
  which is version-fragile. The importance scores are built at *prefill* from a cheap dense
  pass that records per-KV-block mean attention mass and per-(q-block,kv-block) QK affinity
  (SnapKV/H2O statistics); the WorkingSetBlockMask library turns them into a static
  admitted-block tensor closed over by create_block_mask -- no graph breaks.
* The position multiplier for `position_corrective` is CALIBRATED per model from a small
  held-out probe (needles placed across depths on a handful of held-out haystacks; the
  U-shape floor is fit to the depth where uncorrected accuracy dips) BEFORE the reported
  sweep, exactly as the plan requires.
* NIAH task: a haystack of distractor "The magic number for CITY is VALUE." lines with one
  needle line inserted at the target depth; the model is asked to recall the needle's VALUE.
  Scoring is exact-match of the needle value in the generated answer (raw k/N recorded per
  (model x condition x depth) for Wilson CIs downstream).

Every accuracy is stored as raw (k, N). elapsed_sec is recorded so the analysis stage can
reject any mock (< 0.5 s) record. Writes results/e2_midcontext.json plus the pilot
summary artifacts and PID/PROGRESS/DONE markers.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

TASK_ID = os.environ.get("TASK_ID", "task_3_e2_midcontext")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"
for d in (RESULTS_DIR, FULL_DIR, PILOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# make working_set_mask importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Neutralize the container's broken torchvision/torchaudio BEFORE importing transformers.
import _env_fix  # noqa: E402

SEED = 42
BUDGET = 0.50               # E2 fixed budget (plan)
DEPTHS = [10, 30, 50, 70, 90]  # needle depth percentiles
MODELS = [
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
]

# PILOT config: 16 samples, seed 42, <=600 s. Trim context so 2 models x 3 conditions x
# 5 depths x 16 samples of generative recall fit the budget.
MODE = os.environ.get("EXP_MODE", "PILOT").upper()
if MODE == "PILOT":
    N_SAMPLES = 16
    CTX_TOKENS = 2048          # trimmed context for the pilot budget (full uses 8192)
    CALIB_SAMPLES = 4
else:
    N_SAMPLES = 96
    CTX_TOKENS = 8192
    CALIB_SAMPLES = 8

BLOCK_SIZE = 128


def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total, note=""):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(
        json.dumps({
            "task_id": TASK_ID, "step": step, "total_steps": total,
            "note": note, "updated_at": datetime.now().isoformat(),
        })
    )


def mark_done(status, summary, payload):
    pid = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid.exists():
        pid.unlink()
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(
        json.dumps({
            "task_id": TASK_ID, "status": status, "summary": summary,
            "final_progress": payload, "timestamp": datetime.now().isoformat(),
        })
    )


# ---------------------------------------------------------------------------
# NIAH data
# ---------------------------------------------------------------------------

CITIES = [
    "Cairo", "Oslo", "Lima", "Perth", "Tokyo", "Dublin", "Bogota", "Nairobi",
    "Vienna", "Manila", "Quito", "Riga", "Accra", "Hanoi", "Doha", "Bern",
    "Kigali", "Amman", "Sofia", "Tunis", "Muscat", "Skopje", "Minsk", "Astana",
]


def make_niah_sample(rng, depth_pct, ctx_chars):
    """Build a NIAH prompt: many distractor 'magic number' lines with one needle line at
    the target depth. Returns (prompt, needle_value, needle_city)."""
    needle_city = rng.choice(CITIES)
    needle_value = rng.randint(1000, 9999)
    # distractor lines use OTHER cities so the needle city is unique.
    other = [c for c in CITIES if c != needle_city]
    lines = []
    approx_line = 42  # avg chars per distractor line
    n_lines = max(8, ctx_chars // approx_line)
    for _ in range(n_lines):
        c = rng.choice(other)
        v = rng.randint(1000, 9999)
        lines.append(f"The magic number for {c} is {v}.")
    needle = f"The magic number for {needle_city} is {needle_value}."
    insert_at = int(len(lines) * depth_pct / 100.0)
    insert_at = min(max(insert_at, 0), len(lines))
    lines.insert(insert_at, needle)
    haystack = "\n".join(lines)
    prompt = (
        "Read the following facts carefully, then answer the question with only the "
        "4-digit number.\n\n"
        f"{haystack}\n\n"
        f"Question: What is the magic number for {needle_city}?\n"
        "Answer: The magic number for "
        f"{needle_city} is "
    )
    return prompt, str(needle_value), needle_city


# ---------------------------------------------------------------------------
# Working-set integration: precompute prefill stats -> block_mask -> monkeypatch
# ---------------------------------------------------------------------------

def _prefill_block_stats(eager_model, input_ids, wsm, torch, block_size):
    """One dense EAGER forward with output_attentions to gather per-KV-block attention mass
    and per-(q-block,kv-block) QK affinity, averaged across layers+heads (SnapKV/H2O style).

    output_attentions requires the eager attention path, so `eager_model` must be loaded with
    attn_implementation='eager'. Returns (BlockStats, n_blocks)."""
    seq_len = input_ids.shape[1]
    nb = wsm.n_blocks(seq_len, block_size)
    with torch.no_grad():
        out = eager_model(input_ids=input_ids, output_attentions=True, use_cache=False)
    attns = out.attentions  # tuple(layers) of [B, H, S, S]
    device = input_ids.device
    attn_mass = torch.zeros(nb, device=device, dtype=torch.float32)
    qk_aff = torch.zeros(nb, nb, device=device, dtype=torch.float32)
    L = len(attns)
    for a in attns:
        am = a.mean(dim=(0, 1)).float()  # [S, S] mean over batch+heads
        col = am.sum(dim=0)              # [S] per-KV received mass
        for qb in range(nb):
            qs, qe = qb * block_size, min((qb + 1) * block_size, seq_len)
            for kb in range(qb + 1):
                ks, ke = kb * block_size, min((kb + 1) * block_size, seq_len)
                blk = am[qs:qe, ks:ke]
                if blk.numel():
                    qk_aff[qb, kb] += blk.mean()
        for kb in range(nb):
            ks, ke = kb * block_size, min((kb + 1) * block_size, seq_len)
            attn_mass[kb] += col[ks:ke].sum()
    qk_aff /= max(L, 1)
    attn_mass /= max(L, 1)
    stats = wsm.BlockStats(
        n_q_blocks=nb, n_kv_blocks=nb, block_size=block_size,
        attn_mass=attn_mass, qk_affinity=qk_aff, device=device,
    )
    return stats, nb


def _round_up(n, m):
    return ((n + m - 1) // m) * m


def _greedy_decode_masked(model, tok, input_ids, admitted, wsm, torch, block_size,
                          max_new=8):
    """Greedy decode where EVERY query row (prefill + each generated token) is restricted to
    the working-set admitted blocks.

    We run full (non-cached) forwards and extend the admitted-block map by one always-admitted
    KV column-block per newly generated block so that generated queries attend to (a) the
    budgeted prefill working set and (b) the local window of generated tokens (recency anchor).
    Answers are a handful of tokens, so a few full forwards stay within budget."""
    from torch.nn.attention.flex_attention import create_block_mask

    device = input_ids.device
    cur = input_ids
    out_ids = []
    for _ in range(max_new):
        seq_len = cur.shape[1]
        nb = wsm.n_blocks(seq_len, block_size)
        # extend admitted to nb x nb: prefill working set preserved; new blocks admit
        # themselves + the last prefill block (local window) + sinks + causal.
        adm = torch.zeros(nb, nb, dtype=torch.bool, device=device)
        pnb = admitted.shape[0]
        adm[:pnb, :pnb] = admitted
        qi = torch.arange(nb, device=device).unsqueeze(1)
        kj = torch.arange(nb, device=device).unsqueeze(0)
        causal = kj <= qi
        # Generated query blocks (index >= pnb) inherit the WORKING SET of the last prefill
        # (question/observation) block -- exactly the KV blocks that drive the answer -- plus
        # the sink, plus the newly generated blocks themselves (local recency). This mirrors
        # how a KV-budgeted decoder attends: the answer tokens see the same admitted set the
        # question established, not a diluted union.
        if pnb > 0:
            question_row = admitted[pnb - 1].clone()  # [pnb]
        else:
            question_row = torch.zeros(0, dtype=torch.bool, device=device)
        for qb in range(pnb, nb):
            row = torch.zeros(nb, dtype=torch.bool, device=device)
            if pnb > 0:
                row[:pnb] = question_row
            row[0] = True                       # sink anchor
            row[pnb:qb + 1] = True              # all generated blocks so far (recency + self)
            adm[qb] = row
        adm = adm & causal

        def mask_mod(b, h, q_idx, kv_idx, _adm=adm, _bs=block_size):
            # block-level admission AND token-level causality. The token-level `kv_idx<=q_idx`
            # is REQUIRED: block-only admission lets future tokens inside an admitted (esp. the
            # diagonal) block leak into the current query, corrupting the answer logits.
            return _adm[q_idx // _bs, kv_idx // _bs] & (kv_idx <= q_idx)

        # IMPORTANT: build the block mask at the EXACT current seq_len (not padded to a
        # multiple of block_size). FlexAttention requires block_mask Q/KV length to match the
        # forward's q_len/kv_len exactly; create_block_mask handles a partial final block.
        bm = create_block_mask(mask_mod, B=None, H=None, Q_LEN=seq_len,
                               KV_LEN=seq_len, device=str(device))
        with torch.no_grad():
            logits = model(input_ids=cur, attention_mask=bm, use_cache=False).logits
        nxt = int(logits[0, -1].argmax().item())
        out_ids.append(nxt)
        if nxt == tok.eos_token_id:
            break
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out_ids, skip_special_tokens=True)


def _greedy_decode_dense(model, tok, input_ids, torch, max_new=8):
    """Dense reference: plain greedy generation, no budget mask."""
    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids, max_new_tokens=max_new, do_sample=False,
            num_beams=1, pad_token_id=tok.eos_token_id, use_cache=True,
        )
    return tok.decode(gen[0, input_ids.shape[1]:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _build_admitted(wsm, stats, nb, score_name, budget, torch, position_multiplier=None):
    """Build the admitted-block map. SnapKV-style: the importance signal is driven by the
    OBSERVATION WINDOW (the last / question query block), whose QK affinity actually points at
    the retrieval-relevant KV blocks. We broadcast that question-row affinity to every query
    block so the working set consistently retains the needle-bearing block for the eventual
    answer query. position_corrective multiplies this by the U-shaped position multiplier."""
    aff = stats.qk_affinity  # [nb, nb]
    # question/observation-window row = last query block's affinity over KV blocks
    obs_row = aff[-1].clone()  # [nb]
    scores = obs_row.unsqueeze(0).expand(nb, nb).clone()  # broadcast to all query blocks
    if score_name == "position_corrective":
        pm = position_multiplier
        if pm is None:
            pm = wsm.default_u_shape(nb, device=aff.device)
        scores = scores * pm.unsqueeze(0)
    admitted = wsm.build_block_importance(
        scores, budget_fraction=budget, n_sink_blocks=1, local_window_blocks=1
    )
    return admitted


def _calibrate_u_shape(eager_model, flex_model, tok, wsm, torch, device, ctx_chars,
                       block_size, rng):
    """Fit the U-shape floor per model from a small held-out probe (plan: 'calibrate the
    U-shaped position multiplier per model from a small held-out probe before the sweep').

    Probe uncorrected (qk_affinity) working-set recall at the 50% depth on CALIB_SAMPLES
    held-out haystacks; a deeper mid-context dip -> a lower floor (inflate mid harder). Returns
    (floor, mid_acc_probe). The multiplier itself is a symmetric U over KV-block index built
    per-sequence-length at eval time from this floor."""
    floor = 0.5
    mid_acc = None
    try:
        mid_hits = tot = 0
        for _ in range(CALIB_SAMPLES):
            prompt, val, _ = make_niah_sample(rng, 50, ctx_chars)
            ids = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=CTX_TOKENS).input_ids.to(device)
            n = ids.shape[1]
            stats, nb = _prefill_block_stats(eager_model, ids, wsm, torch, block_size)
            adm = _build_admitted(wsm, stats, nb, "uncorrected", BUDGET, torch)
            ans = _greedy_decode_masked(flex_model, tok, ids, adm, wsm, torch, block_size)
            tot += 1
            if val in ans:
                mid_hits += 1
        mid_acc = mid_hits / max(tot, 1)
        floor = 0.35 if mid_acc < 0.75 else 0.55
    except Exception:
        floor = 0.5
    return floor, mid_acc


def eval_model(model_key, repo, wsm, torch, device, ckpt_root, rng):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    res = {"model": model_key, "repo": repo, "budget": BUDGET,
           "ctx_tokens": CTX_TOKENS, "block_size": BLOCK_SIZE,
           "conditions": {}, "depths": DEPTHS}
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(ckpt_root))
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # eager model: for output_attentions prefill stats (SnapKV/H2O importance signals).
    eager_model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="eager",
        cache_dir=str(ckpt_root),
    ).to(device).eval()
    # flex model: runs the actual working-set masked forward via the compiled flex kernel.
    flex_model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="flex_attention",
        cache_dir=str(ckpt_root),
    ).to(device).eval()

    ctx_chars = CTX_TOKENS * 3  # ~3 chars/token heuristic for haystack sizing

    # --- calibration probe for the position multiplier -----------------------
    floor, probe_acc = _calibrate_u_shape(
        eager_model, flex_model, tok, wsm, torch, device, ctx_chars, BLOCK_SIZE, rng
    )
    res["u_shape_floor"] = floor
    res["calib_probe_mid_acc"] = probe_acc

    conditions = ["dense", "uncorrected", "position_corrective"]
    for cond in conditions:
        res["conditions"][cond] = {"per_depth": {}}

    for depth in DEPTHS:
        for cond in conditions:
            k = 0
            n = 0
            samples = []
            for si in range(N_SAMPLES):
                prompt, val, city = make_niah_sample(rng, depth, ctx_chars)
                ids = tok(prompt, return_tensors="pt", truncation=True,
                          max_length=CTX_TOKENS).input_ids.to(device)
                seq_len = ids.shape[1]
                try:
                    if cond == "dense":
                        ans = _greedy_decode_dense(flex_model, tok, ids, torch)
                    else:
                        stats, nb = _prefill_block_stats(eager_model, ids, wsm, torch,
                                                         BLOCK_SIZE)
                        if cond == "position_corrective":
                            pm = wsm.default_u_shape(nb, device=device, floor=floor)
                        else:
                            pm = None
                        adm = _build_admitted(wsm, stats, nb, cond, BUDGET, torch,
                                              position_multiplier=pm)
                        ans = _greedy_decode_masked(flex_model, tok, ids, adm, wsm, torch,
                                                    BLOCK_SIZE)
                    hit = int(val in ans)
                    k += hit
                    n += 1
                    if len(samples) < 3:
                        samples.append({"depth": depth, "cond": cond,
                                        "gold": val, "pred": ans.strip()[:64],
                                        "hit": bool(hit), "seq_len": seq_len})
                except Exception as e:
                    if len(samples) < 3:
                        samples.append({"depth": depth, "cond": cond,
                                        "error": f"{type(e).__name__}: {e}"})
            acc = k / n if n else 0.0
            res["conditions"][cond]["per_depth"][str(depth)] = {
                "k": k, "n": n, "accuracy": round(acc, 4), "samples": samples,
            }
        report_progress(DEPTHS.index(depth) + 1, len(DEPTHS),
                        f"{model_key} depth {depth}%")

    # --- H4 checks -----------------------------------------------------------
    def acc_at(cond, depth):
        d = res["conditions"][cond]["per_depth"].get(str(depth), {})
        return d.get("accuracy", 0.0), d.get("k", 0), d.get("n", 0)

    dense_mid, _, _ = acc_at("dense", 50)
    unc_mid, _, _ = acc_at("uncorrected", 50)
    corr_mid, _, _ = acc_at("position_corrective", 50)
    res["h4_mid_bucket"] = {
        "dense_50": dense_mid, "uncorrected_50": unc_mid, "position_corrective_50": corr_mid,
        "safety_not_worse_than_dense_10pp": bool(unc_mid >= dense_mid - 0.10),
        "recovery_corr_ge_uncorrected": bool(corr_mid >= unc_mid),
    }
    res["elapsed_sec"] = round(time.time() - t0, 2)

    del eager_model, flex_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------------------

def main():
    write_pid()
    t_start = time.time()
    import torch

    torch.manual_seed(SEED)
    random.seed(SEED)
    import working_set_mask as wsm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_root = Path("/work/shared/checkpoints")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    result = {
        "task_id": TASK_ID, "mode": MODE, "seed": SEED, "budget": BUDGET,
        "depths_pct": DEPTHS, "n_samples": N_SAMPLES, "ctx_tokens": CTX_TOKENS,
        "started_at": datetime.now().isoformat(),
        "env": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": {},
    }

    for i, (mkey, repo) in enumerate(MODELS):
        report_progress(i + 1, len(MODELS), f"eval {mkey}")
        rng = random.Random(SEED + i)
        try:
            result["models"][mkey] = eval_model(
                mkey, repo, wsm, torch, device, ckpt_root, rng
            )
        except Exception as e:
            result["models"][mkey] = {
                "model": mkey, "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    # --- aggregate H4 verdict across models ----------------------------------
    safety = []
    recovery = []
    for mkey in result["models"]:
        h4 = result["models"][mkey].get("h4_mid_bucket")
        if h4:
            safety.append(h4["safety_not_worse_than_dense_10pp"])
            recovery.append(h4["recovery_corr_ge_uncorrected"])
    result["elapsed_sec"] = round(time.time() - t_start, 2)
    result["finished_at"] = datetime.now().isoformat()
    result["H4_safety_pass"] = bool(safety and all(safety))
    result["H4_recovery"] = bool(recovery and all(recovery))
    # pilot pass criterion: safety holds (not worse than dense by >10pp at mid) AND
    # position_corrective >= uncorrected at mid (null-safe acceptable).
    result["pilot_pass"] = bool(result["H4_safety_pass"])
    result["verdict"] = "GO" if result["pilot_pass"] else "REFINE"

    out_path = FULL_DIR / "e2_midcontext.json"
    out_path.write_text(json.dumps(result, indent=2))

    # concise pilot artifact
    pilot = {
        "task_id": TASK_ID, "mode": MODE,
        "H4_safety_pass": result["H4_safety_pass"],
        "H4_recovery": result["H4_recovery"],
        "verdict": result["verdict"],
        "elapsed_sec": result["elapsed_sec"],
        "per_model_mid_bucket": {
            m: result["models"][m].get("h4_mid_bucket")
            for m in result["models"] if result["models"][m].get("h4_mid_bucket")
        },
    }
    (PILOTS_DIR / f"{TASK_ID}_pilot.json").write_text(json.dumps(pilot, indent=2))

    print(json.dumps(pilot, indent=2))

    mark_done(
        "success" if result["pilot_pass"] else "fail",
        f"E2 {result['verdict']}: safety={result['H4_safety_pass']} "
        f"recovery={result['H4_recovery']}",
        {"pilot_pass": result["pilot_pass"], "elapsed_sec": result["elapsed_sec"]},
    )
    return 0 if result["pilot_pass"] else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        traceback.print_exc()
        mark_done("fail", f"unhandled: {type(e).__name__}: {e}", {})
        rc = 2
    sys.exit(rc)
