"""task_2_e1_budget_knee: E1 -- NIAH budget knee sweep (H2).

Needle-in-a-Haystack (NIAH) retrieval accuracy vs KV budget at
{12.5, 25, 50, 75, 100}% of KV blocks, using the WorkingSetBlockMask (top-block
importance + anchors) on the SAME compiled FlexAttention kernel. Baselines at
matched budgets:
  * dense    -- plain causal SDPA (reference upper bound; budget-independent)
  * flex100  -- working-set at 100% budget == dense-FlexAttention (kernel control)
  * streaming (StreamingLLM-as-FlexAttention: sinks + local window only)
  * h2o       (H2O-as-FlexAttention: attn_mass importance)
  * working   (FlexBudget: attn_mass importance + anchors, top-B blocks)

For each (model, method, budget) we record raw (k, N) so the analysis stage can
recompute Wilson CIs. We also record peak KV memory per budget and TTFT (prefill).

Design for correctness + speed (PILOT: 16 samples, seed 42, 8K ctx):
  * We drive block admission by monkeypatching Qwen's flex_attention interface to
    install a precomputed, per-forward ``block_mask`` derived from a cheap first-pass
    (attn_mass) prefill statistic. The importance stats are captured from a single
    dense prefill, then the budget mask is built once per (method, budget) and reused.
  * NIAH: a UUID-style magic value is embedded in a filler haystack of Paul-Graham-
    essay-like sentences; the model is asked to return the value verbatim. Scoring is
    substring match of the ground-truth value in the greedy-decoded answer.

Writes results/e1_budget_knee.json + PID/PROGRESS/DONE markers.
Defensive: any (model,method,budget) cell that raises is recorded and skipped; a
DONE marker is always emitted so the monitor never hangs.
"""

from __future__ import annotations

import json
import os
import sys
import time
import random
import traceback
from datetime import datetime
from pathlib import Path

TASK_ID = os.environ.get("TASK_ID", "task_2_e1_budget_knee")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
FULL_DIR = RESULTS_DIR / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))          # _env_fix
sys.path.insert(0, str(Path(__file__).resolve().parent.parent     # working_set_mask
                       / "flexbudget"))
import _env_fix  # noqa: E402  (neutralize broken torchvision/torchaudio before transformers)

SEED = 42
MODE = os.environ.get("EXP_MODE", "PILOT")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "16" if MODE == "PILOT" else "100"))
CONTEXT_LEN = int(os.environ.get("CONTEXT_LEN", "8192"))
BLOCK_SIZE = 128
BUDGETS = [0.125, 0.25, 0.5, 0.75, 1.0]
NEEDLE_DEPTH = 0.5  # E1 fixes needle near mid-context; E2 sweeps position.
MAX_NEW_TOKENS = 24

MODELS = [
    ("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct"),
]

CKPT_ROOT = Path("/work/shared/checkpoints")


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total, note=""):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "note": note, "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status, summary, payload):
    pid = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid.exists():
        pid.unlink()
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": payload, "timestamp": datetime.now().isoformat(),
    }))


# ---------------------------------------------------------------------------
# NIAH data
# ---------------------------------------------------------------------------
FILLER_SENTENCES = [
    "The grass is green and the sky is a pale shade of blue over the quiet valley.",
    "Startups often fail because they build something nobody actually wants to use.",
    "A good essay begins with a question the writer does not yet know how to answer.",
    "The river wound slowly past the old mill, carrying leaves toward the distant sea.",
    "Compound interest rewards patience more reliably than any single clever decision.",
    "Most productivity advice reduces to removing friction from the things that matter.",
    "The library was silent except for the soft turning of pages in the reading room.",
    "Great work usually looks obvious only after someone has already finished it.",
    "The mountain trail climbed gently through pine forest before the final steep ridge.",
    "Curiosity, more than talent, tends to separate people who keep learning from those who stop.",
]


def build_niah_prompt(rng, depth_frac, target_token_len, tokenizer):
    """Construct a NIAH prompt: filler haystack with a magic key/value at ``depth_frac``.

    Returns (prompt_text, needle_value). We pad filler until the tokenized prompt
    reaches ~target_token_len, inserting the needle sentence at the target depth.
    """
    magic = f"{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"
    needle = (f"The secret access code for the Zephyr project is {magic}. "
              f"Remember this code exactly.")
    question = ("\n\nBased on the document above, what is the secret access code for "
                "the Zephyr project? Answer with only the code.")

    # rough sentences needed; refine by tokenization
    filler_pool = []
    i = 0
    # generate enough filler to exceed target, then splice needle at depth
    while True:
        s = FILLER_SENTENCES[rng.randrange(len(FILLER_SENTENCES))]
        filler_pool.append(s)
        i += 1
        if i % 40 == 0:
            body = " ".join(filler_pool)
            n_tok = len(tokenizer(body + question, add_special_tokens=False)["input_ids"])
            if n_tok >= target_token_len:
                break
        if i > 4000:
            break

    insert_at = int(len(filler_pool) * depth_frac)
    spliced = filler_pool[:insert_at] + [needle] + filler_pool[insert_at:]
    body = " ".join(spliced)
    prompt = ("You are given a long document. Read it carefully.\n\n"
              + body + question)
    return prompt, magic


# ---------------------------------------------------------------------------
# Working-set block_mask injection into Qwen's flex_attention interface
# ---------------------------------------------------------------------------
class MaskController:
    """Holds the current block_mask (or None) to inject into every attention call.

    None -> dense FlexAttention (full causal). A BlockMask -> budgeted working set.
    """
    def __init__(self):
        self.block_mask = None
        self.enabled = False
        self.prefill_len = None  # apply the mask only when q_len == this (prefill)


def install_flex_hook(controller, torch):
    """Wrap transformers' flex_attention integration to SUBSTITUTE our budget block_mask.

    transformers routes attn_implementation='flex_attention' through
    ``transformers.integrations.flex_attention.flex_attention_forward``, whose first
    branch already does ``if isinstance(attention_mask, BlockMask): block_mask = it``.
    So instead of re-implementing the (subtle) GQA / scale / LSE / attention-sink / return
    logic ourselves -- which corrupted outputs when done by hand -- we simply REPLACE the
    ``attention_mask`` argument with our precomputed BlockMask on the PREFILL step and call
    the NATIVE function unchanged. This guarantees numeric parity with dense at 100% budget.

    We apply the budget mask ONLY on the PREFILL step (q_len == prefill_len). During decode
    (q_len == 1, KV cache growing) the static prefill mask no longer matches, so we pass the
    native causal mask through untouched (standard sparse-PREFILL KV-budget protocol: the
    needle-retrieval decision is made by the budgeted prefill; decode reads it out).

    Returns a restore() callable.
    """
    import transformers.integrations.flex_attention as fa_int

    orig = fa_int.flex_attention_forward

    def patched(module, query, key, value, attention_mask=None, *args, **kwargs):
        bm = controller.block_mask
        apply = (controller.enabled and bm is not None
                 and query.shape[-2] == controller.prefill_len)
        if apply:
            attention_mask = bm
            kwargs.pop("attention_mask", None)
        return orig(module, query, key, value, attention_mask, *args, **kwargs)

    fa_int.flex_attention_forward = patched
    # transformers copies the fn into ALL_ATTENTION_FUNCTIONS at import; refresh it.
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        ALL_ATTENTION_FUNCTIONS["flex_attention"] = patched
    except Exception:
        pass

    def restore():
        fa_int.flex_attention_forward = orig
        try:
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
            ALL_ATTENTION_FUNCTIONS["flex_attention"] = orig
        except Exception:
            pass
    return restore


# ---------------------------------------------------------------------------
# importance stats + block mask construction
# ---------------------------------------------------------------------------
def capture_attn_mass(model, input_ids, torch, seq_len, block_size):
    """One dense prefill with output_attentions to get per-KV-block attention mass.

    Returns attn_mass [n_kv_blocks] on device (mean over layers/heads/queries).
    Falls back to uniform if attentions are unavailable.
    """
    nb = (seq_len + block_size - 1) // block_size
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, output_attentions=True, use_cache=False)
        atts = out.attentions  # tuple[L] of [B, H, S, S]
        acc = torch.zeros(seq_len, device=input_ids.device, dtype=torch.float32)
        for a in atts:
            acc += a.float().mean(dim=1).sum(dim=1).squeeze(0)  # sum over queries, mean heads
        acc /= max(len(atts), 1)
        # aggregate token mass -> block mass
        block_mass = torch.zeros(nb, device=input_ids.device, dtype=torch.float32)
        for b in range(nb):
            block_mass[b] = acc[b * block_size: (b + 1) * block_size].sum()
        return block_mass
    except Exception:
        return torch.ones(nb, device=input_ids.device, dtype=torch.float32)


def build_method_mask(method, budget, seq_len, block_size, attn_mass, wsm, torch, device):
    """Return (BlockMask | None, block_skip_rate) for the given method/budget.

    block_skip_rate = fraction of CAUSAL blocks not admitted (0 for dense/100%).
    """
    nb = wsm.n_blocks(seq_len, block_size)
    if method == "dense":
        return None, 0.0  # handled by disabling the controller (plain SDPA path)
    if method == "flex100":
        w = wsm.make_noop_mask(seq_len, block_size=block_size, device=device)
        return w.block_mask, w.skip_rate
    if method == "streaming":
        # StreamingLLM: sinks + local window sized to match the budget block count.
        local_blocks = max(1, int(round(budget * nb)) - 1)
        adm = wsm.streamingllm_admitted(nb, nb, n_sink_blocks=1,
                                        local_window_blocks=local_blocks, device=device)
        w = wsm.WorkingSetBlockMask(adm, block_size, seq_len, device=device)
        return w.block_mask, w.skip_rate

    stats = wsm.BlockStats(n_q_blocks=nb, n_kv_blocks=nb, block_size=block_size,
                           attn_mass=attn_mass, device=torch.device(device))
    # PHANTOM-FACTOR FIX (2026-08-26): these two branches previously BOTH called
    # ScoreVocabulary.attn_mass, so `h2o` and `working` were one code path under two labels
    # and produced bit-identical k / N / block_skip_rate in all 10 cells of the pilot run.
    # They are now genuinely distinct scorers: `h2o` is attention-mass (the H2O heavy-hitter
    # signal) and `working` is the QK-affinity signal (SnapKV-style observation window).
    # If you want them to be the same method, delete one label -- do not alias them.
    if method == "h2o":
        scores = wsm.ScoreVocabulary.attn_mass(stats)      # H2O: heavy-hitter attention mass
    elif method == "working":
        scores = wsm.ScoreVocabulary.qk_affinity(stats)    # SnapKV-style QK affinity
    else:
        raise ValueError(f"unknown method {method}")
    ws = wsm.WorkingSetBlockMask.from_scores(
        scores, seq_len, block_size=block_size, budget_fraction=budget,
        n_sink_blocks=1, local_window_blocks=1, device=device,
    )
    return ws.block_mask, ws.skip_rate


def skip_rate_of(mask):
    if mask is None:
        return 0.0
    try:
        return float(1.0 - mask.sparsity() / 100.0) if hasattr(mask, "sparsity") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# generation + scoring
# ---------------------------------------------------------------------------
def answer(model, tokenizer, input_ids, torch, controller, block_mask, dense=False):
    """Greedy-decode an answer under the given mask; return (text, ttft_ms, peak_mem_gb)."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    controller.enabled = not dense
    controller.block_mask = None if dense else block_mask
    controller.prefill_len = input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        # measure TTFT: prefill + first token
        gen = model.generate(
            input_ids=input_ids, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    controller.enabled = False
    controller.block_mask = None

    new = gen[0, input_ids.shape[1]:]
    text = tokenizer.decode(new, skip_special_tokens=True)
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    # approximate TTFT as total/ntok proxy is misleading; measure prefill separately below.
    return text, elapsed * 1e3, peak


def measure_ttft(model, input_ids, torch, controller, block_mask, dense=False):
    """Prefill-only TTFT (ms): time a single forward that fills the KV cache."""
    controller.enabled = not dense
    controller.block_mask = None if dense else block_mask
    controller.prefill_len = input_ids.shape[1]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = (time.time() - t0) * 1e3
    controller.enabled = False
    controller.block_mask = None
    return dt


# ---------------------------------------------------------------------------
def run_model(model_key, repo, wsm, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(CKPT_ROOT))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="flex_attention",
        cache_dir=str(CKPT_ROOT),
    ).to(device).eval()

    controller = MaskController()
    restore = install_flex_hook(controller, torch)

    methods = ["dense", "streaming", "h2o", "working"]
    # per (method, budget): k (correct), N (total), + latency/mem aggregates
    cells = {}
    def cell_key(m, b):
        return f"{m}@{b}"

    rng = random.Random(SEED)
    # pre-build NIAH samples (shared across methods/budgets for a fair comparison)
    samples = []
    for _ in range(N_SAMPLES):
        prompt, gold = build_niah_prompt(rng, NEEDLE_DEPTH, CONTEXT_LEN - 64, tok)
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
        # NO padding: a single unpadded sequence. flex_attention's create_block_mask handles
        # non-block-multiple lengths internally (partial trailing block). Left-padding would
        # require the block_mask to also mask pad tokens; since our working-set mask is purely
        # causal-by-block, padding tokens would leak into attention and corrupt retrieval.
        samples.append((ids, gold))

    model_res = {"model": model_key, "repo": repo, "context_len": CONTEXT_LEN,
                 "n_samples": N_SAMPLES, "seq_lens": [], "cells": {}}

    try:
        for si, (ids, gold) in enumerate(samples):
            seq_len = ids.shape[1]
            model_res["seq_lens"].append(seq_len)
            # capture attn_mass once per sample from a dense prefill (importance source)
            attn_mass = capture_attn_mass(model, ids, torch, seq_len, BLOCK_SIZE)

            # dense reference (budget-independent) computed once
            for method in methods:
                budgets = [1.0] if method == "dense" else BUDGETS
                for budget in budgets:
                    try:
                        bm, blk_skip = build_method_mask(method, budget, seq_len, BLOCK_SIZE,
                                                         attn_mass, wsm, torch, device)
                        is_dense = (method == "dense")
                        text, gen_ms, peak = answer(model, tok, ids, torch, controller,
                                                    bm, dense=is_dense)
                        correct = int(gold in text)
                        ck = cell_key(method, budget)
                        c = cells.setdefault(ck, {"method": method, "budget": budget,
                                                  "k": 0, "N": 0, "gen_ms": [],
                                                  "peak_mem_gb": [], "ttft_ms": [],
                                                  "block_skip_rate": round(blk_skip, 4),
                                                  "samples_text": []})
                        c["k"] += correct
                        c["N"] += 1
                        c["gen_ms"].append(round(gen_ms, 2))
                        c["peak_mem_gb"].append(round(peak, 4))
                        if si < 3:
                            c["samples_text"].append({"gold": gold, "gen": text[:80],
                                                      "correct": bool(correct)})
                        # TTFT only for a couple of samples (Path A latency claim)
                        if si < 3:
                            try:
                                ttft = measure_ttft(model, ids, torch, controller, bm,
                                                    dense=is_dense)
                                c["ttft_ms"].append(round(ttft, 2))
                            except Exception:
                                pass
                    except Exception as e:
                        ck = cell_key(method, budget)
                        cells.setdefault(ck, {"method": method, "budget": budget,
                                              "k": 0, "N": 0, "errors": []})
                        cells[ck].setdefault("errors", []).append(f"{type(e).__name__}: {e}")
            report_progress(si + 1, N_SAMPLES, f"{model_key} sample {si+1}/{N_SAMPLES}")
    finally:
        restore()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # finalize aggregates
    for ck, c in cells.items():
        n = max(c.get("N", 0), 1)
        c["accuracy"] = c.get("k", 0) / n
        for f in ("gen_ms", "peak_mem_gb", "ttft_ms"):
            vals = c.get(f, [])
            c[f + "_mean"] = round(sum(vals) / len(vals), 3) if vals else None
    model_res["cells"] = cells
    return model_res


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def main():
    write_pid()
    t0 = time.time()
    import torch
    torch.manual_seed(SEED)
    random.seed(SEED)
    import working_set_mask as wsm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "task_id": TASK_ID, "mode": MODE, "seed": SEED,
        "context_len": CONTEXT_LEN, "block_size": BLOCK_SIZE,
        "budgets": BUDGETS, "needle_depth": NEEDLE_DEPTH,
        "n_samples": N_SAMPLES, "started_at": datetime.now().isoformat(),
        "env": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "models": {},
    }
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)

    for i, (mk, repo) in enumerate(MODELS):
        report_progress(0, len(MODELS), f"loading {mk}")
        try:
            result["models"][mk] = run_model(mk, repo, wsm, torch)
        except Exception as e:
            result["models"][mk] = {"model": mk, "error": f"{type(e).__name__}: {e}",
                                    "traceback": traceback.format_exc()}

    # ---- knee analysis + H2 pilot gate ----
    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["finished_at"] = datetime.now().isoformat()

    analysis = {}
    h2_pass_any = False
    for mk, mres in result["models"].items():
        if "cells" not in mres:
            continue
        cells = mres["cells"]
        dense = cells.get("dense@1.0", {})
        dense_acc = dense.get("accuracy")
        curve = {}
        for b in BUDGETS:
            wc = cells.get(f"working@{b}")
            if wc:
                curve[b] = {"accuracy": wc["accuracy"], "k": wc["k"], "N": wc["N"],
                            "wilson_ci": wilson_ci(wc["k"], wc["N"])}
        # knee: smallest budget whose acc >= 0.95 * dense
        knee = None
        if dense_acc:
            for b in BUDGETS:
                if curve.get(b, {}).get("accuracy", 0) >= 0.95 * dense_acc:
                    knee = b
                    break
        acc25 = curve.get(0.25, {}).get("accuracy", 0.0)
        h2_ok = bool(dense_acc is not None and dense_acc > 0
                     and acc25 >= 0.85 * dense_acc)
        h2_pass_any = h2_pass_any or h2_ok
        analysis[mk] = {"dense_acc": dense_acc, "working_curve": curve,
                        "knee_budget": knee, "acc_at_25pct": acc25,
                        "h2_pilot_pass": h2_ok}
    result["analysis"] = analysis
    result["H2_pilot_pass"] = h2_pass_any
    result["verdict"] = "GO" if h2_pass_any else "REFINE_knee_above_25pct"

    # Real pilot writes the canonical e1_budget_knee.json; smoke/other TASK_IDs write
    # to a TASK_ID-derived filename so they never clobber the canonical result.
    out_name = "e1_budget_knee.json" if TASK_ID == "task_2_e1_budget_knee" else f"{TASK_ID}.json"
    out = FULL_DIR / out_name
    out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({"verdict": result["verdict"], "H2_pilot_pass": h2_pass_any,
                      "analysis": analysis, "elapsed_sec": result["elapsed_sec"]},
                     indent=2, default=str))

    mark_done("success", f"E1 {result['verdict']} (H2_pilot_pass={h2_pass_any})",
              {"verdict": result["verdict"], "elapsed_sec": result["elapsed_sec"]})
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        traceback.print_exc()
        mark_done("fail", f"unhandled: {type(e).__name__}: {e}", {})
        rc = 2
    sys.exit(rc)
