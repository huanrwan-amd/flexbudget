"""task_5_e4_signal_comparison: E4 -- equal-budget, same-kernel score_mod signal comparison (H3).

The headline controlled comparison for cand_flexbudget. At *equal* KV budget (25% and 50%)
and a fixed block size, we swap ONLY the block-importance signal that decides which KV blocks
are admitted into the working set, holding everything else (kernel, budget, anchors, model,
prompts) fixed. The question (H3): does one signal dominate, is the ordering task-dependent,
or are all signals within ~1pp (a publishable null)?

Signals compared (from working_set_mask.ScoreVocabulary + a StreamingLLM anchor-only baseline):
    recency, attn_mass, qk_affinity, combined, streamingllm(anchor-only).
Dense (100%-budget) is included as the ceiling reference.

Tasks (self-contained, hermetic -- NO external dataset download; compute nodes may be offline):
    * niah          -- Needle-in-a-Haystack single-fact retrieval (single-doc QA analogue).
    * multihop      -- two chained needles ("the CODE for X is Y; the KEY for Y is Z"; ask Z),
                       a HotpotQA-style multi-hop analogue that stresses non-local evidence.
    * summ_extract  -- list several labelled facts scattered in filler, ask to reproduce one
                       specific labelled field (summarization/extraction analogue).
Each task builds long contexts from deterministic filler so the needle lands at a controlled
depth and the context spans many KV blocks; accuracy = exact-match of the queried token(s) in
the greedy continuation.

Mechanism (faithful, same kernel path as the E0 gate):
    Pass 1: a dense prefill of the full prompt with output_attentions to collect, per layer,
            the block-level attn_mass and qk_affinity statistics the scorers need.
    Pass 2: for a given (signal, budget), build a per-layer WorkingSetBlockMask from those
            stats and run a budgeted flex_attention forward + greedy decode. We inject the
            precomputed block_mask by patching transformers' flex_attention_forward interface
            (mirrors the eager-capture patch proven in task_1).

PILOT: samples=16 per task, seed=42, timeout 600s, Qwen2.5-0.5B + 1.5B. Records raw (k,N) per
(signal x task x budget x model) and Wilson 95% CIs. Writes exp/results/pilots/ +
pilot_summary.{md,json}; also drops a copy of the structured result in exp/results/full/.
Defensive: every leg is wrapped; a DONE marker is always emitted.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

TASK_ID = os.environ.get("TASK_ID", "task_5_e4_signal_comparison")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "exp/results"))
FULL_DIR = RESULTS_DIR / "full"
PILOT_DIR = RESULTS_DIR / "pilots"
for d in (RESULTS_DIR, FULL_DIR, PILOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_fix  # noqa: E402  (neutralize broken torchvision/torchaudio before transformers)

SEED = 42
MODE = os.environ.get("EXP_MODE", "PILOT").upper()
PILOT_SAMPLES = int(os.environ.get("PILOT_SAMPLES", "16"))
BUDGETS = (0.25, 0.50)
BLOCK_SIZE = 128
SEQ_LEN = 2048            # context length for pilot: multiple of block size, spans 16 blocks
N_SINK_BLOCKS = 1
LOCAL_WINDOW_BLOCKS = 1
MODELS = {
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}
SIGNALS = ["recency", "attn_mass", "qk_affinity", "combined", "streamingllm", "dense"]
TASKS = ["niah", "multihop", "summ_extract"]


# --------------------------------------------------------------------------- markers
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


# --------------------------------------------------------------------------- stats
def wilson_ci(k, n, z=1.96):
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# --------------------------------------------------------------------------- task battery
_WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
          "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
          "xray yankee zulu north south east west river mountain valley forest desert "
          "ocean meadow canyon glacier harbor island").split()


def _filler(rng, n_words):
    return " ".join(rng.choice(_WORDS) for _ in range(n_words))


def _make_niah(rng, idx):
    """Single needle: 'The secret code is NNNN.' buried in filler; ask for the code."""
    code = f"{rng.randint(1000, 9999)}"
    depth = rng.uniform(0.15, 0.85)
    pre = _filler(rng, int(650 * depth) + 20)
    post = _filler(rng, int(650 * (1 - depth)) + 20)
    ctx = (f"{pre} The secret code is {code}. {post}")
    prompt = (f"Read the passage and answer with only the number.\n\n{ctx}\n\n"
              f"Question: What is the secret code? Answer:")
    return prompt, code


def _make_multihop(rng, idx):
    """Two chained facts: token for X is Y; value for Y is Z. Ask value for X (needs both)."""
    y = "".join(rng.choice("ABCDEFGHJKLMNP") for _ in range(3))
    z = f"{rng.randint(1000, 9999)}"
    d1 = rng.uniform(0.1, 0.4)
    d2 = rng.uniform(0.55, 0.9)
    seg0 = _filler(rng, int(500 * d1) + 15)
    fact1 = f"The token for the vault is {y}."
    seg1 = _filler(rng, int(500 * (d2 - d1)) + 15)
    fact2 = f"The value for token {y} is {z}."
    seg2 = _filler(rng, int(500 * (1 - d2)) + 15)
    ctx = f"{seg0} {fact1} {seg1} {fact2} {seg2}"
    prompt = (f"Read the passage and answer with only the number.\n\n{ctx}\n\n"
              f"Question: What is the value for the vault? Answer:")
    return prompt, z


def _make_summ_extract(rng, idx):
    """Several labelled fields scattered in filler; ask to reproduce one specific field."""
    labels = ["ORDER", "INVOICE", "TICKET", "BATCH"]
    rng.shuffle(labels)
    vals = {lab: f"{rng.randint(10000, 99999)}" for lab in labels}
    target = rng.choice(labels)
    parts = []
    for lab in labels:
        parts.append(_filler(rng, rng.randint(80, 160)))
        parts.append(f"The {lab} number is {vals[lab]}.")
    parts.append(_filler(rng, 60))
    ctx = " ".join(parts)
    prompt = (f"Read the passage and answer with only the number.\n\n{ctx}\n\n"
              f"Question: What is the {target} number? Answer:")
    return prompt, vals[target]


TASK_BUILDERS = {"niah": _make_niah, "multihop": _make_multihop, "summ_extract": _make_summ_extract}


# --------------------------------------------------------------------------- eager injection
#
# We route the whole experiment through the EAGER attention interface for two reasons:
#   (1) flex_attention does NOT materialize attention weights, so output_attentions=True is a
#       no-op there -- we need real per-layer weights to compute attn_mass / qk_affinity stats.
#   (2) A single eager patch can serve BOTH roles: capture stats on a dense pass, AND apply a
#       block-level admission mask (additive -inf bias) on a budgeted pass. The admitted-block
#       SET is byte-identical to what the flex block_mask would enforce (E0 already proved the
#       flex kernel is bit-exact to dense on admitted rows), so eager masking faithfully answers
#       the E4 question: "which importance SIGNAL picks the better KV blocks at equal budget?"
class _EagerController:
    """Mutable controller consumed by the patched eager attention, once per layer per forward.

    mode == "capture": record block-level attn_mass + qk_affinity per layer (dense pass).
    mode == "mask":    add token_bias[layer] (-inf on non-admitted (q,k) pairs) to the logits.
    mode == "off":     plain dense attention (used for decode steps).
    """
    def __init__(self):
        self.mode = "off"
        self.captured = []       # list[dict(attn_mass, qk_affinity)] built during capture
        self.token_bias = None   # list[[S,S]] additive bias per layer during mask mode
        self._i = 0

    def reset(self, mode, token_bias=None):
        self.mode = mode
        self.token_bias = token_bias
        self._i = 0
        if mode == "capture":
            self.captured = []


def install_eager_patch(ctrl, wsm, torch):
    """Patch the qwen2 module-level eager_attention_forward (same point task_1 used).

    Returns restore(). The patched fn (a) records per-layer block stats in "capture" mode and
    (b) injects a per-layer additive admission bias (-inf on non-admitted (q,k)) in "mask" mode.
    """
    import torch.nn.functional as F
    import transformers.models.qwen2.modeling_qwen2 as qwen_mod

    orig = qwen_mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        q, k, v = query, key, value
        if k.shape[1] != q.shape[1]:  # GQA expand kv heads to query heads
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        sm = scaling if scaling is not None else (1.0 / math.sqrt(q.shape[-1]))
        Sq = q.shape[-2]
        Sk = k.shape[-2]
        logits = torch.matmul(q, k.transpose(-1, -2)) * sm  # [B,H,Sq,Sk]
        # causal mask (query index i attends to key j<=i within the shared length)
        qi = torch.arange(Sq, device=q.device).unsqueeze(1)
        kj = torch.arange(Sk, device=q.device).unsqueeze(0)
        causal = kj > (qi + (Sk - Sq))  # allow for KV longer than Q (not used here, use_cache=False)
        logits = logits.masked_fill(causal, float("-inf"))
        if ctrl.mode == "mask" and ctrl.token_bias is not None and ctrl._i < len(ctrl.token_bias):
            bias = ctrl.token_bias[ctrl._i]  # [Sq,Sk]
            if bias.shape == logits.shape[-2:]:
                logits = logits + bias
        probs = F.softmax(logits.float(), dim=-1).to(v.dtype)
        if ctrl.mode == "capture":
            bs = BLOCK_SIZE
            nq = wsm.n_blocks(Sq, bs)
            a = probs[0].float().mean(0)  # [Sq,Sk] head-avg
            padq = nq * bs - a.shape[0]
            padk = nq * bs - a.shape[1]
            if padq > 0 or padk > 0:
                a = F.pad(a, (0, max(0, padk), 0, max(0, padq)))
            ab = a.reshape(nq, bs, nq, bs).mean(dim=(1, 3))  # [nq,nq]
            ctrl.captured.append({"attn_mass": ab.sum(0).to(q.device),
                                  "qk_affinity": ab.to(q.device), "S": Sq, "nq": nq})
        out = torch.matmul(probs, v)             # [B,H,Sq,D]
        out = out.transpose(1, 2).contiguous()   # [B,Sq,H,D] like the interface returns
        ctrl._i += 1
        return out, probs

    qwen_mod.eager_attention_forward = patched

    def restore():
        qwen_mod.eager_attention_forward = orig

    return restore


# --------------------------------------------------------------------------- per-model run
def admitted_for(signal, budget, st, seq_len, device, wsm, torch):
    """Return a [nq,nq] bool admitted-block map for one layer's stats under signal+budget."""
    nq = wsm.n_blocks(seq_len, BLOCK_SIZE)
    if signal == "dense":
        return wsm.dense_admitted(nq, nq, device=device)
    if signal == "streamingllm":
        local_w = max(1, int(round(budget * nq)) - N_SINK_BLOCKS)
        return wsm.streamingllm_admitted(nq, nq, n_sink_blocks=N_SINK_BLOCKS,
                                         local_window_blocks=local_w, device=device)
    stats = wsm.BlockStats(
        n_q_blocks=nq, n_kv_blocks=nq, block_size=BLOCK_SIZE,
        attn_mass=st.get("attn_mass"), qk_affinity=st.get("qk_affinity"),
        device=torch.device(device),
    )
    scores = wsm.SCORERS[signal](stats)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0).expand(nq, -1).contiguous()
    return wsm.build_block_importance(
        scores, budget_fraction=budget, n_sink_blocks=N_SINK_BLOCKS,
        local_window_blocks=LOCAL_WINDOW_BLOCKS,
    )


def build_token_bias(signal, budget, per_layer_stats, seq_len, device, wsm, torch):
    """Per-layer additive token-level bias: 0 where admitted, -inf where masked out."""
    nq = wsm.n_blocks(seq_len, BLOCK_SIZE)
    bias = []
    NEG = torch.finfo(torch.float32).min
    for st in per_layer_stats:
        adm = admitted_for(signal, budget, st, seq_len, device, wsm, torch)  # [nq,nq] bool
        # expand block map -> token map [S,S]
        tok_adm = adm.repeat_interleave(BLOCK_SIZE, 0).repeat_interleave(BLOCK_SIZE, 1)
        tok_adm = tok_adm[:seq_len, :seq_len]
        b = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
        b = b.masked_fill(~tok_adm, NEG)
        bias.append(b)
    return bias


def greedy_answer(model, tok, input_ids, ctrl, token_bias, max_new=8, device="cuda"):
    """Greedy-decode a short answer; token_bias (budgeted admission) applied to the prefill."""
    import torch
    gen = []
    cur = input_ids
    for t in range(max_new):
        if t == 0:
            ctrl.reset("mask", token_bias)   # budgeted prefill: selection under test
        else:
            ctrl.reset("off")                # dense short decode
        with torch.no_grad():
            out = model(input_ids=cur, use_cache=False)
        nxt = int(out.logits[0, -1].argmax().item())
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        txt = tok.decode(gen)
        if any(c in txt for c in (".", "\n")) and t >= 1:
            break
    return tok.decode(gen)


def run_model(model_key, repo, device, wsm, torch, ckpt_root, budget_signals):
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(ckpt_root))
    model = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.bfloat16, attn_implementation="eager",
        cache_dir=str(ckpt_root),
    ).to(device).eval()

    ctrl = _EagerController()
    restore = install_eager_patch(ctrl, wsm, torch)

    counts = {}
    examples = []

    try:
        for task in TASKS:
            builder = TASK_BUILDERS[task]
            rng = random.Random(SEED + hash(task) % 1000)
            for i in range(PILOT_SAMPLES):
                prompt, gold = builder(rng, i)
                enc = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ_LEN)
                ids = enc["input_ids"].to(device)
                if ids.shape[1] < SEQ_LEN:
                    pad_id = tok.pad_token_id or tok.eos_token_id
                    pad = torch.full((1, SEQ_LEN - ids.shape[1]), pad_id, device=device)
                    ids = torch.cat([pad, ids], dim=1)  # left-pad; needle stays in-context
                else:
                    ids = ids[:, -SEQ_LEN:]
                seq_len = ids.shape[1]

                # Pass 1: dense capture of per-layer block stats (shared across signals/budgets)
                ctrl.reset("capture")
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                stats = ctrl.captured

                for signal in SIGNALS:
                    budgets = [1.0] if signal == "dense" else list(BUDGETS)
                    for budget in budgets:
                        bias = build_token_bias(signal, budget, stats, seq_len,
                                                device, wsm, torch)
                        ans = greedy_answer(model, tok, ids, ctrl, bias, device=device)
                        ok = gold in ans.replace(" ", "")
                        key = f"{signal}|{task}|{budget}"
                        c = counts.setdefault(key, [0, 0])
                        c[0] += int(ok)
                        c[1] += 1
                        if i < 2:
                            examples.append({"signal": signal, "task": task,
                                             "budget": budget, "gold": gold,
                                             "answer": ans.strip()[:40], "correct": ok})
                report_progress(1, 1, f"{model_key} {task} {i+1}/{PILOT_SAMPLES}")
    finally:
        restore()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # assemble per (signal,task,budget) with Wilson CI
    rows = []
    for key, (k, n) in sorted(counts.items()):
        signal, task, budget = key.split("|")
        lo, hi = wilson_ci(k, n)
        rows.append({
            "model": model_key, "signal": signal, "task": task,
            "budget": float(budget), "k": k, "n": n,
            "accuracy": round(k / n, 4) if n else 0.0,
            "wilson_ci": [round(lo, 4), round(hi, 4)],
        })
    return rows, examples


# --------------------------------------------------------------------------- analysis
def summarize(all_rows):
    """Per (task,budget): rank signals; report separation between best/worst non-dense signal."""
    from collections import defaultdict
    grp = defaultdict(list)
    for r in all_rows:
        if r["signal"] == "dense":
            continue
        grp[(r["model"], r["task"], r["budget"])].append(r)
    findings = []
    for (model, task, budget), rows in sorted(grp.items()):
        rows = sorted(rows, key=lambda x: -x["accuracy"])
        best, worst = rows[0], rows[-1]
        sep = round(best["accuracy"] - worst["accuracy"], 4)
        findings.append({
            "model": model, "task": task, "budget": budget,
            "best_signal": best["signal"], "best_acc": best["accuracy"],
            "worst_signal": worst["signal"], "worst_acc": worst["accuracy"],
            "separation_pp": round(sep * 100, 2),
            "ordering": [r["signal"] for r in rows],
            "signal_agnostic_within_1pp": sep <= 0.01,
        })
    return findings


# --------------------------------------------------------------------------- main
def main():
    write_pid()
    t0 = time.time()
    import torch
    torch.manual_seed(SEED)
    import working_set_mask as wsm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_root = Path("/work/shared/checkpoints")
    ckpt_root.mkdir(parents=True, exist_ok=True)

    result = {
        "task_id": TASK_ID, "mode": MODE, "seed": SEED,
        "pilot_samples": PILOT_SAMPLES, "budgets": list(BUDGETS),
        "block_size": BLOCK_SIZE, "seq_len": SEQ_LEN, "signals": SIGNALS, "tasks": TASKS,
        "started_at": datetime.now().isoformat(),
        "env": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "rows": [], "examples": [], "errors": {},
    }

    for i, (mk, repo) in enumerate(MODELS.items()):
        report_progress(i + 1, len(MODELS), f"model {mk}")
        try:
            rows, examples = run_model(mk, repo, device, wsm, torch, ckpt_root, None)
            result["rows"].extend(rows)
            result["examples"].extend(examples[:12])
        except Exception as e:
            result["errors"][mk] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    result["findings"] = summarize(result["rows"])
    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["finished_at"] = datetime.now().isoformat()

    # pipeline-health gate: did we produce comparable per-signal (k,N) for every signal?
    n_signals_seen = len({r["signal"] for r in result["rows"] if r["signal"] != "dense"})
    pipeline_ok = (len(result["rows"]) > 0 and n_signals_seen >= 4)
    any_sep = any(f["separation_pp"] > 1.0 for f in result["findings"])
    all_null = (len(result["findings"]) > 0 and
                all(f["signal_agnostic_within_1pp"] for f in result["findings"]))
    result["pipeline_ok"] = pipeline_ok
    result["verdict"] = "GO" if pipeline_ok else "NO_GO"
    result["h3_pilot_signal_ordering_emerges"] = any_sep
    result["h3_pilot_signal_agnostic_null"] = all_null

    (FULL_DIR / "e4_signal_comparison.json").write_text(json.dumps(result, indent=2))
    (PILOT_DIR / f"{TASK_ID}_pilot.json").write_text(json.dumps(result, indent=2))

    print(json.dumps({
        "verdict": result["verdict"], "pipeline_ok": pipeline_ok,
        "n_rows": len(result["rows"]), "signals_seen": n_signals_seen,
        "ordering_emerges": any_sep, "signal_agnostic_null": all_null,
        "errors": list(result["errors"].keys()), "elapsed_sec": result["elapsed_sec"],
    }, indent=2))

    mark_done("success" if pipeline_ok else "fail",
              f"E4 pilot {result['verdict']}: {len(result['rows'])} (signal,task,budget) rows, "
              f"{n_signals_seen} signals; ordering_emerges={any_sep} null={all_null}",
              {"pipeline_ok": pipeline_ok, "elapsed_sec": result["elapsed_sec"]})
    return 0 if pipeline_ok else 1


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as e:
        traceback.print_exc()
        mark_done("fail", f"unhandled: {type(e).__name__}: {e}", {})
        rc = 2
    sys.exit(rc)
