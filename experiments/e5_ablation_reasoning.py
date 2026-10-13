"""task_6_e5_ablation_reasoning: E5 -- anchor/block-size ablation + reasoning probe (H7).

Two parts, both training-free / inference-only, single-GPU (MI300X gfx942/ROCm), bf16.

Part (a) ABLATIONS (Qwen2.5-0.5B, 25% budget, attn_mass signal):
    * Anchor set ON vs OFF: with anchors = sinks (n_sink_blocks=1) + local window
      (local_window_blocks=1); OFF = pure top-B admission (no sinks, no local window).
    * Block size {64, 128, 256}: same 25% budget, same signal, different page size.
    Reports accuracy + block skip rate per ablation leg (deltas vs the default
    anchors-on / block-128 configuration).

Part (b) REASONING PROBE (H7, Qwen2.5-0.5B):
    * GSM8K-long: few-shot GSM8K-style arithmetic word problems padded to a long context
      with deterministic filler, so the reasoning target must survive KV budgeting. Compare
      attn_mass working-set (25% budget) vs dense.
    * value-norm concentration diagnostic: at prefill we capture per-KV-block mean value-vector
      L2 norm; concentration = normalized Gini-like top-mass ratio (fraction of total value
      norm carried by the top-25% of blocks). H7: reasoning-specific degradation is expected to
      correlate with high value-norm concentration (importance scoring silently drops
      value-heavy but low-attention blocks). A NULL (no degradation) is also publishable ==
      importance scoring is robust for reasoning.

Mechanism: identical eager-injection harness proven in task_5 (E4). Pass 1 = dense capture of
per-layer block stats (attn_mass, qk_affinity, value_norm). Pass 2 = budgeted admission via an
additive -inf token bias for the exact admitted-block set the flex block_mask would enforce
(E0 proved flex is bit-exact to dense on admitted rows). This isolates the E5 question --
what does the anchor set / block size / reasoning load do to working-set accuracy -- from
kernel quality.

PILOT: samples=16 (ablation) / 16 (reasoning), seed=42, timeout 600s. Records raw (k,N) for
every leg + a numeric value-norm concentration score. Writes exp/results/full/
e5_ablation_reasoning.json + exp/results/pilots/. Defensive: every leg wrapped; DONE always
emitted. SEQ_LEN kept modest for the pilot (2048) to fit the pilot budget; FULL mode uses 8K.
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

TASK_ID = os.environ.get("TASK_ID", "task_6_e5_ablation_reasoning")
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "exp/results"))
FULL_DIR = RESULTS_DIR / "full"
PILOT_DIR = RESULTS_DIR / "pilots"
for d in (RESULTS_DIR, FULL_DIR, PILOT_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env_fix  # noqa: E402  (neutralize broken torchvision before transformers)

SEED = 42
MODE = os.environ.get("EXP_MODE", "PILOT").upper()
PILOT_SAMPLES = int(os.environ.get("PILOT_SAMPLES", "16"))
# pilot: 2048 tokens (16 blocks @128); full: 8192.
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048" if MODE == "PILOT" else "8192"))
BUDGET = 0.25
DEFAULT_BLOCK_SIZE = 128
BLOCK_SIZES = [64, 128, 256]
MODEL_KEY = "qwen2.5-0.5b"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"


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
    """Single needle used by the ablation legs (same as E4 niah)."""
    code = f"{rng.randint(1000, 9999)}"
    depth = rng.uniform(0.15, 0.85)
    pre = _filler(rng, int(650 * depth) + 20)
    post = _filler(rng, int(650 * (1 - depth)) + 20)
    ctx = (f"{pre} The secret code is {code}. {post}")
    prompt = (f"Read the passage and answer with only the number.\n\n{ctx}\n\n"
              f"Question: What is the secret code? Answer:")
    return prompt, code


# --- GSM8K-long: deterministic 1-2 step arithmetic word problems padded to long context ---
_NAMES = "Ava Ben Cara Dan Eva Finn Gia Hal Ivy Jae Kim Leo Mae Ned".split()
_ITEMS = "apples pencils marbles coins stickers cards ribbons stamps".split()


def _make_gsm8k_long(rng, idx):
    """A short arithmetic word problem in a long context; answer is a small integer.

    Single-step form: 'X has A. Buys B more. How many now?' -> A+B (small numbers), which a
    0.5B model can solve DENSE -> H7 gets a real (non-degenerate) signal. Worked few-shot
    exemplars establish the pattern. The context is padded to long length with deterministic
    filler placed BEFORE the exemplars (so the reasoning target survives near the tail but the
    KV cache still spans many blocks); working-set budgeting must retain the exemplars + query
    blocks. This isolates "does budgeting drop the reasoning-relevant blocks?" from prompt
    unsolvability.
    """
    def one(rng):
        name = rng.choice(_NAMES)
        item = rng.choice(_ITEMS)
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        ans = a + b
        q = f"{name} has {a} {item} and buys {b} more {item}. How many {item} does {name} have?"
        return q, ans

    # three worked few-shot exemplars, then the query -- kept contiguous with "Answer:".
    shots = []
    for _ in range(3):
        q, ans = one(rng)
        shots.append(f"Question: {q}\nAnswer: {ans}")
    shot_txt = "\n\n".join(shots)
    q, ans = one(rng)
    # long-context padding placed BEFORE the reasoning block so budgeting must skip filler
    # blocks while retaining exemplar+query blocks; question stays adjacent to the answer cue.
    pad = _filler(rng, 900)
    prompt = (f"Solve the arithmetic word problems. Answer with only the final number.\n\n"
              f"{pad}\n\n{shot_txt}\n\nQuestion: {q}\nAnswer:")
    return prompt, str(ans)


# --------------------------------------------------------------------------- eager injection
class _EagerController:
    """mode capture: record per-layer attn_mass, qk_affinity, value_norm (dense pass).
    mode mask: add token_bias[layer] (-inf on non-admitted (q,k)) to logits.
    mode off: plain dense attention (decode steps)."""
    def __init__(self):
        self.mode = "off"
        self.captured = []
        self.token_bias = None
        self.block_size = DEFAULT_BLOCK_SIZE
        self._i = 0

    def reset(self, mode, token_bias=None, block_size=None):
        self.mode = mode
        self.token_bias = token_bias
        if block_size is not None:
            self.block_size = block_size
        self._i = 0
        if mode == "capture":
            self.captured = []


def install_eager_patch(ctrl, wsm, torch):
    import torch.nn.functional as F
    import transformers.models.qwen2.modeling_qwen2 as qwen_mod

    orig = qwen_mod.eager_attention_forward

    def patched(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        q, k, v = query, key, value
        if k.shape[1] != q.shape[1]:  # GQA expand
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        sm = scaling if scaling is not None else (1.0 / math.sqrt(q.shape[-1]))
        Sq = q.shape[-2]
        Sk = k.shape[-2]
        logits = torch.matmul(q, k.transpose(-1, -2)) * sm
        qi = torch.arange(Sq, device=q.device).unsqueeze(1)
        kj = torch.arange(Sk, device=q.device).unsqueeze(0)
        causal = kj > (qi + (Sk - Sq))
        logits = logits.masked_fill(causal, float("-inf"))
        if ctrl.mode == "mask" and ctrl.token_bias is not None and ctrl._i < len(ctrl.token_bias):
            bias = ctrl.token_bias[ctrl._i]
            if bias.shape == logits.shape[-2:]:
                logits = logits + bias
        probs = F.softmax(logits.float(), dim=-1).to(v.dtype)
        if ctrl.mode == "capture":
            bs = ctrl.block_size
            nq = wsm.n_blocks(Sq, bs)
            a = probs[0].float().mean(0)  # [Sq,Sk] head-avg
            padq = nq * bs - a.shape[0]
            padk = nq * bs - a.shape[1]
            if padq > 0 or padk > 0:
                a = F.pad(a, (0, max(0, padk), 0, max(0, padq)))
            ab = a.reshape(nq, bs, nq, bs).mean(dim=(1, 3))  # [nq,nq]
            # value-norm per KV block: mean L2 norm of value vectors, head-averaged.
            vn_tok = v[0].float().norm(dim=-1).mean(0)  # [Sk] head-avg per-token value L2
            padv = nq * bs - vn_tok.shape[0]
            if padv > 0:
                vn_tok = F.pad(vn_tok, (0, padv))
            vn_block = vn_tok.reshape(nq, bs).mean(dim=1)  # [nq]
            ctrl.captured.append({
                "attn_mass": ab.sum(0).to(q.device),
                "qk_affinity": ab.to(q.device),
                "value_norm": vn_block.to(q.device),
                "S": Sq, "nq": nq,
            })
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous()
        ctrl._i += 1
        return out, probs

    qwen_mod.eager_attention_forward = patched

    def restore():
        qwen_mod.eager_attention_forward = orig

    return restore


# --------------------------------------------------------------------------- admission
def admitted_for_ablation(st, seq_len, block_size, anchors_on, device, wsm, torch):
    """attn_mass admission at 25% budget with anchors toggled, at a given block size."""
    nq = wsm.n_blocks(seq_len, block_size)
    n_sink = 1 if anchors_on else 0
    local_w = 1 if anchors_on else 0
    stats = wsm.BlockStats(
        n_q_blocks=nq, n_kv_blocks=nq, block_size=block_size,
        attn_mass=st.get("attn_mass"), qk_affinity=st.get("qk_affinity"),
        value_norm=st.get("value_norm"), device=torch.device(device),
    )
    scores = wsm.SCORERS["attn_mass"](stats)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0).expand(nq, -1).contiguous()
    return wsm.build_block_importance(
        scores, budget_fraction=BUDGET, n_sink_blocks=n_sink,
        local_window_blocks=local_w,
    )


def admitted_dense(seq_len, block_size, device, wsm, torch):
    nq = wsm.n_blocks(seq_len, block_size)
    return wsm.dense_admitted(nq, nq, device=device)


def skip_rate_of(admitted, device, torch):
    nq, nk = admitted.shape
    qi = torch.arange(nq, device=device).unsqueeze(1)
    kj = torch.arange(nk, device=device).unsqueeze(0)
    causal = kj <= qi
    n_causal = causal.sum().item()
    n_adm = (admitted & causal).sum().item()
    return 0.0 if n_causal == 0 else 1.0 - (n_adm / n_causal)


def block_map_to_token_bias(admitted, block_size, seq_len, device, torch):
    tok_adm = admitted.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    tok_adm = tok_adm[:seq_len, :seq_len]
    NEG = torch.finfo(torch.float32).min
    b = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    return b.masked_fill(~tok_adm, NEG)


def value_norm_concentration(per_layer_stats, top_frac=0.25):
    """Fraction of total value-norm mass carried by the top-`top_frac` KV blocks,
    averaged across layers. 1.0 == maximally concentrated (single block), top_frac == uniform.
    """
    import torch
    vals = []
    for st in per_layer_stats:
        vn = st.get("value_norm")
        if vn is None or vn.numel() == 0:
            continue
        vn = vn.float().abs()
        total = vn.sum().item()
        if total <= 0:
            continue
        n = vn.numel()
        k = max(1, int(round(top_frac * n)))
        top = torch.topk(vn, k).values.sum().item()
        vals.append(top / total)
    if not vals:
        return None
    return float(sum(vals) / len(vals))


# --------------------------------------------------------------------------- decode
def greedy_answer(model, tok, ids, ctrl, token_bias, block_size, max_new=8, device="cuda"):
    import torch
    gen = []
    cur = ids
    for t in range(max_new):
        if t == 0 and token_bias is not None:
            ctrl.reset("mask", token_bias, block_size=block_size)
        else:
            ctrl.reset("off")
        with torch.no_grad():
            out = model(input_ids=cur, use_cache=False)
        nxt = int(out.logits[0, -1].argmax().item())
        gen.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        txt = tok.decode(gen)
        if any(c in txt for c in (".", "\n")) and t >= 1:
            break
    return tok.decode(gen)


def encode_padded(tok, prompt, seq_len, device, torch):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=seq_len)
    ids = enc["input_ids"].to(device)
    if ids.shape[1] < seq_len:
        pad_id = tok.pad_token_id or tok.eos_token_id
        pad = torch.full((1, seq_len - ids.shape[1]), pad_id, device=device)
        ids = torch.cat([pad, ids], dim=1)  # left-pad; needle stays in-context
    else:
        ids = ids[:, -seq_len:]
    return ids


# --------------------------------------------------------------------------- main run
def run(device, wsm, torch, ckpt_root):
    import random
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=str(ckpt_root))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO, torch_dtype=torch.bfloat16, attn_implementation="eager",
        cache_dir=str(ckpt_root),
    ).to(device).eval()

    ctrl = _EagerController()
    restore = install_eager_patch(ctrl, wsm, torch)

    ablation_counts = {}          # leg -> [k, n]
    ablation_skip = {}            # leg -> list of skip rates (avg later)
    reasoning_counts = {}         # method -> [k, n]
    vn_conc_samples = []          # per-sample value-norm concentration (dense capture)
    examples = []

    try:
        # ---------- Part (a): ablations on NIAH ----------
        rng = random.Random(SEED)
        ablation_legs = (
            [("anchors_on_bs128", 128, True), ("anchors_off_bs128", 128, False)]
            + [(f"bs{bs}", bs, True) for bs in BLOCK_SIZES if bs != 128]
        )
        # bs128 anchors_on is the shared default; also serves as the "bs128" block-size point.
        for i in range(PILOT_SAMPLES):
            prompt, gold = _make_niah(rng, i)
            for leg_name, bs, anchors_on in ablation_legs:
                ids = encode_padded(tok, prompt, SEQ_LEN, device, torch)
                seq_len = ids.shape[1]
                # capture per-layer stats at THIS block size (stats are block-size dependent)
                ctrl.reset("capture", block_size=bs)
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False)
                stats = ctrl.captured
                per_layer_bias = []
                skips = []
                for st in stats:
                    adm = admitted_for_ablation(st, seq_len, bs, anchors_on, device, wsm, torch)
                    skips.append(skip_rate_of(adm, device, torch))
                    per_layer_bias.append(
                        block_map_to_token_bias(adm, bs, seq_len, device, torch))
                ans = greedy_answer(model, tok, ids, ctrl, per_layer_bias, bs, device=device)
                ok = gold in ans.replace(" ", "")
                c = ablation_counts.setdefault(leg_name, [0, 0])
                c[0] += int(ok)
                c[1] += 1
                ablation_skip.setdefault(leg_name, []).append(
                    sum(skips) / len(skips) if skips else 0.0)
                if i < 2:
                    examples.append({"part": "ablation", "leg": leg_name, "gold": gold,
                                     "answer": ans.strip()[:40], "correct": ok})
            report_progress(1, 2, f"ablation {i+1}/{PILOT_SAMPLES}")

        # ---------- Part (b): reasoning probe (GSM8K-long) ----------
        rng = random.Random(SEED + 7)
        for i in range(PILOT_SAMPLES):
            prompt, gold = _make_gsm8k_long(rng, i)
            ids = encode_padded(tok, prompt, SEQ_LEN, device, torch)
            seq_len = ids.shape[1]
            # dense capture (also gives value_norm for the concentration diagnostic)
            ctrl.reset("capture", block_size=DEFAULT_BLOCK_SIZE)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
            stats = ctrl.captured
            conc = value_norm_concentration(stats)
            if conc is not None:
                vn_conc_samples.append(conc)

            # dense answer
            ans_dense = greedy_answer(model, tok, ids, ctrl, None, DEFAULT_BLOCK_SIZE,
                                      device=device)
            ok_d = gold in ans_dense.replace(" ", "")
            reasoning_counts.setdefault("dense", [0, 0])
            reasoning_counts["dense"][0] += int(ok_d)
            reasoning_counts["dense"][1] += 1

            # attn_mass working-set @25%
            per_layer_bias = []
            for st in stats:
                adm = admitted_for_ablation(st, seq_len, DEFAULT_BLOCK_SIZE, True,
                                            device, wsm, torch)
                per_layer_bias.append(
                    block_map_to_token_bias(adm, DEFAULT_BLOCK_SIZE, seq_len, device, torch))
            ans_ws = greedy_answer(model, tok, ids, ctrl, per_layer_bias, DEFAULT_BLOCK_SIZE,
                                   device=device)
            ok_w = gold in ans_ws.replace(" ", "")
            reasoning_counts.setdefault("attn_mass_ws25", [0, 0])
            reasoning_counts["attn_mass_ws25"][0] += int(ok_w)
            reasoning_counts["attn_mass_ws25"][1] += 1

            if i < 3:
                examples.append({"part": "reasoning", "gold": gold,
                                 "dense": ans_dense.strip()[:30], "ws25": ans_ws.strip()[:30],
                                 "dense_ok": ok_d, "ws_ok": ok_w})
            report_progress(2, 2, f"reasoning {i+1}/{PILOT_SAMPLES}")
    finally:
        restore()
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    return ablation_counts, ablation_skip, reasoning_counts, vn_conc_samples, examples


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
        "pilot_samples": PILOT_SAMPLES, "budget": BUDGET,
        "default_block_size": DEFAULT_BLOCK_SIZE, "block_sizes": BLOCK_SIZES,
        "seq_len": SEQ_LEN, "model": MODEL_KEY,
        "started_at": datetime.now().isoformat(),
        "env": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "ablation": [], "reasoning": [], "examples": [], "errors": {},
    }

    try:
        (abl_counts, abl_skip, reas_counts,
         vn_conc, examples) = run(device, wsm, torch, ckpt_root)

        for leg, (k, n) in sorted(abl_counts.items()):
            lo, hi = wilson_ci(k, n)
            skips = abl_skip.get(leg, [])
            result["ablation"].append({
                "leg": leg, "k": k, "n": n,
                "accuracy": round(k / n, 4) if n else 0.0,
                "wilson_ci": [round(lo, 4), round(hi, 4)],
                "mean_block_skip_rate": round(sum(skips) / len(skips), 4) if skips else 0.0,
            })
        for method, (k, n) in sorted(reas_counts.items()):
            lo, hi = wilson_ci(k, n)
            result["reasoning"].append({
                "method": method, "k": k, "n": n,
                "accuracy": round(k / n, 4) if n else 0.0,
                "wilson_ci": [round(lo, 4), round(hi, 4)],
            })
        result["value_norm_concentration"] = (
            round(sum(vn_conc) / len(vn_conc), 4) if vn_conc else None)
        result["value_norm_concentration_n"] = len(vn_conc)
        result["examples"] = examples[:12]
    except Exception as e:
        result["errors"]["run"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["finished_at"] = datetime.now().isoformat()

    # pipeline-health gate: both ablation legs + GSM8K probe produced (k,N) and the
    # value-norm diagnostic emitted a numeric concentration score without error.
    n_abl_legs = len(result["ablation"])
    n_reas = len(result["reasoning"])
    vn_ok = result.get("value_norm_concentration") is not None
    pipeline_ok = (n_abl_legs >= 4 and n_reas >= 2 and vn_ok and not result["errors"])
    result["pipeline_ok"] = pipeline_ok
    result["verdict"] = "GO" if pipeline_ok else "NO_GO"

    # H7 read: reasoning-specific degradation (dense - ws) and whether it co-occurs with
    # high value-norm concentration (both are publishable; this is a directional read only).
    reas_by = {r["method"]: r["accuracy"] for r in result["reasoning"]}
    if "dense" in reas_by and "attn_mass_ws25" in reas_by:
        deg = round(reas_by["dense"] - reas_by["attn_mass_ws25"], 4)
        result["h7_reasoning_degradation_pp"] = round(deg * 100, 2)
        result["h7_reasoning_robust_null"] = (deg <= 0.05)  # <=5pp drop -> robust
    # anchor-effect and block-size-effect deltas (interpretation)
    abl_by = {a["leg"]: a["accuracy"] for a in result["ablation"]}
    if "anchors_on_bs128" in abl_by and "anchors_off_bs128" in abl_by:
        result["anchor_effect_pp"] = round(
            (abl_by["anchors_on_bs128"] - abl_by["anchors_off_bs128"]) * 100, 2)

    (FULL_DIR / "e5_ablation_reasoning.json").write_text(json.dumps(result, indent=2))
    (PILOT_DIR / f"{TASK_ID}_pilot.json").write_text(json.dumps(result, indent=2))

    print(json.dumps({
        "verdict": result["verdict"], "pipeline_ok": pipeline_ok,
        "ablation_legs": n_abl_legs, "reasoning_methods": n_reas,
        "value_norm_concentration": result.get("value_norm_concentration"),
        "h7_degradation_pp": result.get("h7_reasoning_degradation_pp"),
        "anchor_effect_pp": result.get("anchor_effect_pp"),
        "errors": list(result["errors"].keys()), "elapsed_sec": result["elapsed_sec"],
    }, indent=2))

    mark_done("success" if pipeline_ok else "fail",
              f"E5 pilot {result['verdict']}: {n_abl_legs} ablation legs, {n_reas} reasoning "
              f"methods, vn_conc={result.get('value_norm_concentration')}, "
              f"h7_deg_pp={result.get('h7_reasoning_degradation_pp')}",
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
