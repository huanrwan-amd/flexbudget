# FlexBudget

**Interchangeable KV importance signals under one compiled FlexAttention budget.**

Poster, PyTorch Conference North America 2026.

---

## The problem

Long-context inference forces you to evict part of the KV cache. The literature
offers several competing rules for *what to drop* — keep the most recent tokens
(sliding window), keep sink tokens plus a local window (StreamingLLM), keep the
tokens that accumulated the most attention (H2O), keep the tokens with the
highest query–key affinity (SnapKV).

Each of those methods ships its heuristic **welded to its own hand-written
kernel**. So when one beats another, the comparison confounds two things at
once: the rule may have been smarter, or the code may simply have been faster.

## The move

Separate them. `flexbudget/working_set_mask.py` expresses eviction as a single
compiled `mask_mod` for `torch.nn.attention.flex_attention`: per query block it
admits the top-B KV blocks under a fixed token budget, plus anchors, intersected
with causality. The importance signal that ranks blocks is factored out into an
interchangeable scorer.

The budget, block size, context length, task mix and seed are held **fixed**;
only the signal swaps. The eviction policy becomes a controlled variable.

Training-free. No custom kernels. Core PyTorch only.

```python
import torch, flexbudget as fb
from torch.nn.attention.flex_attention import flex_attention

stats  = fb.BlockStats(n_q_blocks=nq, n_kv_blocks=nk, block_size=128,
                       attn_mass=mass, qk_affinity=aff, device=dev)
scores = fb.SCORERS["qk_affinity"](stats)         # <- the swappable part
ws     = fb.WorkingSetBlockMask.from_scores(scores, seq_len, budget_fraction=0.25)

out = torch.compile(flex_attention, dynamic=False)(q, k, v, block_mask=ws.block_mask)
```

Swapping `"qk_affinity"` for `"recency"`, `"attn_mass"`, `"combined"` or
`"position_corrective"` changes the policy and nothing else.

### Why it reduces exactly to dense attention at full budget

```python
def mask_mod(b, h, q_idx, kv_idx):
    qb = q_idx // block_size
    kb = kv_idx // block_size
    return adm[qb, kb] & (kv_idx <= q_idx)
```

The `& (kv_idx <= q_idx)` term is token-level causality. Block admission alone
would let the diagonal cell `qb == kb` attend to future tokens inside its own
block. With it, a 100% budget is numerically identical to dense causal
attention — `make_noop_mask()` asserts this, and `e0_parity_gate` measures it
against `torch` SDPA, which shares no code with the mask builder.

---

## What is here

```
flexbudget/working_set_mask.py   the contribution: ~360 lines, torch + stdlib only
experiments/                     e0..e5 drivers, the env shim, the result validator
figures/                         figure generators + a geometric layout auditor
results/                         the JSON every number traces to
slurm/run_experiment.sbatch      generic ROCm + pyxis launcher
```

| Experiment | Question |
|---|---|
| `e0_parity_gate` | Does the mask reduce to dense causal at 100% budget? |
| `e1_budget_knee` | How far can the budget fall before accuracy breaks? |
| `e2_midcontext`  | Does it survive a mid-context needle? |
| `e3_systems`     | What does it cost in memory, TTFT and decode throughput? |
| `e4_signal_comparison` | Do the rival signals differ at equal budget? |
| `e5_ablation_reasoning` | What do the anchors contribute? |

## Reproducing

```bash
pip install -r requirements.txt
make selftest       # mask degenerates to dense causal at 100% budget
make figures        # regenerate every figure from results/*.json
make audit          # geometric layout audit of the mechanism figure
```

Figures read only `results/*.json`. Every Wilson interval is **recomputed from
the raw (k, N) counts**; CI fields written into the JSON are ignored.

To rerun an experiment on a SLURM + ROCm cluster:

```bash
sbatch --export=ALL,EXPERIMENT=e0_parity_gate slurm/run_experiment.sbatch
```

---

## Results, and their limits

Measured on **one AMD Instinct MI300X**, ROCm 7.2.1, torch 2.10.

| | |
|---|---|
| 71–74% | lower **resident per-query** KV working set, Qwen2.5-7B geometry, 8K–32K |
| 3.21–3.78× | decode throughput vs dense at 8K / 16K / 32K |
| 0.26–0.29× | time to first token vs dense, block-sparse prefill |
| 15/16 vs 0/16 | at a 50% budget: scored working set vs a fixed sink+window baseline |
| ~2.1× | faster block-sparse attention at a 25% budget (wall clock ratio 0.4764) |
| < 0.016 | max deviation from dense causal at full budget, all three head-dim legs, against an independent `torch` SDPA reference (tolerance 0.05) |

**Read the memory number carefully.** 71–74% is how much of the cache *one
query touches*. It is not a smaller allocation — `peak_alloc_ratio = 1.00` and
`union_frac = 1.00` are recorded in `results/e3_systems.json`.

### What this does not claim

- **Pilot scale.** 16 trials per cell, one seed. Variance is unmeasured.
- **Mid-context is a failure.** At 30% needle depth the working set collapses
  to 1/16 where dense holds 16/16. A position-corrective variant repaired the
  edges and left the middle on the floor. It is reported as a negative result,
  not quietly dropped.
- **`e1` has a known defect.** In the budget sweep the `h2o` and `working` arms
  shared a code path, so those two are not independent in
  `results/e1_budget_knee.json`. Fixed in code; **not yet re-run**. Do not read
  that file as a four-way comparison.
- **`e4`/`e5` run on the eager path**, applying admission as a large-negative
  additive bias, not through `flex_attention`. The signal comparison is still
  same-code/same-budget; the *kernel* claims rest on `e0` and `e3`.
- **`e5`'s reasoning probe is a synthetic GSM8K-*style* generator**, not the
  GSM8K benchmark.
- No claim beyond one GPU, two model sizes, and these tasks.

## Notes for anyone bringing this up on ROCm

Three defects in the published ROCm PyTorch container block real workloads.
`experiments/_env_fix.py` works around the first two; import it **before**
`transformers`.

1. **torchvision** is built against a different torch and raises
   `operator torchvision::nms does not exist`. `transformers` imports it on
   model load even for pure-text LLMs. Fixed by forcing
   `is_torchvision_available()` false — *not* by stubbing `sys.modules`, which
   leaks placeholder objects into torch and breaks `flex_attention`.
2. **torchaudio**'s native lib is broken and `transformers` imports it
   unconditionally via `loss_utils`. Availability flags do not help; a
   dunder-guarded `sys.modules` stub does.
3. **`pip install --target` without `--no-deps` pulls a CPU build of torch**
   that shadows the ROCm build on `PYTHONPATH`. Jobs then run silently on CPU
   and report the timings as if they were GPU numbers.
   `experiments/validate_result.py` gates on `env.cuda_available` and a
   `device_name` match precisely to catch this.

`$HOME` is read-only inside the container, so `TRITON_CACHE_DIR`,
`TORCHINDUCTOR_CACHE_DIR`, `HF_HOME`, `MPLCONFIGDIR` and `HOME` must all be
redirected to writable paths, or nothing compiles. See
`slurm/run_experiment.sbatch`.

## Licence

MIT. See `LICENSE`.
