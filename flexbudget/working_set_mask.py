"""WorkingSetBlockMask: a compiled KV-budget controller for torch.nn.attention.flex_attention.

FlexBudget core artifact (cand_flexbudget). Training-free, inference-only.

Framing (VM-controller): the full KV cache = disk; the budgeted working set = RAM;
the ``mask_mod`` = the residence table; the ``score_mod`` / importance scorer = the
page-replacement policy.

Design
------
* Block-level importance scores are computed at prefill from an interchangeable
  ``importance scorer``. Per query block, the top-B KV blocks by score are admitted
  via a ``mask_mod``, plus always-on anchors (attention sinks + local sliding window),
  under a strict causal constraint.
* Budget B is a *fraction of KV blocks* (block size stated explicitly, default 128).
* Everything is expressed as static boolean block-tensors closed over by the mask_mod,
  so ``torch.compile(dynamic=False)`` produces one static graph with no per-input
  recompiles or graph breaks.
* Composable algebra via ``and_masks`` / ``or_masks`` mirrors the FlexAttention API:
  ``causal AND (sinks OR local_window OR top_B_blocks)``.

Public API
----------
* ``ScoreVocabulary`` -- named block-importance scorers: recency, attn_mass,
  qk_affinity, combined, position_corrective.
* ``build_block_importance(...)`` -- turn per-(query-block, kv-block) scores into a
  static admitted-block boolean tensor honouring budget + anchors + causality.
* ``WorkingSetBlockMask`` -- wraps ``create_block_mask`` with an admitted-block tensor;
  exposes ``.block_mask`` for ``flex_attention(..., block_mask=...)`` and ``.skip_rate``.
* ``make_noop_mask(...)`` -- causal-equivalent working-set mask (100% budget) used by the
  self-test to prove the wrapper reduces to plain causal attention.
* Baselines as masks: ``streamingllm_admitted``, ``dense_admitted``.

Keep this file dependency-light: only torch + stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch.nn.attention.flex_attention import create_block_mask, and_masks, or_masks

# ---------------------------------------------------------------------------
# Block geometry helpers
# ---------------------------------------------------------------------------

DEFAULT_BLOCK_SIZE = 128


def n_blocks(seq_len: int, block_size: int) -> int:
    return (seq_len + block_size - 1) // block_size


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Block-importance scorers (the "page-replacement policy" vocabulary)
#
# Each scorer maps prefill statistics -> a [n_q_blocks, n_kv_blocks] score tensor.
# Higher score == more important == more likely to be admitted under budget.
# Scores are only *ranked*, so absolute scale is irrelevant.
# ---------------------------------------------------------------------------


@dataclass
class BlockStats:
    """Prefill statistics aggregated to KV-block granularity.

    All tensors are on CPU/GPU float; shapes are documented per field. Any field
    may be ``None`` if a given scorer does not need it, but the requested scorer
    will raise if its required field is missing.
    """

    n_q_blocks: int
    n_kv_blocks: int
    block_size: int
    # [n_kv_blocks] mean attention mass received by each KV block over all queries
    # (H2O-style accumulated attention).
    attn_mass: Optional[torch.Tensor] = None
    # [n_q_blocks, n_kv_blocks] mean query-key affinity (pre-softmax logit) between
    # query block i and kv block j (SnapKV-style).
    qk_affinity: Optional[torch.Tensor] = None
    # [n_kv_blocks] mean value-vector L2 norm per KV block (reasoning diagnostic).
    value_norm: Optional[torch.Tensor] = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))


class ScoreVocabulary:
    """Named block-importance scorers. Each returns [n_q_blocks, n_kv_blocks] scores."""

    @staticmethod
    def recency(stats: BlockStats) -> torch.Tensor:
        """Prefer KV blocks closest (in block index) to the query block."""
        qi = torch.arange(stats.n_q_blocks, device=stats.device).unsqueeze(1)
        kj = torch.arange(stats.n_kv_blocks, device=stats.device).unsqueeze(0)
        # closer -> higher score; distance is (qi - kj) for causal pairs.
        return -(qi - kj).abs().float()

    @staticmethod
    def attn_mass(stats: BlockStats) -> torch.Tensor:
        """H2O-style: prefer KV blocks that historically receive the most attention."""
        if stats.attn_mass is None:
            raise ValueError("attn_mass scorer requires stats.attn_mass")
        return stats.attn_mass.unsqueeze(0).expand(stats.n_q_blocks, -1).float()

    @staticmethod
    def qk_affinity(stats: BlockStats) -> torch.Tensor:
        """SnapKV-style: prefer KV blocks with high query-specific QK affinity."""
        if stats.qk_affinity is None:
            raise ValueError("qk_affinity scorer requires stats.qk_affinity")
        return stats.qk_affinity.float()

    @staticmethod
    def combined(stats: BlockStats) -> torch.Tensor:
        """z-normalized attn_mass + qk_affinity."""
        a = ScoreVocabulary.attn_mass(stats)
        q = ScoreVocabulary.qk_affinity(stats)
        return _zscore(a) + _zscore(q)

    @staticmethod
    def position_corrective(
        stats: BlockStats, position_multiplier: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """QK-affinity x a U-shaped position multiplier (lost-in-the-middle correction).

        ``position_multiplier`` is a [n_kv_blocks] tensor (per-model calibrated). If not
        supplied, a default symmetric U-shape is used (edges 1.0, middle depressed).
        """
        base = ScoreVocabulary.qk_affinity(stats)
        if position_multiplier is None:
            position_multiplier = default_u_shape(stats.n_kv_blocks, device=stats.device)
        return base * position_multiplier.unsqueeze(0)


def _zscore(x: torch.Tensor) -> torch.Tensor:
    mu = x.mean()
    sd = x.std()
    if sd < 1e-8:
        return x - mu
    return (x - mu) / sd


def default_u_shape(n: int, device=None, floor: float = 0.5) -> torch.Tensor:
    """Symmetric U-shaped multiplier: 1.0 at the edges, ``floor`` in the middle."""
    if n <= 1:
        return torch.ones(n, device=device)
    idx = torch.linspace(-1.0, 1.0, n, device=device)
    # parabola: 1 at |idx|=1, floor at idx=0
    return floor + (1.0 - floor) * idx.pow(2)


SCORERS: dict[str, Callable[[BlockStats], torch.Tensor]] = {
    "recency": ScoreVocabulary.recency,
    "attn_mass": ScoreVocabulary.attn_mass,
    "qk_affinity": ScoreVocabulary.qk_affinity,
    "combined": ScoreVocabulary.combined,
    "position_corrective": ScoreVocabulary.position_corrective,
}


# ---------------------------------------------------------------------------
# Admitted-block construction: budget + anchors + causality
# ---------------------------------------------------------------------------


def build_block_importance(
    scores: torch.Tensor,
    budget_fraction: float,
    n_sink_blocks: int = 1,
    local_window_blocks: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """Turn [n_q_blocks, n_kv_blocks] importance scores into a boolean admitted-block map.

    Admitted[q, k] is True iff KV block k is in the working set of query block q, i.e.
    it is a causal block AND (k is a sink block OR k is within the local window OR k is
    among the top-B causal blocks by score, where B = ceil(budget_fraction * n_causal)).

    Returns a bool tensor [n_q_blocks, n_kv_blocks].
    """
    if not (0.0 < budget_fraction <= 1.0):
        raise ValueError(f"budget_fraction must be in (0,1], got {budget_fraction}")
    nq, nk = scores.shape
    device = scores.device
    qi = torch.arange(nq, device=device).unsqueeze(1)
    kj = torch.arange(nk, device=device).unsqueeze(0)

    causal_mask = (kj <= qi) if causal else torch.ones(nq, nk, dtype=torch.bool, device=device)

    # anchors: sink blocks (first n_sink_blocks) and local window (last local_window_blocks
    # before/at the query block), intersected with causality.
    sink_mask = (kj < n_sink_blocks) & causal_mask
    local_mask = (kj > (qi - local_window_blocks)) & causal_mask

    # top-B by score among causal blocks. Mask out non-causal with -inf so they never win.
    masked_scores = scores.masked_fill(~causal_mask, float("-inf"))
    # per query block, number of causal candidates = qi+1 (block indices 0..qi)
    n_causal = (qi.squeeze(1) + 1).clamp(min=1)  # [nq]
    B = torch.ceil(budget_fraction * n_causal.float()).long().clamp(min=1)  # [nq]

    topk_mask = torch.zeros(nq, nk, dtype=torch.bool, device=device)
    maxB = int(B.max().item())
    # argsort descending once, then threshold per-row by B[row].
    order = torch.argsort(masked_scores, dim=1, descending=True)  # [nq, nk]
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(nk, device=device).expand(nq, nk))
    topk_mask = rank < B.unsqueeze(1)
    topk_mask = topk_mask & causal_mask  # never admit a non-causal block

    admitted = causal_mask & (sink_mask | local_mask | topk_mask)
    return admitted


def admitted_to_mask_mod(admitted: torch.Tensor, block_size: int):
    """Build a token-level mask_mod closure from a [n_q_blocks, n_kv_blocks] bool tensor.

    The mask_mod maps (b, h, q_idx, kv_idx) -> bool by looking up the block cell AND
    enforcing TOKEN-LEVEL causality (kv_idx <= q_idx). The block-level admitted tensor
    admits or rejects whole (q_block, k_block) cells; on the diagonal cell (qb == kb) a
    pure block lookup would attend to FUTURE tokens within the same block, breaking
    causality and corrupting generation. AND-ing with (kv_idx <= q_idx) makes the mask
    reduce EXACTLY to dense causal attention at 100% budget (matches SDPA / native flex).
    The admitted tensor is closed over as a static tensor, so torch.compile sees fixed
    shapes.
    """
    adm = admitted  # captured

    def mask_mod(b, h, q_idx, kv_idx):
        qb = q_idx // block_size
        kb = kv_idx // block_size
        return adm[qb, kb] & (kv_idx <= q_idx)

    return mask_mod


# ---------------------------------------------------------------------------
# Baseline admitted maps (same kernel path -> algorithmic-only differences)
# ---------------------------------------------------------------------------


def dense_admitted(nq: int, nk: int, device=None) -> torch.Tensor:
    """Full causal (100% budget) -- the noop / dense-FlexAttention reference."""
    qi = torch.arange(nq, device=device).unsqueeze(1)
    kj = torch.arange(nk, device=device).unsqueeze(0)
    return kj <= qi


def streamingllm_admitted(
    nq: int, nk: int, n_sink_blocks: int = 1, local_window_blocks: int = 3, device=None
) -> torch.Tensor:
    """StreamingLLM: sink blocks + local sliding window only (no importance ranking)."""
    qi = torch.arange(nq, device=device).unsqueeze(1)
    kj = torch.arange(nk, device=device).unsqueeze(0)
    causal = kj <= qi
    sink = (kj < n_sink_blocks) & causal
    local = (kj > (qi - local_window_blocks)) & causal
    return sink | local


# ---------------------------------------------------------------------------
# WorkingSetBlockMask wrapper
# ---------------------------------------------------------------------------


class WorkingSetBlockMask:
    """Wrap create_block_mask with an admitted-block tensor + budget bookkeeping."""

    def __init__(
        self,
        admitted: torch.Tensor,
        block_size: int,
        seq_len: int,
        device=None,
    ):
        self.admitted = admitted
        self.block_size = block_size
        self.seq_len = seq_len
        self.device = device
        self.mask_mod = admitted_to_mask_mod(admitted, block_size)
        self.block_mask = create_block_mask(
            self.mask_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=str(device) if device is not None else "cpu",
        )

    @property
    def skip_rate(self) -> float:
        """Fraction of causal blocks that are NOT admitted (block skip rate)."""
        nq, nk = self.admitted.shape
        qi = torch.arange(nq, device=self.admitted.device).unsqueeze(1)
        kj = torch.arange(nk, device=self.admitted.device).unsqueeze(0)
        causal = kj <= qi
        n_causal = causal.sum().item()
        n_admitted = (self.admitted & causal).sum().item()
        if n_causal == 0:
            return 0.0
        return 1.0 - (n_admitted / n_causal)

    @classmethod
    def from_scores(
        cls,
        scores: torch.Tensor,
        seq_len: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
        budget_fraction: float = 0.25,
        n_sink_blocks: int = 1,
        local_window_blocks: int = 1,
        device=None,
    ) -> "WorkingSetBlockMask":
        admitted = build_block_importance(
            scores,
            budget_fraction=budget_fraction,
            n_sink_blocks=n_sink_blocks,
            local_window_blocks=local_window_blocks,
        )
        return cls(admitted, block_size, seq_len, device=device)


def make_noop_mask(
    seq_len: int, block_size: int = DEFAULT_BLOCK_SIZE, device=None
) -> WorkingSetBlockMask:
    """Causal-equivalent working-set mask (100% budget) for the self-test.

    Constructed via the full admitted-block path (build_block_importance at budget 1.0)
    to prove the wrapper reduces to plain causal attention.
    """
    nq = n_blocks(seq_len, block_size)
    nk = nq
    # uniform scores + 100% budget => all causal blocks admitted == dense causal.
    scores = torch.zeros(nq, nk, device=device)
    admitted = build_block_importance(scores, budget_fraction=1.0)
    ref = dense_admitted(nq, nk, device=device)
    assert torch.equal(admitted, ref), "noop mask is not causal-equivalent"
    return WorkingSetBlockMask(admitted, block_size, seq_len, device=device)


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "n_blocks",
    "BlockStats",
    "ScoreVocabulary",
    "SCORERS",
    "default_u_shape",
    "build_block_importance",
    "admitted_to_mask_mod",
    "dense_admitted",
    "streamingllm_admitted",
    "WorkingSetBlockMask",
    "make_noop_mask",
    "and_masks",
    "or_masks",
]
