"""
Kashf — Compact Recurrent-Depth Transformer.

Architecture:
    FactoredEmbedding → PreludeBlock → KashfRecurrentBlock → CodaBlock (tied) → LM Head

Key design decisions:
    FactoredEmbedding   two-stage lookup to halve embedding parameter cost
    MQAttention         single shared KV head with QK-Norm and attention soft-cap
    MicroMoEFFN         hard-routed 2-expert MoE + 1 shared expert with STE routing
    LoopGate            per-loop scalar gate replaces LoRA depth adapter
    LTIInjection        stable recurrent update; spectral radius < 1 by construction
    ACTHalting          adaptive computation time; easy tokens halt early
    Weight tying        prelude and coda share the same block instance
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class KashfConfig:
    """Hyperparameter configuration for Kashf."""

    vocab_size: int = 50257         # GPT-2 tokenizer default
    dim: int = 256                  # model hidden dimension
    embed_dim: int = 128            # factored embedding lookup dimension (< dim)
    n_heads: int = 4                # query attention heads
    head_dim: int = 64              # dimension per attention head
    max_seq_len: int = 512          # maximum sequence length for RoPE precomputation
    max_loop_iters: int = 6         # default recurrent depth T
    prelude_layers: int = 1
    coda_layers: int = 1            # same block instance as prelude (weights tied)
    n_routed_experts: int = 2       # number of hard-routed MoE experts
    n_shared_experts: int = 1       # always-active shared expert count
    expert_dim: int = 256           # inner dimension of each expert FFN
    act_threshold: float = 0.99     # ACT halting threshold
    rope_theta: float = 500000.0    # RoPE base frequency
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def precompute_rope_freqs(
    dim: int, max_len: int, theta: float = 500000.0
) -> torch.Tensor:
    """Precompute complex RoPE rotation matrices for positions 0..max_len-1."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embeddings to a query or key tensor (B, T, H, head_dim)."""
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return (
        torch.view_as_real(xc * freqs_cis.unsqueeze(0).unsqueeze(2))
        .flatten(-2)
        .to(x.dtype)
    )


def loop_index_embedding(
    h: torch.Tensor, loop_t: int, loop_dim: int, theta: float = 10000.0
) -> torch.Tensor:
    """Inject a sinusoidal loop-index signal into the first loop_dim channels of h."""
    freqs = 1.0 / (
        theta ** (torch.arange(0, loop_dim, 2, device=h.device, dtype=h.dtype) / loop_dim)
    )
    angles = loop_t * freqs
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)[:loop_dim]
    emb_full = torch.zeros(h.shape[-1], device=h.device, dtype=h.dtype)
    emb_full[:loop_dim] = emb
    return h + emb_full.unsqueeze(0).unsqueeze(0)


class Expert(nn.Module):
    """Single SwiGLU feed-forward expert: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up   = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim,  bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class LTIInjection(nn.Module):
    """
    Stable recurrent state update: h = A·h + B·e + transformer_out.

    Guarantees spectral radius ρ(A) < 1 by construction via ZOH discretization:
        A = exp(-exp(log_dt + log_A))  — always in (0, 1) per element.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.log_A  = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(1))
        self.B      = nn.Parameter(torch.ones(dim) * 0.1)

    def get_A(self) -> torch.Tensor:
        return torch.exp(-torch.exp((self.log_dt + self.log_A).clamp(-20, 20)))

    def forward(self, h: torch.Tensor, e: torch.Tensor, transformer_out: torch.Tensor) -> torch.Tensor:
        A = self.get_A()
        return A * h + self.B * e + transformer_out


class ACTHalting(nn.Module):
    """Adaptive Computation Time halting (Graves, 2016). Predicts per-position halt probability."""

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Factored Embedding
# ---------------------------------------------------------------------------


class FactoredEmbedding(nn.Module):
    """
    Two-stage token embedding: Embedding(vocab, embed_dim) → Linear(embed_dim, dim).

    Reduces embedding parameter count by vocab * (dim - embed_dim) compared to
    a standard single-stage embedding. At vocab=50257, dim=256, embed_dim=128:
    standard = 12.9M params; factored = 6.6M params.
    """

    def __init__(self, cfg: KashfConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.proj  = nn.Linear(cfg.embed_dim, cfg.dim, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(token_ids))


# ---------------------------------------------------------------------------
# Multi-Query Attention
# ---------------------------------------------------------------------------


class MQAttention(nn.Module):
    """
    Multi-Query Attention: all query heads share one K and one V head.

    Additions over vanilla MQA:
        QK-Norm     RMSNorm on Q and K before RoPE to stabilize across loop depth
        Soft-cap    tanh(logits / 50) * 50 prevents attention entropy collapse
    """

    def __init__(self, cfg: KashfConfig):
        super().__init__()
        self.n_heads  = cfg.n_heads
        self.head_dim = cfg.head_dim

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.dim, bias=False)
        self.dropout_p = cfg.dropout
        self.q_norm = RMSNorm(cfg.head_dim)
        self.k_norm = RMSNorm(cfg.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, 1, self.head_dim)
        v = self.wv(x).view(B, T, 1, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        if kv_cache is not None:
            if cache_key in kv_cache:
                k = torch.cat([kv_cache[cache_key]["k"], k], dim=1)
                v = torch.cat([kv_cache[cache_key]["v"], v], dim=1)
            kv_cache[cache_key] = {"k": k.detach(), "v": v.detach()}

        S = k.shape[1]
        k = k.expand(B, S, self.n_heads, self.head_dim)
        v = v.expand(B, S, self.n_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        attn  = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn  = torch.tanh(attn / 50.0) * 50.0
        if mask is not None:
            attn = attn + mask
        attn = F.dropout(F.softmax(attn, dim=-1), p=self.dropout_p, training=self.training)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Micro-MoE FFN
# ---------------------------------------------------------------------------


class MicroMoEFFN(nn.Module):
    """
    2 hard-routed experts (top-1) + 1 always-active shared expert.

    Routing uses a straight-through estimator (STE): argmax in the forward pass,
    softmax gradient in the backward pass — making the router differentiable with
    zero change to inference behaviour. A load-balance bias buffer (not a learned
    param) prevents token collapse onto one expert (DeepSeek-V3 scheme).
    """

    def __init__(self, cfg: KashfConfig):
        super().__init__()
        self.n_routed_experts = cfg.n_routed_experts
        self.router = nn.Linear(cfg.dim, cfg.n_routed_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_routed_experts))
        self.routed_experts = nn.ModuleList(
            [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_routed_experts)]
        )
        self.shared_expert = Expert(cfg.dim, cfg.expert_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        flat   = x.view(B * T, D)
        logits = self.router(flat)
        scores = F.softmax(logits, dim=-1)
        hard   = F.one_hot((logits + self.router_bias).argmax(dim=-1), self.n_routed_experts).to(flat.dtype)
        ste_weights = hard + scores - scores.detach()

        out = torch.zeros_like(flat)
        for eid in range(self.n_routed_experts):
            w    = ste_weights[:, eid]
            mask = w > 0.5
            if mask.any():
                out[mask] = out[mask] + w[mask].unsqueeze(1) * self.routed_experts[eid](flat[mask])

        out = out + self.shared_expert(flat)
        return out.view(B, T, D)


# ---------------------------------------------------------------------------
# Loop Gate
# ---------------------------------------------------------------------------


class LoopGate(nn.Module):
    """
    Per-loop learned scalar gate in (0, 1) applied to transformer output.
    Initialized to ones so sigmoid(1) ≈ 0.73 at the start of training.
    """

    def __init__(self, max_loops: int):
        super().__init__()
        self.gate = nn.Embedding(max_loops, 1)
        nn.init.ones_(self.gate.weight)

    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        t_idx = min(loop_t, self.gate.num_embeddings - 1)
        g = torch.sigmoid(self.gate(torch.tensor(t_idx, device=x.device)))
        return x * g


# ---------------------------------------------------------------------------
# Kashf Transformer Block
# ---------------------------------------------------------------------------


class KashfBlock(nn.Module):
    """
    Pre-norm transformer block: MQAttention + dense or MoE FFN.

    use_moe=False  Prelude/Coda: dense SwiGLU with expert_dim = dim * 4 // 3
    use_moe=True   Recurrent:    MicroMoEFFN with hard-routed + shared experts
    """

    def __init__(self, cfg: KashfConfig, use_moe: bool = False):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.ffn_norm  = RMSNorm(cfg.dim)
        self.attn = MQAttention(cfg)
        self.ffn  = MicroMoEFFN(cfg) if use_moe else Expert(cfg.dim, cfg.dim * 4 // 3)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[dict] = None,
        cache_key: str = "default",
    ) -> torch.Tensor:
        x = x + self.resid_drop(self.attn(self.attn_norm(x), freqs_cis, mask, kv_cache, cache_key))
        x = x + self.resid_drop(self.ffn(self.ffn_norm(x)))
        return x


# ---------------------------------------------------------------------------
# Kashf Recurrent Block
# ---------------------------------------------------------------------------


class KashfRecurrentBlock(nn.Module):
    """
    Single KashfBlock looped T times with LTI injection, LoopGate, and ACT halting.

    Per-loop step:
        1. loop_index_embedding  inject sinusoidal depth signal into h
        2. RMSNorm(h + e)        combine hidden state with frozen prelude output
        3. KashfBlock (MoE)      compute attention + MoE FFN
        4. LoopGate              scale output by per-loop scalar
        5. LTIInjection          stable update h = A·h + B·e + gated_out
        6. ACTHalting            accumulate per-position halting probabilities

    ACT remainder trick ensures weights sum exactly to 1.0 per position.
    """

    def __init__(self, cfg: KashfConfig):
        super().__init__()
        self.cfg = cfg
        self.block     = KashfBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act       = ACTHalting(cfg.dim)
        self.loop_gate = LoopGate(cfg.max_loop_iters)
        self.norm      = RMSNorm(cfg.dim)

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
    ) -> torch.Tensor:
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted       = torch.zeros(B, T, device=h.device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=h.device)
        h_out        = torch.zeros_like(h)

        for t in range(n_loops):
            h_loop   = loop_index_embedding(h, t, self.cfg.dim // 8)
            combined = self.norm(h_loop + e)
            trans_out = self.block(combined, freqs_cis, mask, kv_cache, f"kashf_loop_{t}")
            trans_out = self.loop_gate(trans_out, t)
            h = self.injection(h, e, trans_out)

            p             = self.act(h)
            still_running = ~halted
            is_last       = t == n_loops - 1
            remainder     = (1.0 - cumulative_p).clamp(min=0)

            weight = torch.where(
                (cumulative_p + p >= self.cfg.act_threshold) | is_last,
                remainder,
                p,
            )
            weight = weight * still_running.float()
            h_out  = h_out + weight.unsqueeze(-1) * h

            cumulative_p = cumulative_p + p * still_running.float()
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            if halted.all() and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# Kashf Model
# ---------------------------------------------------------------------------


class KashfModel(nn.Module):
    """
    Kashf — compact Recurrent-Depth Transformer designed for single-GPU training.

        FactoredEmbedding
             ↓
        PreludeBlock (KashfBlock dense, weight-tied with Coda)
             ↓
        KashfRecurrentBlock (KashfBlock MoE, looped T times)
             ↓
        CodaBlock (same KashfBlock instance as Prelude)
             ↓
        RMSNorm → LM Head

    Weight tying: prelude_block and coda_block are the same Python object.
    """

    def __init__(self, cfg: KashfConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = FactoredEmbedding(cfg)

        freqs = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cis", freqs)

        self.prelude_block = KashfBlock(cfg, use_moe=False)
        self.coda_block    = self.prelude_block   # same object — weights tied

        self.recurrent = KashfRecurrentBlock(cfg)
        self.norm      = RMSNorm(cfg.dim)
        self.head      = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        """Residual output projections use std/√2; all others use std=0.02."""
        for name, p in self.named_parameters():
            if "loop_gate" in name or p.dim() < 2:
                continue
            std = 0.02 / (2.0 ** 0.5) if name.endswith((".wo.weight", ".down.weight")) else 0.02
            nn.init.normal_(p, std=std)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        mask = torch.full((1, 1, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
        return torch.triu(mask, diagonal=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        n_loops: Optional[int] = None,
        kv_cache: Optional[dict] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        T      = input_ids.shape[1]
        device = input_ids.device

        x         = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos : start_pos + T]
        mask      = self._causal_mask(T, device, x.dtype) if T > 1 else None

        x = self.prelude_block(x, freqs_cis, mask, kv_cache, "prelude")
        e = x
        x = self.recurrent(x, e, freqs_cis, mask, n_loops, kv_cache)
        x = self.coda_block(x, freqs_cis, mask, kv_cache, "coda")

        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        n_loops: int = 6,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Autoregressive generation with KV caching."""
        kv_cache: dict = {}
        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            cur_ids   = input_ids if step == 0 else input_ids[:, -1:]
            start_pos = 0 if step == 0 else prompt_len + step - 1
            logits    = self.forward(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            logits    = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = logits.topk(top_k)
                logits[logits < v[:, -1:]] = float("-inf")

            next_tok   = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids  = torch.cat([input_ids, next_tok], dim=1)

        return input_ids

    def parameter_count(self) -> dict:
        """Return total (with tied duplicates) and unique (deduped) parameter counts."""
        total  = sum(p.numel() for _, p in self.named_parameters(remove_duplicate=False))
        unique = sum(p.numel() for _, p in self.named_parameters(remove_duplicate=True))
        return {"total": total, "unique (deduped)": unique}
