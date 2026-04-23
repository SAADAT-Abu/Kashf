import torch
import pytest
from kashf.model import (
    ACTHalting,
    Expert,
    FactoredEmbedding,
    KashfBlock,
    KashfConfig,
    KashfModel,
    KashfRecurrentBlock,
    LTIInjection,
    LoopGate,
    MQAttention,
    MicroMoEFFN,
    RMSNorm,
    precompute_rope_freqs,
)

B, T = 2, 8


def cfg(**overrides) -> KashfConfig:
    defaults = dict(
        vocab_size=200, dim=64, embed_dim=32, n_heads=2, head_dim=32,
        max_seq_len=32, max_loop_iters=3, n_routed_experts=2,
        n_shared_experts=1, expert_dim=32, act_threshold=0.99,
        rope_theta=500000.0, dropout=0.0,
    )
    defaults.update(overrides)
    return KashfConfig(**defaults)


# ---------------------------------------------------------------------------
# FactoredEmbedding
# ---------------------------------------------------------------------------

class TestFactoredEmbedding:
    def setup_method(self):
        self.c = cfg(); self.m = FactoredEmbedding(self.c)

    def test_output_shape(self):
        assert self.m(torch.randint(0, self.c.vocab_size, (B, T))).shape == (B, T, self.c.dim)

    def test_fewer_params_than_standard(self):
        assert sum(p.numel() for p in self.m.parameters()) < self.c.vocab_size * self.c.dim

    def test_no_nan(self):
        assert not torch.isnan(self.m(torch.randint(0, self.c.vocab_size, (B, T)))).any()


# ---------------------------------------------------------------------------
# MQAttention
# ---------------------------------------------------------------------------

class TestMQAttention:
    def setup_method(self):
        self.c = cfg(); self.m = MQAttention(self.c)
        self.f = precompute_rope_freqs(self.c.head_dim, self.c.max_seq_len)

    def test_output_shape(self):
        x = torch.randn(B, T, self.c.dim)
        assert self.m(x, self.f[:T]).shape == (B, T, self.c.dim)

    def test_single_kv_head(self):
        assert self.m.wk.out_features == self.c.head_dim
        assert self.m.wv.out_features == self.c.head_dim

    def test_qk_norm_exists(self):
        assert isinstance(self.m.q_norm, RMSNorm)
        assert isinstance(self.m.k_norm, RMSNorm)

    def test_kv_cache_accumulates(self):
        cache = {}; x = torch.randn(B, T, self.c.dim)
        self.m(x, self.f[:T], kv_cache=cache, cache_key="a")
        k1 = cache["a"]["k"].shape[1]
        self.m(x, self.f[:T], kv_cache=cache, cache_key="a")
        assert cache["a"]["k"].shape[1] == k1 + T

    def test_finite_on_extreme_input(self):
        x = torch.randn(B, T, self.c.dim) * 100.0
        assert torch.isfinite(self.m(x, self.f[:T])).all()

    def test_no_bias_on_projections(self):
        for n in ("wq", "wk", "wv", "wo"):
            assert getattr(self.m, n).bias is None


# ---------------------------------------------------------------------------
# MicroMoEFFN
# ---------------------------------------------------------------------------

class TestMicroMoEFFN:
    def setup_method(self):
        self.c = cfg(); self.m = MicroMoEFFN(self.c)

    def test_output_shape(self):
        assert self.m(torch.randn(B, T, self.c.dim)).shape == (B, T, self.c.dim)

    def test_shared_expert_always_fires(self):
        for e in self.m.routed_experts:
            for p in e.parameters(): p.data.zero_()
        assert self.m(torch.randn(B, T, self.c.dim)).abs().sum() > 0

    def test_router_bias_is_buffer(self):
        assert "router_bias" not in {n for n, _ in self.m.named_parameters()}

    def test_no_nan(self):
        assert not torch.isnan(self.m(torch.randn(B, T, self.c.dim))).any()

    def test_router_gets_gradient(self):
        x = torch.randn(B, T, self.c.dim)
        self.m(x).sum().backward()
        assert self.m.router.weight.grad is not None
        assert self.m.router.weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# LoopGate
# ---------------------------------------------------------------------------

class TestLoopGate:
    def setup_method(self):
        self.m = LoopGate(max_loops=6)

    def test_output_shape(self):
        x = torch.randn(B, T, 64)
        assert self.m(x, 0).shape == x.shape

    def test_gate_in_01(self):
        g = torch.sigmoid(self.m.gate.weight)
        assert (g > 0).all() and (g < 1).all()

    def test_depth_extrapolation_clamps(self):
        x = torch.randn(B, T, 64)
        assert self.m(x, 100).shape == x.shape

    def test_init_ones(self):
        assert torch.allclose(self.m.gate.weight, torch.ones_like(self.m.gate.weight))


# ---------------------------------------------------------------------------
# LTIInjection
# ---------------------------------------------------------------------------

class TestLTIInjection:
    def setup_method(self):
        self.m = LTIInjection(64)

    def test_spectral_radius_lt_1(self):
        assert self.m.get_A().max().item() < 1.0

    def test_output_shape(self):
        h = torch.randn(B, T, 64); e = torch.randn(B, T, 64); t = torch.randn(B, T, 64)
        assert self.m(h, e, t).shape == (B, T, 64)

    def test_stable_after_grad_step(self):
        opt = torch.optim.SGD(self.m.parameters(), lr=1)
        h = torch.randn(B, T, 64); e = torch.randn(B, T, 64); t = torch.randn(B, T, 64)
        self.m(h, e, t).sum().backward()
        opt.step()
        assert self.m.get_A().max().item() < 1.0


# ---------------------------------------------------------------------------
# KashfBlock
# ---------------------------------------------------------------------------

class TestKashfBlock:
    def setup_method(self):
        self.c = cfg()
        self.f = precompute_rope_freqs(self.c.head_dim, self.c.max_seq_len)

    def test_dense_shape(self):
        b = KashfBlock(self.c, use_moe=False)
        assert b(torch.randn(B, T, self.c.dim), self.f[:T]).shape == (B, T, self.c.dim)

    def test_moe_shape(self):
        b = KashfBlock(self.c, use_moe=True)
        assert b(torch.randn(B, T, self.c.dim), self.f[:T]).shape == (B, T, self.c.dim)

    def test_dense_uses_expert(self):
        assert isinstance(KashfBlock(self.c, use_moe=False).ffn, Expert)

    def test_moe_uses_micromoe(self):
        assert isinstance(KashfBlock(self.c, use_moe=True).ffn, MicroMoEFFN)

    def test_no_nan(self):
        b = KashfBlock(self.c, use_moe=True)
        assert not torch.isnan(b(torch.randn(B, T, self.c.dim), self.f[:T])).any()


# ---------------------------------------------------------------------------
# KashfRecurrentBlock
# ---------------------------------------------------------------------------

class TestKashfRecurrentBlock:
    def setup_method(self):
        self.c = cfg(); self.b = KashfRecurrentBlock(self.c)
        self.f = precompute_rope_freqs(self.c.head_dim, self.c.max_seq_len)

    def _fwd(self, **kw):
        h = torch.randn(B, T, self.c.dim); e = torch.randn(B, T, self.c.dim)
        return self.b(h, e, self.f[:T], **kw)

    def test_output_shape(self):
        assert self._fwd().shape == (B, T, self.c.dim)

    def test_single_loop(self):
        assert self._fwd(n_loops=1).shape == (B, T, self.c.dim)

    def test_loops_change_output(self):
        assert not torch.allclose(self._fwd(n_loops=1), self._fwd(n_loops=3))

    def test_depth_extrapolation(self):
        assert self._fwd(n_loops=self.c.max_loop_iters + 5).shape == (B, T, self.c.dim)

    def test_spectral_radius(self):
        assert self.b.injection.get_A().max().item() < 1.0

    def test_no_nan(self):
        assert not torch.isnan(self._fwd()).any()

    def test_kv_cache_keys(self):
        cache = {}; self._fwd(kv_cache=cache)
        assert any(k.startswith("kashf_loop_") for k in cache)


# ---------------------------------------------------------------------------
# KashfModel
# ---------------------------------------------------------------------------

class TestKashfModel:
    def setup_method(self):
        self.c = cfg(); self.m = KashfModel(self.c)
        self.ids = torch.randint(0, self.c.vocab_size, (B, T))

    def test_forward_shape(self):
        assert self.m(self.ids).shape == (B, T, self.c.vocab_size)

    def test_no_nan(self):
        assert not torch.isnan(self.m(self.ids)).any()

    def test_generate_shape(self):
        assert self.m.generate(self.ids, max_new_tokens=4, n_loops=2).shape == (B, T + 4)

    def test_weight_tying(self):
        assert self.m.prelude_block is self.m.coda_block

    def test_distinct_cache_keys(self):
        cache = {}
        with torch.no_grad(): self.m(self.ids, kv_cache=cache)
        assert "prelude" in cache and "coda" in cache

    def test_parameter_count_tied(self):
        c = self.m.parameter_count()
        assert c["total"] > c["unique (deduped)"] > 0

    def test_lti_spectral_radius(self):
        assert self.m.recurrent.injection.get_A().max().item() < 1.0

    def test_single_token(self):
        ids = torch.randint(0, self.c.vocab_size, (B, 1))
        assert self.m(ids).shape == (B, 1, self.c.vocab_size)

    def test_depth_extrapolation(self):
        assert self.m(self.ids, n_loops=self.c.max_loop_iters + 4).shape == (B, T, self.c.vocab_size)

    def test_residual_init_scaled(self):
        import statistics
        wo_stds, wq_stds = [], []
        for _ in range(20):
            m = KashfModel(self.c)
            wo_stds.append(m.prelude_block.attn.wo.weight.std().item())
            wq_stds.append(m.prelude_block.attn.wq.weight.std().item())
        assert statistics.mean(wo_stds) < statistics.mean(wq_stds)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradients:
    def setup_method(self):
        self.c = cfg(); self.m = KashfModel(self.c)

    def test_all_params_receive_gradients(self):
        ids = torch.randint(0, self.c.vocab_size, (B, T))
        tgt = torch.randint(0, self.c.vocab_size, (B, T))
        loss = torch.nn.functional.cross_entropy(
            self.m(ids).view(-1, self.c.vocab_size), tgt.view(-1)
        )
        loss.backward()
        exempt = {"loop_gate", "recurrent.block.ffn.routed_experts"}
        missing = [
            n for n, p in self.m.named_parameters()
            if p.grad is None and not any(e in n for e in exempt)
        ]
        assert missing == [], f"No gradient: {missing}"

    def test_router_gradient_via_ste(self):
        self.m(torch.randint(0, self.c.vocab_size, (B, T))).sum().backward()
        g = self.m.recurrent.block.ffn.router.weight.grad
        assert g is not None and g.abs().sum() > 0

    def test_lti_params_gradient(self):
        self.m(torch.randint(0, self.c.vocab_size, (B, T))).sum().backward()
        inj = self.m.recurrent.injection
        assert inj.log_A.grad is not None and inj.B.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
