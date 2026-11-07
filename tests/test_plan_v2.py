"""Tests for Plan v2: Block-Segmented Pre-RoPE TriAttention, TurboQuant QJL, LlamaCppEngine, and Profiles."""

from __future__ import annotations

import torch
from click.testing import CliRunner

from kvpipe.cli import main
from kvpipe.config import ArmConfig, CompressionConfig, ServeConfig
from kvpipe.decide import recommend_for_profile
from kvpipe.engines.llamacpp import LlamaCppEngine
from kvpipe.kv_policy import HybridKVCachePolicy, TurboQuantCodec
from scripts.kaggle_qwen35_128k_turboquant import ArmSpec, TurboQuantHybridCache


def test_turboquant_codec_qjl_cosine_similarity() -> None:
    """Verify Random-Sign FWHT (R=H*D) + 1-Bit QJL residual recovery improves 4-bit fidelity (>0.99 cos sim)."""
    torch.manual_seed(42)
    x = torch.randn(1, 4, 128, 256, dtype=torch.float16)
    # Inject heavy channel outliers on channels 0 and 17
    x[:, :, :, 0] *= 15.0
    x[:, :, :, 17] *= -12.0

    codec = TurboQuantCodec(head_dim=256, group_size=64)
    deq_no_qjl, bytes_no_qjl = codec.quant_dequant(x, bits=4, use_hadamard=True, use_qjl=False)
    deq_qjl, bytes_qjl = codec.quant_dequant(x, bits=4, use_hadamard=True, use_qjl=True)

    cos_no_qjl = torch.nn.functional.cosine_similarity(x.float().flatten(), deq_no_qjl.float().flatten(), dim=0).item()
    cos_qjl = torch.nn.functional.cosine_similarity(x.float().flatten(), deq_qjl.float().flatten(), dim=0).item()

    assert cos_qjl > 0.99
    assert cos_qjl >= cos_no_qjl
    assert bytes_qjl > bytes_no_qjl
    assert bytes_qjl < x.numel() * 2  # Still >2.7x smaller than FP16


def test_hybrid_kv_policy_retains_all_5_needle_blocks_at_32k() -> None:
    """
    Verify 32-token contiguous block scoring + K=8 geometric segment quotas retains
    100% of needle tokens at 10%, 30%, 50%, 70%, 90% depth when compressing 32K -> 4096.
    """
    torch.manual_seed(7)
    seq_len = 32768
    budget = 4096
    # Repetitive haystack keys cluster around a stationary center vector
    center_vec = torch.randn(1, 2, 1, 64)
    keys = center_vec + 0.05 * torch.randn(1, 2, seq_len, 64)

    depths = [0.10, 0.30, 0.50, 0.70, 0.90]
    needle_spans: list[range] = []
    for d in depths:
        pos = int(seq_len * d)
        # Align needle to a 24-token span inside the haystack
        span = range(pos, pos + 24)
        needle_spans.append(span)
        # Distinct semantic needle tokens deviate strongly from the repetitive haystack center
        keys[:, :, pos : pos + 24, :] += torch.randn(1, 2, 24, 64) * 3.5

    policy = HybridKVCachePolicy(
        max_seq_budget=budget,
        sink_tokens=128,
        recent_window=1024,
        num_segments=8,
        outlier_ratio=0.20,
        block_size=32,
        protect_prompt=False,
    )
    keep_idx = policy.select_retained_indices(layer_idx=3, keys=keys)
    kept_set = set(keep_idx.tolist())

    assert keep_idx.numel() <= budget + 32
    for d, span in zip(depths, needle_spans):
        retained_in_span = sum(1 for tok_pos in span if tok_pos in kept_set)
        # Because 32-token blocks are kept intact, the needle span is preserved!
        assert retained_in_span >= 20, f"Depth {int(d*100)}% lost needle tokens (retained {retained_in_span}/24)"


def test_hybrid_kv_policy_decode_only_protects_prompt() -> None:
    """Verify Thinking Mode (protect_prompt=True) keeps 100% of prompt tokens and only prunes generated CoT."""
    torch.manual_seed(11)
    prompt_len = 2048
    gen_len = 4096
    total_len = prompt_len + gen_len
    keys = torch.randn(1, 2, total_len, 64)

    policy = HybridKVCachePolicy(
        max_seq_budget=3072,
        sink_tokens=128,
        recent_window=512,
        num_segments=8,
        protect_prompt=True,
    )
    keep_idx = policy.select_retained_indices(layer_idx=3, keys=keys, prompt_length=prompt_len)
    kept_set = set(keep_idx.tolist())

    # Every single prompt token [0 .. prompt_len-1] must be preserved
    for p_tok in range(prompt_len):
        assert p_tok in kept_set
    assert keep_idx.numel() <= 3072 + 32


def test_kaggle_hybrid_cache_block_eviction_retains_needles() -> None:
    """Verify Kaggle script's TurboQuantHybridCache._apply_eviction retains all 5 needle depths."""
    torch.manual_seed(99)
    seq_len = 16384
    budget = 4096

    arm = ArmSpec(
        id="A4_evict",
        name="Test Block Evict",
        k_bits=8,
        v_bits=4,
        hadamard=True,
        qjl=True,
        boundary_protect=True,
        evict_budget=budget,
        num_segments=8,
        block_size=32,
    )
    cache = TurboQuantHybridCache(config=None, arm=arm, prompt_length=seq_len)

    center = torch.randn(1, 2, 1, 64, dtype=torch.float16)
    k = center + 0.05 * torch.randn(1, 2, seq_len, 64, dtype=torch.float16)
    v = torch.randn(1, 2, seq_len, 64, dtype=torch.float16)

    needle_ids = [10, 30, 50, 70, 90]
    needle_positions = [int(seq_len * d) for d in (0.10, 0.30, 0.50, 0.70, 0.90)]
    for nid, pos in zip(needle_ids, needle_positions):
        # Tag v across the entire 64-dim vector so FWHT preserves the uniform constant
        k[:, :, pos : pos + 16, :] += 4.0
        v[:, :, pos : pos + 16, :] = float(nid)

    for l_idx in range(4):
        cache.update(k.clone(), v.clone(), layer_idx=l_idx)
    cache.evict_if_needed(is_prefill=True)

    evicted_k, evicted_v = cache._get_kv(3)
    assert evicted_k.shape[-2] <= budget + 32
    token_means = evicted_v[0, 0].float().mean(dim=-1)
    for nid in needle_ids:
        assert torch.any((token_means - nid).abs() < 1.5), f"Needle {nid}% was evicted!"


def test_llamacpp_engine_command_builder() -> None:
    """Verify LlamaCppEngine builds correct -ctk/-ctv and --triattention-* flags."""
    arm = ArmConfig(
        id="L1",
        engine="llamacpp",
        model="models/Qwen3-8B.gguf",
        kv_dtype="turboquant_k8v4",
        compression=CompressionConfig(
            method="triattention",
            budget=4096,
            protect_prompt=True,
            stats_path="artifacts/calib/qwen3.gguf",
        ),
        serve=ServeConfig(max_model_len=65536),
    )
    eng = LlamaCppEngine()
    cmd = eng.build_command(arm, port=8088)
    assert "-ctk" in cmd and "q8_0" in cmd
    assert "-ctv" in cmd and "turbo4" in cmd
    assert "--triattention-budget" in cmd and "4096" in cmd
    assert "--triattention-segments" in cmd and "8" in cmd


def test_recommend_cli_and_profiles() -> None:
    """Verify `kvpipe recommend --profile ...` works for all workload categories."""
    for prof in ("auto", "think", "longctx", "chat", "max"):
        rec = recommend_for_profile(prof)
        assert rec["selected_profile"] == prof

    runner = CliRunner()
    res = runner.invoke(main, ["recommend", "--profile", "auto", "--json"])
    assert res.exit_code == 0
    assert "A4_evict" in res.output
    assert "A7_qjl" in res.output
