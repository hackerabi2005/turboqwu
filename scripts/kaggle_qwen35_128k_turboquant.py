#!/usr/bin/env python3
"""
kvpipe x TurboQuant: Qwen/Qwen3.5-0.8B 128K-256K Context & Recall Horizon Benchmark
Target Hardware: Kaggle 2x NVIDIA T4 GPUs (30 GB total VRAM, SM75 Turing -> float16)
"""

from __future__ import annotations

import gc
import math
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

MODEL_ID = "Qwen/Qwen3.5-0.8B"
FULL_ATTN_LAYERS = (3, 7, 11, 15, 19, 23)
BOUNDARY_FULL_ATTN_LAYERS = (3, 23)  # First & last full-attention layers (PLAN.md E8)
GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# 1. Fast Walsh-Hadamard Transform (FWHT) + Block Quantizers (TurboQuant L1)
# ---------------------------------------------------------------------------
def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal Fast Walsh-Hadamard Transform along last dim (head_dim=256)."""
    orig_shape = x.shape
    d = orig_shape[-1]
    y = x.reshape(-1, d).float()
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        a, b = y[:, :, 0, :], y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2)
        h *= 2
    return (y.view(orig_shape) / math.sqrt(d)).to(x.dtype)


def quant_dequant_tensor(
    x: torch.Tensor,
    bits: int,
    use_hadamard: bool = False,
    group_size: int = GROUP_SIZE,
) -> tuple[torch.Tensor, int]:
    """
    Simulates exact hardware group-wise quantization (with optional Hadamard rotation)
    and returns (dequantized_tensor, compressed_bytes).
    """
    numel = x.numel()
    if bits >= 16:
        return x, numel * 2

    orig_dtype, orig_shape = x.dtype, x.shape
    work = fwht_last_dim(x) if use_hadamard else x

    flat = work.reshape(-1, group_size).float()
    num_groups = flat.shape[0]

    qmax = (1 << (bits - 1)) - 1
    amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    scale = amax / qmax
    q = torch.clamp(torch.round(flat / scale), -qmax, qmax)
    deq = (q * scale).reshape(orig_shape).to(orig_dtype)

    if use_hadamard:
        deq = fwht_last_dim(deq)  # FWHT is symmetric & self-inverse

    payload_bytes = int(math.ceil(numel * bits / 8.0))
    scale_bytes = num_groups * 2
    return deq, payload_bytes + scale_bytes


# ---------------------------------------------------------------------------
# 2. Custom Qwen3.5 Hybrid TurboQuant + TriAttention Cache (Seams L1 & L2)
# ---------------------------------------------------------------------------
@dataclass
class ArmSpec:
    id: str
    name: str
    k_bits: int
    v_bits: int
    hadamard: bool = False
    boundary_protect: bool = False  # Keep layers 3 & 23 at FP8 (8-bit)
    evict_budget: int | None = None  # If set, keep sink + recent + high-norm K tokens


class TurboQuantHybridCache(DynamicCache):
    """
    Drop-in replacement for transformers.cache_utils.DynamicCache that intercepts
    full-attention layers (3, 7, 11, 15, 19, 23) while keeping the 18 GatedDeltaNet
    linear-attention layers intact across both transformers 4.x and 5.x.
    """

    def __init__(self, config: Any, arm: ArmSpec):
        super().__init__(config=config)
        self.arm = arm
        self.compressed_kv_bytes: dict[int, int] = {}
        self.uncompressed_kv_bytes: dict[int, int] = {}
        self.eviction_events: int = 0

    def _layer_bits(self, layer_idx: int) -> tuple[int, int]:
        if self.arm.boundary_protect and layer_idx in BOUNDARY_FULL_ATTN_LAYERS:
            return max(self.arm.k_bits, 8), max(self.arm.v_bits, 8)
        return self.arm.k_bits, self.arm.v_bits

    def _get_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if hasattr(self, "layers") and layer_idx < len(self.layers):
            return self.layers[layer_idx].keys, self.layers[layer_idx].values
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _set_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if hasattr(self, "layers") and layer_idx < len(self.layers):
            self.layers[layer_idx].keys = k
            self.layers[layer_idx].values = v
        else:
            self.key_cache[layer_idx] = k
            self.value_cache[layer_idx] = v

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        k_bits, v_bits = self._layer_bits(layer_idx)

        q_k, k_bytes = quant_dequant_tensor(key_states, k_bits, use_hadamard=self.arm.hadamard)
        q_v, v_bytes = quant_dequant_tensor(value_states, v_bits, use_hadamard=self.arm.hadamard)

        keys, values = super().update(q_k, q_v, layer_idx, *args, **kwargs)

        self.compressed_kv_bytes[layer_idx] = self.compressed_kv_bytes.get(layer_idx, 0) + k_bytes + v_bytes
        self.uncompressed_kv_bytes[layer_idx] = (
            self.uncompressed_kv_bytes.get(layer_idx, 0) + (key_states.numel() + value_states.numel()) * 2
        )

        return keys, values

    def evict_if_needed(self) -> None:
        """Run KV eviction between forward steps so causal mask dimensions always match KV length."""
        if self.arm.evict_budget is None:
            return
        for layer_idx in FULL_ATTN_LAYERS:
            if layer_idx < len(self.layers) or (hasattr(self, "key_cache") and layer_idx < len(self.key_cache)):
                k, _ = self._get_kv(layer_idx)
                if k is not None and k.numel() > 0 and k.shape[-2] > self.arm.evict_budget:
                    self._apply_eviction(layer_idx, self.arm.evict_budget)

    def _apply_eviction(self, layer_idx: int, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self._get_kv(layer_idx)
        seq_len = k.shape[-2]
        if seq_len <= budget:
            return k, v

        sink = min(128, budget // 4)
        recent = min(1024, budget // 2)
        mid_budget = max(0, budget - sink - recent)

        if mid_budget > 0 and seq_len > (sink + recent):
            mid_k = k[:, :, sink : seq_len - recent, :]
            scores = mid_k.float().norm(dim=-1).mean(dim=1)  # [batch, mid_len]
            topk_idx = scores.topk(mid_budget, dim=-1).indices.sort(dim=-1).values[0] + sink
            keep_idx = torch.cat(
                [
                    torch.arange(0, sink, device=k.device),
                    topk_idx,
                    torch.arange(seq_len - recent, seq_len, device=k.device),
                ],
                dim=0,
            )
        else:
            keep_idx = torch.cat(
                [
                    torch.arange(0, sink, device=k.device),
                    torch.arange(seq_len - (budget - sink), seq_len, device=k.device),
                ],
                dim=0,
            )

        new_k = k.index_select(-2, keep_idx)
        new_v = v.index_select(-2, keep_idx)
        self._set_kv(layer_idx, new_k, new_v)
        self.eviction_events += 1

        k_bits, v_bits = self._layer_bits(layer_idx)
        retained_numel = new_k.numel()
        num_groups = max(1, retained_numel // GROUP_SIZE)
        k_b = int(math.ceil(retained_numel * k_bits / 8.0)) + num_groups * 2 if k_bits < 16 else retained_numel * 2
        v_b = int(math.ceil(retained_numel * v_bits / 8.0)) + num_groups * 2 if v_bits < 16 else retained_numel * 2
        self.compressed_kv_bytes[layer_idx] = k_b + v_b
        return new_k, new_v

    def summary_metrics(self) -> dict[str, float]:
        raw_b = sum(self.uncompressed_kv_bytes.values())
        comp_b = sum(self.compressed_kv_bytes.values())
        return {
            "fp16_kv_mb": raw_b / (1024**2),
            "compressed_kv_mb": comp_b / (1024**2),
            "kv_compression_ratio": raw_b / max(comp_b, 1),
            "eviction_events": float(self.eviction_events),
        }


# ---------------------------------------------------------------------------
# 3. Seam L3: Integrity Check (from kvpipe/run/integrity.py, Evidence E8)
# ---------------------------------------------------------------------------
def check_integrity(text: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if "\ufffd" in text:
        reasons.append("utf8_replacement_char")
    if "????" in text:
        reasons.append("question_mark_run")
    tokens = text.split()
    if len(tokens) >= 4:
        ngrams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
        counts = Counter(ngrams)
        if counts and counts.most_common(1)[0][1] > 5:
            reasons.append("4gram_repetition")
    return len(reasons) == 0, reasons


# ---------------------------------------------------------------------------
# 4. Multi-Needle Haystack Builder + Chunked Prefill Engine (128K-256K Safe)
# ---------------------------------------------------------------------------
def build_multi_needle_prompt(
    tokenizer: Any,
    target_tokens: int,
    depths: list[float] = [0.10, 0.30, 0.50, 0.70, 0.90],
) -> tuple[torch.Tensor, dict[float, str]]:
    haystack_sentence = (
        "Distributed systems rely on consensus protocols, bounded replication logs, "
        "and deterministic state machines to preserve fault tolerance under network partitions. "
    )
    block_ids = tokenizer.encode(haystack_sentence, add_special_tokens=False)

    needles: dict[float, str] = {}
    needle_ids_map: dict[float, list[int]] = {}
    for d in depths:
        code = f"TQ-{int(d * 100):02d}-{max(1, target_tokens // 1024)}K"
        needles[d] = code
        needle_str = (
            f"\n[CRITICAL ARCHIVE FACT AT DEPTH {int(d * 100)}%]: "
            f"The secret vault authorization key for sector {int(d * 100)} is {code}.\n"
        )
        needle_ids_map[d] = tokenizer.encode(needle_str, add_special_tokens=False)

    question_str = (
        "\n\n=== VERIFICATION QUERY ===\n"
        "List the exact secret vault authorization keys for sectors 10, 30, 50, 70, and 90 "
        "in order, separated by commas.\n"
    )
    bos_ids = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    q_ids = tokenizer.encode(
        question_str + "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\nThe keys are ",
        add_special_tokens=False,
    )

    reserved = len(bos_ids) + len(q_ids) + sum(len(v) for v in needle_ids_map.values())
    filler_budget = max(64, target_tokens - reserved)

    repeats = (filler_budget // len(block_ids)) + 2
    filler = (block_ids * repeats)[:filler_budget]

    for d in sorted(depths, reverse=True):
        pos = min(len(filler), max(0, int(len(filler) * d)))
        filler[pos:pos] = needle_ids_map[d]

    full_ids = bos_ids + filler + q_ids
    return torch.tensor([full_ids], dtype=torch.long), needles


@torch.inference_mode()
def evaluate_arm_at_context(
    model: Any,
    tokenizer: Any,
    arm: ArmSpec,
    context_len: int,
    chunk_size: int = 4096,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    input_ids, needles = build_multi_needle_prompt(tokenizer, context_len)
    seq_len = input_ids.shape[1]
    first_device = next(model.parameters()).device

    text_config = getattr(model.config, "text_config", model.config)
    cache = TurboQuantHybridCache(text_config, arm=arm)

    t0 = time.perf_counter()
    # Chunked prefill so 128K-256K sequences never OOM on 2x T4 GPUs
    for start in range(0, seq_len - 1, chunk_size):
        end = min(start + chunk_size, seq_len - 1)
        chunk = input_ids[:, start:end].to(first_device)
        out = model(
            input_ids=chunk,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache.evict_if_needed()
        del out, chunk

    # Final prefill token + greedy decoding loop
    next_token = input_ids[:, -1:].to(first_device)
    generated_ids: list[int] = []
    eos_id = getattr(tokenizer, "eos_token_id", None)

    for _ in range(max_new_tokens):
        out = model(
            input_ids=next_token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = int(next_token.item())
        generated_ids.append(tok_id)
        cache.evict_if_needed()
        if eos_id is not None and tok_id == eos_id:
            break

    elapsed = time.perf_counter() - t0
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    depth_hits: dict[str, bool] = {}
    for d, code in needles.items():
        depth_hits[f"{int(d * 100)}%"] = code in output_text

    recall_score = sum(depth_hits.values()) / max(len(depth_hits), 1) * 100.0
    integrity_ok, integrity_issues = check_integrity(output_text)
    cache_stats = cache.summary_metrics()

    total_vram_gb = 0.0
    if torch.cuda.is_available():
        for dev_id in range(torch.cuda.device_count()):
            total_vram_gb += torch.cuda.max_memory_allocated(dev_id) / (1024**3)

    del cache, input_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "arm_id": arm.id,
        "arm_name": arm.name,
        "context_tokens": seq_len,
        "recall_pct": recall_score,
        "depth_hits": depth_hits,
        "integrity_ok": integrity_ok,
        "integrity_issues": integrity_issues,
        "fp16_kv_mb": cache_stats["fp16_kv_mb"],
        "compressed_kv_mb": cache_stats["compressed_kv_mb"],
        "compression_ratio": cache_stats["kv_compression_ratio"],
        "peak_vram_gb": total_vram_gb,
        "time_s": elapsed,
        "tok_per_s": seq_len / max(elapsed, 1e-3),
        "sample_output": output_text[:120],
    }


def main() -> None:
    print("=" * 88)
    print(f"kvpipe x TurboQuant: {MODEL_ID} Long-Context Recall & Compression Suite")
    print("=" * 88)
    print(f"Transformers: {transformers.__version__} | PyTorch: {torch.__version__}")
    num_gpus = torch.cuda.device_count()
    print(f"Detected GPUs: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / (1024**3):.2f} GB, SM{props.major}{props.minor})")

    print("\nLoading tokenizer & model in float16 across available GPUs (device_map='auto')...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    arms = [
        ArmSpec(id="A0", name="Baseline FP16 (K16/V16)", k_bits=16, v_bits=16),
        ArmSpec(id="A1", name="FP8 Symmetric (K8/V8)", k_bits=8, v_bits=8),
        ArmSpec(
            id="A7_plus",
            name="TurboQuant+ Asym K8/V4 + Boundary Protect (E8 Safe)",
            k_bits=8,
            v_bits=4,
            hadamard=True,
            boundary_protect=True,
        ),
        ArmSpec(
            id="A7_4bit",
            name="TurboQuant 4-bit Symmetric (K4/V4 + FWHT)",
            k_bits=4,
            v_bits=4,
            hadamard=True,
            boundary_protect=False,
        ),
        ArmSpec(
            id="A4_evict",
            name="TurboQuant+ K8/V4 + KV Eviction (Budget=4096)",
            k_bits=8,
            v_bits=4,
            hadamard=True,
            boundary_protect=True,
            evict_budget=4096,
        ),
    ]

    # Add 262144 to context_lengths to test full 256K native horizon
    context_lengths = [8192, 32768, 65536, 131072]

    all_results: list[dict[str, Any]] = []
    for ctx_len in context_lengths:
        print(f"\n--- Testing Context Length: {ctx_len:,} tokens ({ctx_len // 1024}K) ---")
        for arm in arms:
            res = evaluate_arm_at_context(model, tokenizer, arm, context_len=ctx_len)
            all_results.append(res)
            depths_str = " ".join(f"{k}:{'OK' if v else 'MISS'}" for k, v in res["depth_hits"].items())
            print(
                f"[{res['arm_id']:8s}] {res['context_tokens']//1024:3d}K | "
                f"KV: {res['fp16_kv_mb']:6.1f}MB -> {res['compressed_kv_mb']:5.1f}MB ({res['compression_ratio']:5.2f}x) | "
                f"Recall: {res['recall_pct']:5.1f}% ({depths_str}) | "
                f"Integrity: {'PASS' if res['integrity_ok'] else 'FAIL'} | "
                f"VRAM: {res['peak_vram_gb']:4.2f}GB | {res['time_s']:5.1f}s"
            )

    print("\n" + "=" * 100)
    print("FINAL KVPIPE DECISION REPORT: COMPRESSION RATIO vs. RECALL HORIZON & ACCURACY LOSS")
    print("=" * 100)
    print(
        f"{'Arm':10s} | {'Ctx':5s} | {'FP16 KV':9s} | {'Comp KV':9s} | {'Ratio':7s} | "
        f"{'Recall':7s} | {'Acc Loss':8s} | {'Integrity':9s} | {'Peak VRAM':9s}"
    )
    print("-" * 100)

    baseline_by_ctx = {
        r["context_tokens"]: r["recall_pct"] for r in all_results if r["arm_id"] == "A0"
    }
    for r in all_results:
        base_acc = baseline_by_ctx.get(r["context_tokens"], r["recall_pct"])
        acc_loss = r["recall_pct"] - base_acc
        print(
            f"{r['arm_id']:10s} | {r['context_tokens']//1024:4d}K | "
            f"{r['fp16_kv_mb']:7.1f}MB | {r['compressed_kv_mb']:7.1f}MB | "
            f"{r['compression_ratio']:6.2f}x | {r['recall_pct']:6.1f}% | "
            f"{acc_loss:+7.1f}% | {'PASS' if r['integrity_ok'] else 'FAIL':9s} | "
            f"{r['peak_vram_gb']:7.2f}GB"
        )


if __name__ == "__main__":
    main()
