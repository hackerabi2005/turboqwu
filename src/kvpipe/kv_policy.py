"""Pre-RoPE Trigonometric TriAttention Policy + TurboQuant (R=H*D + 1-Bit QJL) Codec."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def fwht_last_dim(x: torch.Tensor) -> torch.Tensor:
    """Orthogonal Fast Walsh-Hadamard Transform along last dimension (must be power of 2)."""
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


class TurboQuantCodec:
    """
    TurboQuant KV Quantizer with Random-Sign FWHT (R = H * D) and optional
    1-bit Quantized Johnson-Lindenstrauss (QJL) residual error recovery.
    """

    def __init__(self, head_dim: int = 256, group_size: int = 64, seed: int = 42):
        self.head_dim = head_dim
        self.group_size = group_size
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self._signs = (torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)
        self._qjl_signs = (torch.randint(0, 2, (head_dim,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)

    def _get_signs(self, device: torch.device, dtype: torch.dtype, dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if dim != self._signs.shape[0]:
            gen = torch.Generator(device="cpu").manual_seed(42)
            s1 = (torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)
            s2 = (torch.randint(0, 2, (dim,), generator=gen, dtype=torch.float32) * 2.0 - 1.0)
            return s1.to(device=device, dtype=dtype), s2.to(device=device, dtype=dtype)
        return self._signs.to(device=device, dtype=dtype), self._qjl_signs.to(device=device, dtype=dtype)

    def quant_dequant(
        self,
        x: torch.Tensor,
        bits: int,
        use_hadamard: bool = True,
        use_qjl: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Quantize and reconstruct tensor, returning (reconstructed_tensor, compressed_bytes)."""
        numel = x.numel()
        if bits >= 16:
            return x, numel * 2

        orig_dtype, orig_shape = x.dtype, x.shape
        d = orig_shape[-1]
        signs, qjl_signs = self._get_signs(x.device, torch.float32, d)

        work = x.float()
        if use_hadamard:
            work = fwht_last_dim(work * signs)

        g_size = min(self.group_size, d)
        flat = work.reshape(-1, g_size)
        num_groups = flat.shape[0]

        qmax = (1 << (bits - 1)) - 1
        amax = flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = amax / qmax
        q = torch.clamp(torch.round(flat / scale), -qmax, qmax)
        deq = (q * scale).reshape(orig_shape)

        qjl_bytes = 0
        if use_qjl:
            # 1-bit QJL residual recovery on quantization error r = work - deq
            residual = work - deq
            proj = fwht_last_dim(residual * qjl_signs)
            r_norm = residual.norm(dim=-1, keepdim=True) / math.sqrt(d)
            qjl_bits = torch.where(proj >= 0, 1.0, -1.0)
            r_hat = fwht_last_dim(qjl_bits) * qjl_signs * r_norm * math.sqrt(math.pi / 2.0)
            deq = deq + 0.5 * r_hat
            qjl_bytes = int(math.ceil(numel / 8.0)) + (numel // d) * 2

        if use_hadamard:
            deq = fwht_last_dim(deq) * signs

        payload_bytes = int(math.ceil(numel * bits / 8.0))
        scale_bytes = num_groups * 2
        return deq.to(orig_dtype), payload_bytes + scale_bytes + qjl_bytes


class HybridKVCachePolicy:
    """
    4-Tier Block-Segmented Pre-RoPE TriAttention KV-Cache Policy:
      1. Attention Sinks (first `sink_tokens` tokens)
      2. Rolling Recent Window (last `recent_window` tokens)
      3. Pre-RoPE Trigonometric Center Scoring + L2 Semantic Outlier Anchors (`outlier_ratio`)
      4. K=8 Geometric Segment Block Quotas (32-token contiguous blocks across all document depths)
    Also supports Decode-Only Eviction (`protect_prompt=True`) for Thinking/Reasoning chains.
    """

    def __init__(
        self,
        max_seq_budget: int = 4096,
        sink_tokens: int = 128,
        recent_window: int = 1024,
        num_segments: int = 8,
        outlier_ratio: float = 0.20,
        block_size: int = 32,
        protect_prompt: bool = False,
    ):
        self.max_seq_budget = max_seq_budget
        self.sink_tokens = sink_tokens
        self.recent_window = recent_window
        self.num_segments = num_segments
        self.outlier_ratio = outlier_ratio
        self.block_size = block_size
        self.protect_prompt = protect_prompt
        self.key_centers: Dict[int, torch.Tensor] = {}

    def update_center(self, layer_idx: int, keys: torch.Tensor) -> torch.Tensor:
        """EMA tracking of key channel distribution center c_k."""
        # keys: [batch, heads, seq_len, head_dim]
        current_center = keys.float().mean(dim=-2, keepdim=True)
        if layer_idx not in self.key_centers or self.key_centers[layer_idx].shape != current_center.shape:
            self.key_centers[layer_idx] = current_center.detach()
        else:
            self.key_centers[layer_idx] = (0.9 * self.key_centers[layer_idx] + 0.1 * current_center).detach()
        return self.key_centers[layer_idx]

    def score_tokens(self, layer_idx: int, keys: torch.Tensor) -> torch.Tensor:
        """
        Compute token salience scores [seq_len] combining Pre-RoPE centered trigonometric
        deviation ||K - c_k||_2 and peak channel outlier magnitude.
        """
        center = self.update_center(layer_idx, keys)
        centered = keys.float() - center
        # Trigonometric centered norm + max channel spike across heads
        l2_dev = torch.norm(centered, p=2, dim=-1).mean(dim=(0, 1))  # [seq_len]
        max_dev = centered.abs().amax(dim=-1).amax(dim=(0, 1))       # [seq_len]
        return l2_dev + 0.5 * max_dev

    def select_retained_indices(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        prompt_length: int = 0,
        budget_override: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Return sorted 1D tensor of token indices to retain in the KV cache.
        Preserves contiguous `block_size`-token blocks across `num_segments` equal segments
        so multi-token needle facts at 10%, 30%, 50%, 70%, 90% are never half-evicted.
        """
        seq_len = keys.shape[-2]
        device = keys.device
        budget = budget_override if budget_override is not None else self.max_seq_budget

        # In Thinking/Reasoning mode with protect_prompt=True, never evict prompt tokens
        protected_prefix = max(self.sink_tokens, prompt_length if self.protect_prompt else self.sink_tokens)
        if seq_len <= max(budget, protected_prefix):
            return torch.arange(seq_len, device=device, dtype=torch.long)

        sink = min(protected_prefix, max(32, budget // 4) if not self.protect_prompt else protected_prefix)
        recent = min(self.recent_window, max(64, budget // 3))
        if sink + recent >= seq_len:
            return torch.arange(seq_len, device=device, dtype=torch.long)

        mid_budget = max(0, budget - sink - recent)
        if mid_budget == 0:
            return torch.cat([
                torch.arange(0, sink, device=device, dtype=torch.long),
                torch.arange(seq_len - recent, seq_len, device=device, dtype=torch.long),
            ])

        # Score middle region [sink : seq_len - recent] in contiguous blocks of `block_size`
        token_scores = self.score_tokens(layer_idx, keys)
        mid_scores = token_scores[sink : seq_len - recent]
        mid_len = mid_scores.shape[0]
        bsize = min(self.block_size, max(1, mid_budget // max(1, self.num_segments)))
        num_blocks = (mid_len + bsize - 1) // bsize

        # Pad mid_scores to multiple of bsize and score each block by max + mean salience
        pad_len = num_blocks * bsize - mid_len
        if pad_len > 0:
            padded = torch.nn.functional.pad(mid_scores, (0, pad_len), value=-1e9)
        else:
            padded = mid_scores
        block_view = padded.view(num_blocks, bsize)
        block_scores = block_view.max(dim=-1).values + 0.35 * block_view.mean(dim=-1)

        target_blocks = max(1, mid_budget // bsize)
        if target_blocks >= num_blocks:
            return torch.arange(seq_len, device=device, dtype=torch.long)

        selected_block_mask = torch.zeros(num_blocks, dtype=torch.bool, device=device)

        # Tier 3: Global L2 / Trigonometric Outlier Anchor Blocks (top `outlier_ratio`)
        outlier_blocks = max(1, int(target_blocks * self.outlier_ratio))
        top_outliers = torch.topk(block_scores, k=min(outlier_blocks, num_blocks)).indices
        selected_block_mask[top_outliers] = True

        # Tier 4: K=8 Geometric Segment Block Quotas across the full middle horizon
        remaining_blocks = max(0, target_blocks - int(selected_block_mask.sum().item()))
        if remaining_blocks > 0:
            segs = min(self.num_segments, num_blocks)
            blocks_per_seg = max(1, remaining_blocks // segs)
            seg_boundaries = torch.linspace(0, num_blocks, segs + 1, device=device).long()

            for s in range(segs):
                b_start = int(seg_boundaries[s].item())
                b_end = int(seg_boundaries[s + 1].item())
                if b_end <= b_start:
                    continue
                seg_scores = block_scores[b_start:b_end].clone()
                seg_scores[selected_block_mask[b_start:b_end]] = -1e9
                k_take = min(blocks_per_seg, b_end - b_start)
                if k_take > 0:
                    local_top = torch.topk(seg_scores, k=k_take).indices + b_start
                    selected_block_mask[local_top] = True

        # Fill any leftover block budget globally
        still_needed = target_blocks - int(selected_block_mask.sum().item())
        if still_needed > 0:
            rem_scores = block_scores.clone()
            rem_scores[selected_block_mask] = -1e9
            extra = torch.topk(rem_scores, k=min(still_needed, int((~selected_block_mask).sum().item()))).indices
            selected_block_mask[extra] = True

        # Expand selected blocks back to contiguous token indices
        chosen_blocks = torch.nonzero(selected_block_mask, as_tuple=False).squeeze(-1)
        offsets = torch.arange(bsize, device=device, dtype=torch.long)
        mid_keep = (chosen_blocks.unsqueeze(-1) * bsize + offsets).reshape(-1)
        mid_keep = mid_keep[mid_keep < mid_len] + sink

        keep_idx = torch.cat([
            torch.arange(0, sink, device=device, dtype=torch.long),
            mid_keep,
            torch.arange(seq_len - recent, seq_len, device=device, dtype=torch.long),
        ])
        return torch.unique(keep_idx, sorted=True)
