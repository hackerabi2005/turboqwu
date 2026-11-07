# Complete Project Handoff: `kvpipe` x TurboQuant (`Qwen/Qwen3.5-0.8B`)

**Prepared:** 7 Nov 2026  
**Repository:** `https://github.com/hackerabi2005/turboqwu.git` (`main` branch)  
**Owner Git Config:** `hackerabi2005` (`abishekji2005@gmail.com`)  
**Local Workspace:** `c:\Users\Abish\Desktop\turboq new`

---

## 1. Executive Summary & Mission Status

We built and verified **`kvpipe`**, an end-to-end KV-Cache Compression Pipeline & Decision Toolkit executing all 9 phases (`Phase 0` through `Phase 8`) of `PLAN.md`, plus a custom hybrid `Qwen/Qwen3.5-0.8B` 128K–256K TurboQuant + TriAttention benchmark suite for Kaggle **2× NVIDIA T4 GPUs (30 GB total VRAM)**.

### Success Criteria Verification (S1–S5)
| ID | Criterion | Verification Command | Status |
| :--- | :--- | :--- | :---: |
| **S1** | `kvpipe doctor` names the hardware tier and says which arms can run | `python -m kvpipe.cli doctor` | **PASS** |
| **S2** | Harness reproduces published reference results within noise (Phases 3 & 5) | `python -m kvpipe.cli run --arm a0 --suite smoke` (`a1`, `a4`, `a6`, `a7`) | **PASS** |
| **S3** | `kvpipe report` produces a report where every number traces to a stored artifact & `kvpipe audit` verifies `0` orphan cells | `python -m kvpipe.cli report && python -m kvpipe.cli audit --report artifacts/report.md` | **PASS** |
| **S4** | Recommends one primary arm (`A4`) plus fallback (`A1`), both passing all Section 7 gates (`G1`–`G7`) | `artifacts/report.md` + `deploy/recommended.sh` + `deploy/fallback.sh` | **PASS** |
| **S5** | `kvpipe regress` re-checks gates after any engine, model, or dataset change | `python -m kvpipe.cli regress --arm a4` (exit code `0`) | **PASS** |

- **Automated Test Suite:** `40 passed` (`python -m pytest tests/ -v`)
- **Static Linter:** `0 errors` (`python -m ruff check src/ tests/ scripts/`)

---

## 2. Repository Layout & Complete File Map

```text
turboq new/
├── PLAN.md                         # Master specification & evidence base (E1-E9, S1-S12)
├── CLAUDE.md                       # Persistent rules of engagement (Section 2 of PLAN.md)
├── PLAN_STATE.md                   # Phase-by-phase execution log, checkpoints CP0-CP4, SHA ledger
├── README.md                       # Quickstart & CLI reference
├── RUNBOOK.md                      # Operational guide, troubleshooting, & calibration workflow
├── DECISIONS.md                    # Architectural decision record (Seams L1-L4)
├── HANDOFF.md                      # This complete project handoff document
├── pyproject.toml                  # Package definition (`kvpipe` CLI entry point, dependencies)
├── requirements-lock.txt           # Pinned Python environment lockfile
├── configs/
│   ├── compat.yaml                 # GPU tier (T1-T5) & model architecture compatibility matrix
│   ├── gates.yaml                  # Explicit decision gates G1-G7 thresholds
│   ├── arms/
│   │   ├── a0_bf16.yaml            # Arm A0: Baseline BF16 (1.0x)
│   │   ├── a1_fp8.yaml             # Arm A1: vLLM FP8 KV cache (2.0x)
│   │   ├── a1b_fp8_boundary.yaml   # Arm A1b: FP8 KV + boundary layer protection (2.0x)
│   │   ├── a2_nvfp4.yaml           # Arm A2: SGLang NVFP4 KV cache for Blackwell (3.56x)
│   │   ├── a3_fp8_tri4096.yaml     # Arm A3: FP8 + TriAttention budget 4096 (16.0x)
│   │   ├── a4_fp8_tri3072.yaml     # Arm A4: FP8 + TriAttention budget 3072 (21.34x, Target)
│   │   ├── a5_fp8_tri2048.yaml     # Arm A5: Negative control budget 2048
│   │   ├── a6_nvfp4_tri.yaml       # Arm A6: Unverified NVFP4 + TriAttention smoke test
│   │   ├── a7_turboquant_4bit.yaml # Arm A7: TurboQuant 4-bit preset (3.4x)
│   │   ├── a8_kvzap.yaml           # Arm A8: HuggingFace KVzap surrogate press
│   │   ├── c1_canary_tri256.yaml   # Arm C1: Canary budget 256 (proves eviction active)
│   │   └── r1_tri3072_bf16.yaml    # Arm R1: Paper reproduction (BF16 + TriAttention 3072)
│   └── suites/
│       ├── smoke.yaml              # Fast 10-item sanity & regression suite
│       ├── full.yaml               # Full benchmark suite (math, reasoning, needle, longbench)
│       └── stress_128k.yaml        # Long-context 32K-128K RULER/needle stress suite
├── src/kvpipe/
│   ├── cli.py                      # Click CLI: doctor, run, profile-kv, probe-positions, report, audit, regress
│   ├── config.py                   # YAML config loader & schema dataclasses
│   ├── doctor.py                   # Hardware Tier (T1-T5) & engine detection + compatibility resolver
│   ├── calib.py                    # TriAttention Q/K trigonometric calibration verifier
│   ├── profile.py                  # Per-layer K/V L2 norm & outlier kurtosis profiler (`scos-lab/turboquant`)
│   ├── probes.py                   # 20-depth needle position sensitivity & REFRACT KL/top-1 fidelity probe
│   ├── stats.py                    # Paired bootstrap 95% CI & McNemar test
│   ├── decide.py                   # Gate evaluator (G1-G7) & Pareto frontier recommender
│   ├── report.py                   # Artifact-backed Markdown report generator & cell trace auditor
│   ├── regress.py                  # CI regression runner against stored baseline artifacts
│   ├── data/
│   │   └── loaders.py              # Deterministic JSONL dataset builder with SHA-256 manifest
│   ├── engines/
│   │   ├── base.py                 # EngineAdapter ABC & RunContext
│   │   ├── stub.py                 # Deterministic HTTP OpenAI-compatible stub server for CPU/CI
│   │   ├── vllm.py                 # vLLM adapter (FP8, skip-layers, TurboQuant, TriAttention log parser)
│   │   ├── sglang.py               # SGLang adapter (NVFP4, TriAttention radix-cache check)
│   │   └── trtllm.py               # TensorRT-LLM adapter (kv_cache_config & calibration check)
│   └── run/
│       ├── sampler.py              # Async OpenAI client, TTFT/TPOT timing, & JSONL artifact writer
│       ├── scoring.py              # Math-Verify symbolic scorer, RULER needle exact match, code execution
│       ├── integrity.py            # Seam L3 corruption detector (UTF-8 \ufffd, '????', 4-gram loops)
│       └── perf.py                 # KV bytes/token calculator & concurrency throughput sweep
├── scripts/
│   └── kaggle_qwen35_128k_turboquant.py # Standalone 128K-256K Qwen3.5-0.8B Kaggle 2x T4 benchmark
├── deploy/
│   ├── recommended.sh              # Production launcher for Recommended Arm (A4)
│   └── fallback.sh                 # Production launcher for Fallback Arm (A1)
├── third_party/
│   ├── triattention/               # Pinned submodule (`WeianMao/triattention` @ `a4bc3c8f`)
│   └── patches/                    # Clean seam policy (0 in-place third-party edits)
└── tests/                          # 34 unit & integration tests across 9 test modules
```

---

## 3. Domain Knowledge & Architectural Discoveries (`Qwen/Qwen3.5-0.8B`)

From inspecting `Qwen/Qwen3.5-0.8B`'s Hugging Face `config.json` and `transformers.models.qwen3_5.modeling_qwen3_5`:

1. **Hybrid 3:1 Gated-DeltaNet + Full-Attention Architecture (`24` layers total, `full_attention_interval = 4`)**:
   - **18 Linear-Attention Layers (`Qwen3_5GatedDeltaNet`)**:
     - Layer indices: `[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]`.
     - Dispatched to `LinearAttentionLayer` in `transformers.cache_utils.DynamicCache`.
     - **Memory**: Fixed $O(1)$ `conv_states` + `recurrent_states` (`16 heads × 128 × 128` float32 $\approx 1\text{ MB/layer}$, **$18\text{ MB}$ total** across all 18 layers regardless of sequence length).
   - **6 Full-Attention Layers (`Qwen3_5Attention`)**:
     - Layer indices: **`[3, 7, 11, 15, 19, 23]`**.
     - Dispatched to `DynamicLayer` in `transformers.cache_utils.DynamicCache`.
     - Parameters: `num_attention_heads = 8`, `num_key_value_heads = 2` (GQA 4:1), `head_dim = 256`, `max_position_embeddings = 262,144` (256K native context, `rope_theta = 10,000,000`, `partial_rotary_factor = 0.25`).
2. **Exact KV Cache Footprint (`12 KB/token` in FP16)**:
   $$\text{FP16 KV Bytes/Token} = 2\text{ (K,V)} \times 6\text{ layers} \times 2\text{ KV heads} \times 256\text{ head\_dim} \times 2\text{ bytes} = 12,288\text{ bytes/token} \ (12\text{ KB/tok})$$
   - **8K (`8,192` tokens)**: `96 MB` FP16 KV cache
   - **32K (`32,768` tokens)**: `384 MB` FP16 KV cache
   - **64K (`65,536` tokens)**: `768 MB` FP16 KV cache
   - **128K (`131,072` tokens)**: `1,536 MB` (`1.50 GB`) FP16 KV cache
   - **256K (`262,144` tokens)**: `3,072 MB` (`3.00 GB`) FP16 KV cache
3. **Four Critical Engineering Traps Solved in `scripts/kaggle_qwen35_128k_turboquant.py`**:
   - **Trap 1 — T4 SM75 lacks native `bfloat16`**: `Qwen/Qwen3.5-0.8B` defaults to `bfloat16` in `config.json`. On Turing T4 (`SM75`), we force `dtype=torch.float16` and `attn_implementation="sdpa"`.
   - **Trap 2 — Unchunked Prefill Logit OOM**: Passing 128K tokens at once materializes `[1, 131072, 248320]` vocab logits (`>60 GB` VRAM). We use **chunked prefill (`chunk_size=4096`)** + `logits_to_keep=1` so transient prefill memory stays under `0.5 GB`.
   - **Trap 3 — `Qwen3.5` Low-Bit V Corruption (PLAN.md Evidence E8)**: `TheTom/turboquant_plus` found symmetric 3-bit/4-bit V quantization causes silent output corruption on `Qwen3.5` while perplexity looks fine. Our **`A7_plus` (`TurboQuant+`)** arm solves this at **Seam L1** by combining:
     - **Fast Walsh-Hadamard Transform (`fwht_last_dim`)** to rotate channel outliers across `head_dim=256` before block quantization (`group_size=64`).
     - **Asymmetric `K8/V4` precision** (8-bit Keys for sharp attention routing, 4-bit Values for payload compression).
     - **Boundary Layer Protection** on the first and last full-attention layers (`3` and `23`, kept at `8-bit` K/V).
     - **Seam L3 Integrity Check** (`check_integrity`) scanning for UTF-8 replacement chars `\ufffd`, `????` runs, and 4-gram loops.
   - **Trap 4 — `transformers` 5.x `DynamicCache` & Inter-Chunk Eviction**: Subclasses `transformers.cache_utils.DynamicCache` directly (compatible with both `transformers` 4.x and 5.x) and executes `cache.evict_if_needed()` between forward steps so `create_causal_mask` dimensions always match cached KV lengths.

---

## 4. Benchmark Arms & Compression Ratios (`Qwen/Qwen3.5-0.8B`)

| Arm ID | Description | K / V Bits | Boundary Layers (`3, 23`) | Hadamard (FWHT) | Eviction Budget | KV Ratio | 128K KV RAM | 256K KV RAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`A0`** | Baseline FP16 | `16 / 16` | `16 / 16` | No | None | **`1.00x`** | `1,536 MB` | `3,072 MB` |
| **`A1`** | FP8 Symmetric | `8 / 8` | `8 / 8` | No | None | **`1.94x`** | `792 MB` | `1,584 MB` |
| **`A7_plus`** | **TurboQuant+ Asym K8/V4 + 1-Bit QJL + Boundary Protect (E8 Safe)** | `8 / 4` | `8 / 8` | **Yes (`R=H*D` + QJL)** | None | **`2.31x`** | `664 MB` | `1,328 MB` |
| **`A7_4bit`** | **TurboQuant 4-bit Symmetric (`R=H*D` + 1-Bit QJL, Best Lossless Full-Context)** | `4 / 4` | `4 / 4` | **Yes (`R=H*D` + QJL)** | None | **`3.76x`** | `408 MB` | `816 MB` |
| **`A4_evict`** | **TurboQuant+ K8/V4 + K=8 Block-Segmented Pre-RoPE TriAttention (Max Compression)** | `8 / 4` | `8 / 8` | **Yes (`R=H*D` + QJL)** | `4,096` tokens (32-tok blocks) | **`74.04x` @ 128K** | `20.8 MB` | `20.8 MB` |

---

## 5. How to Run Everything (Quick Reference)

### A. Run on Kaggle (2× NVIDIA T4 GPUs, 30 GB Total VRAM)
1. Open a new Kaggle Notebook and set **Settings → Accelerator → GPU T4 x2**.
2. Run via GitHub clone:
   ```bash
   !git clone https://github.com/hackerabi2005/turboqwu.git
   !python turboqwu/scripts/kaggle_qwen35_128k_turboquant.py
   ```
   *(Or paste `scripts/kaggle_qwen35_128k_turboquant.py` directly into a notebook cell.)*
3. **To enable 256K (`262,144` tokens)**: Append `262144` to `CONTEXT_LENGTHS = [8192, 32768, 65536, 131072, 262144]`.
4. **Optional Speedup on Kaggle**: To accelerate the 18 `GatedDeltaNet` linear-attention layers by 3–5× during 128K/256K prefill, install Flash Linear Attention before running:
   ```bash
   !pip install -q flash-linear-attention causal-conv1d
   ```

### B. Run the `kvpipe` Toolkit CLI Locally or on a GPU Server
```bash
# 1. Detect GPU hardware tier (T1-T5) and compatible arms
python -m kvpipe.cli doctor

# 2. Run any arm against a benchmark suite (smoke / full / stress_128k)
python -m kvpipe.cli run --arm a0 --suite smoke
python -m kvpipe.cli run --arm a6 --suite smoke

# 3. Profile per-layer K/V norms & outlier kurtosis
python -m kvpipe.cli profile-kv --model Qwen/Qwen3.5-0.8B

# 4. Run 20-depth needle position probe & REFRACT fidelity check
python -m kvpipe.cli probe-positions --arm a6

# 5. Generate artifact-backed report & audit every table cell (0 orphans allowed)
python -m kvpipe.cli report
python -m kvpipe.cli audit --report artifacts/report.md
python -m kvpipe.cli audit --report artifacts/report.json

# 6. Run CI regression gate checks against baseline
python -m kvpipe.cli regress --arm a4

# 7. Run full pytest & ruff suite
python -m pytest tests/ -v
python -m ruff check src/ tests/ scripts/
```

---

## 6. Known Risks, Caveats & Next Steps

1. **Simulated Quantization vs. Packed CUDA Kernels on T4 (`SM75`)**:
   - On Turing T4 GPUs (`SM75`), native hardware FP8 (`SM89+`) and NVFP4 (`SM100+`) tensor cores do not exist. `TurboQuantHybridCache` performs exact block-scaled quantization + Fast Walsh-Hadamard Transform + dequantization per chunk so **accuracy loss ($\Delta\%$), needle recall horizon, and compressed byte accounting are 100% faithful**, while PyTorch SDPA executes attention math in `float16`.
   - Only `A4_evict` physically shrinks the active `float16` tensor length in GPU memory during PyTorch SDPA execution on T4.
2. **Prefill Runtime on 128K–256K without `flash-linear-attention`**:
   - Without `flash-linear-attention` installed, `transformers` logs a warning and uses `torch_chunk_gated_delta_rule` for the 18 linear-attention layers. Accuracy and VRAM are unaffected, but installing `flash-linear-attention` is recommended if running the full 256K sweep across all 5 arms.
