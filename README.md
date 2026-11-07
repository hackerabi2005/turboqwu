# kvpipe: KV-Cache Compression Pipeline & Decision Toolkit

`kvpipe` is an evaluation, verification, and decision toolkit for KV-cache compression across LLM inference engines (vLLM, SGLang, TensorRT-LLM) and hardware tiers (Blackwell, Hopper, Ada/Ampere).

---

## Quickstart

### 1. Environment Discovery (`kvpipe doctor`)
Detects local GPUs, compute capability, inference engine packages, and hardware tier (T1 to T4):
```bash
kvpipe doctor
```

### 2. TriAttention Calibration (`kvpipe calibrate`)
Resolves precomputed frequency statistics for verified models or calibrates custom models:
```bash
kvpipe calibrate --model Qwen/Qwen3-8B
```

### 3. Run Benchmark Suite on an Arm (`kvpipe run`)
Executes streaming evaluation against an arm with automatic TTFT/TPOT tracking and append-only caching:
```bash
kvpipe run --arm A4 --suite aime25
```

### 4. Evaluate Gates & Generate Decision Report (`kvpipe report`)
Evaluates gates G1–G7 across candidate arms, selects the highest-compression passing arm, identifies a safe fallback, and generates `artifacts/report.md`, `artifacts/report.json`, and deployment scripts:
```bash
kvpipe report
```

### 5. Audit Claims & Metrics (`kvpipe audit`)
Verifies that every single reported number and metric traces directly to an existing run artifact:
```bash
kvpipe audit
```

### 6. Regression Testing (`kvpipe regress`)
Re-checks golden prompts, canary eviction degradation, and accuracy suites after any engine or model update:
```bash
kvpipe regress --arm A1
```

---

## Hardware Tiers

| Tier | Target Hardware | Supported Engines & Features |
|---|---|---|
| **T1** | Datacenter Blackwell (SM100/103, B200) | TensorRT-LLM TriAttention, SGLang NVFP4 KV, vLLM |
| **T2** | Consumer / Workstation Blackwell (SM120/121, RTX 5090) | vLLM FP8 & TriAttention plugin, SGLang NVFP4 KV |
| **T3** | Hopper (SM90, H100 / H200) | vLLM FP8 KV (validated) |
| **T4** | Ada / Ampere (SM80–SM89, RTX 4090 / A100) | vLLM FP8 KV (experimental), BF16 baseline |
| **None/Stub** | CPU / Environments without NVIDIA GPU | Built-in StubEngine for CI/CD and offline verification |
