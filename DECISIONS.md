# Architectural Decisions & Gate Rationale (DECISIONS.md)

## 1. Why Measured Ratios Over Nominal Ratios
Published claims of "35x to 80x" compression often multiply independent, theoretical layer ratios (e.g. 4x dtype × 10x eviction = 40x). In real serving engines:
- Prompt tokens must be preserved for accuracy (especially in document QA and agentic retrieval).
- KV cache page allocation granularity, block tables, and engine overhead bound true resident memory reduction.
- `kvpipe` only records and recommends based on measured resident tokens and measured capacity ratios from engine startup and runtime metrics.

## 2. Selection of Candidate Arms
- **A0 (BF16 Reference):** The baseline truth against which all degradation and capacity gains are paired.
- **A1 (FP8 KV):** The default production recommendation. Validated on Hopper and Blackwell with ~2x capacity gain and <2 point accuracy delta.
- **A2 (NVFP4 KV):** Blackwell upgrade path via SGLang. Provides ~1.78x reduction over FP8 on SM100/120.
- **R1 (TriAttention BF16):** Paper reproduction benchmark (budget 3072 matches Full Attention parity on AIME25).
- **A3 & A4 (FP8 + TriAttention):** Target reasoning candidate arms combining precision reduction with frequency-domain key eviction.
- **A5 (Negative Control, Budget 2048):** Validates that reducing budget below target threshold induces expected accuracy drops (~8 points on AIME25).
- **C1 (Canary, Budget 256):** Aggressive eviction proving that the eviction mechanism is physically operating and not a silent no-op.
- **A1b (Boundary Layer Protection):** Fallback safeguarding early and late attention layers if A1 regresses on sensitive architectures.

## 3. Rationale for Gates G1 through G7
- **G1 (Integrity):** Catches corrupted outputs (runs of question marks, replacement characters, truncated outputs, high n-gram repetition) that perplexity checks alone can mask.
- **G2 & G3 (Accuracy Margins):** Employs 10,000 paired bootstrap resamples over problems. Arms only PASS if the 95% CI lower bound is above `-margin` (-2.0 standard, -1.0 strict).
- **G4 (Eviction Active):** Requires verified applied eviction events (`applied > 0`) and confirmed drop on Canary C1.
- **G5 (Capacity Threshold):** Requires at least 1.8x measured gain over A0 to justify deployment complexity.
- **G6 (Throughput Target):** Guards latency; requires throughput >= 70% of A1 baseline with acceptable p95 TPOT.
- **G7 (Stability):** 3x repeat standard deviation <= 1.5 points to prevent noisy recommendations.
