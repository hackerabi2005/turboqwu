# KV-Cache Compression Pipeline: Build Plan for Claude Code
Version 1.0 | Prepared 4 Nov 2026 | For Claude Code (autonomous coding agent) and the project owner
How to use: save this file as PLAN.md at the root of a new repo, copy section 2 into CLAUDE.md so the rules persist across sessions, start Claude Code and say: Read PLAN.md and execute Phase 0.

## 1. Mission and success criteria
Build `kvpipe`, a small Python toolkit that (1) detects the GPU and inference-engine environment, (2) launches candidate KV-cache configurations (called arms), (3) measures accuracy loss, real compression and speed on the owner's own model and workload, and (4) recommends the highest-compression arm that passes explicit accuracy gates.

Why it exists: published ratios such as 35x to 80x are products of separately measured ratios, not end-to-end results. The only number worth trusting is one measured on this hardware, with this model and workload.

Success criteria:
• S1. `kvpipe doctor` names the hardware tier and says which arms can run.
• S2. The harness reproduces two published reference results within noise (Phases 3 and 5).
• S3. `kvpipe report` produces a report in which every number traces to a stored run artifact.
• S4. The report recommends one arm plus a fallback, both passing the gates in section 7.
• S5. `kvpipe regress` re-checks the gates after any engine, model or dataset change.

## 2. Rules of engagement (read first)
1. Work phase by phase. Start phase N+1 only after phase N's Done-when checks pass. After every task, log what ran, the command, the result file and the status in PLAN_STATE.md so a fresh session can resume.
2. Stop and ask the owner at checkpoints CP0 to CP4. Never guess the owner's workload, accuracy tolerance or GPU budget.
3. Never state a number you did not measure. Label unmeasured values as expected and name the source from section 3.
4. Pin everything: model and dataset revisions, third-party git commits, package versions (lockfile) and seeds.
5. Before any download over 5 GB or GPU job over 30 minutes, state the estimate and get confirmation. Run a small pilot first and extrapolate from it.
6. Do not enable trust_remote_code for models the owner has not named. The official TriAttention calibrator uses it.
7. Never edit third-party repos in place. Keep patches in third_party/patches/ with a written reason.
8. Depend only on the projects in section 3. Projects in section 9 are not dependencies.
9. Every result file records engine and version, model and revision, weight dtype, effective KV dtype, compression settings, git SHA, seed and timestamp.
10. Make small commits, each with tests. If P0.2 finds no NVIDIA GPU, stop and ask whether to use a remote GPU machine or to build only Phases 1, 3 and 7 against the stub engine.

## 3. Evidence base (what this plan relies on)
Read from primary sources on 4 Nov 2026; the author ran no benchmarks. Phase 0 re-checks these. Treat every figure as a hypothesis to reproduce, not a result. Source codes (S1 to S12) are listed in Appendix C.

| ID | Fact | Source |
|---|---|---|
| E1 | vLLM FP8 KV cache (--kv-cache-dtype fp8): 2x capacity, no throughput cost, above 98% accuracy recovery on two reasoning models. Caveats: head_dim 256 models can regress in prefill; skip sliding-window layers on hybrid models; a Hopper long-context FP8 bug was fixed in 2026, so use a recent build. | S1, S2, S12 |
| E2 | vLLM built-in TurboQuant presets: turboquant_k8v4, turboquant_4bit_nc, turboquant_k3v4_nc, turboquant_3bit_nc. The 4bit_nc preset gives up to 3.4x capacity for 1 to 4 points of accuracy and about 25% less throughput. The 3-bit presets lose up to about 20 points on AIME25 and LiveCodeBench. Hybrid and sliding-window models are unsupported. | S1 |
| E3 | SGLang NVFP4 KV (--kv-cache-dtype nvfp4): about 1.78x smaller than FP8; within 1.6 points of FP8 on GSM8K, GPQA-Diamond, AIME25 and SWE-bench for two Qwen models. Experimental. Needs SM12x or SM100/103, GQA or sparse-MLA models, and an FP8-weight checkpoint in the tested setup. | S3 |
| E4 | TriAttention paper, Qwen3-8B, AIME25, 32K generation: budget 3072 scores 40.8, equal to full attention (10.7x KV, 2.5x throughput). Budget 2048 scores 32.9. AIME24 at budget 4096: 54.6 vs 57.1. MATH-500 at 1024: 68.4 vs 69.6. Protocol: temperature 0.6, top-p 0.95, 32K max tokens, 8 samples per problem. Tested models: Qwen3-8B, DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-7B, GPT-OSS-20B. | S4, S5 |
| E5 | TriAttention in TensorRT-LLM: PyTorch backend, KV cache manager V2, SM100/103/107 GPUs (B200 is the validated target), MHA/MQA/GQA only (no MLA, no hybrid), prompt tokens always preserved, needs a calibration file from the official repo, one compression method per LLM instance. | S6, S7 |
| E6 | TriAttention in vLLM (official plugin): needs a stats file via env var, eager mode and prefix caching off. Known failure: the budget silently not applied (the log shows 0 applied events). An SGLang backend exists since v0.2.0. | S5 |
| E7 | TheTom V3 eviction and longctx are an independent, experimental extension. Safe point is about 90% retention (about 1.1x). V3 plus TurboQuant: about 2.9x at +0.84% perplexity. Reasoning, multi-needle and 128K+ contexts are untested. longctx is a retrieval layer, not KV compression. | S8, S9 |
| E8 | K precision matters more than V precision, but it is model-family dependent. 2-bit V cost +5% to +9.5% perplexity. Asymmetric q8_0-K with turbo3-V produced corrupt output on a Qwen3.5-122B model while speed and perplexity looked fine. | S10 |
| E9 | KVzap: 2x to 4x with negligible loss on Qwen3-8B, Llama-3.1-8B-Instruct and Qwen3-32B; Hugging Face only (kvpress); needs a per-model trained surrogate. | S11 |

### 3.1 Repositories: what to take, borrow or leave
Pin a commit for every Dependency in third_party/ or the lockfile (this extends P0.3). Borrow means re-implement the idea through our harness or engine flags; do not copy code. Leave means do not install, import or depend on it. P0.4 also reopens these links.

| Repository | Decision | Take or borrow | Why |
|---|---|---|---|
| https://github.com/WeianMao/triattention | Dependency | vLLM plugin, SGLang backend, calibration script, shipped stats, docs/reproduction.md | Official and peer-reviewed; NVIDIA integrated it (E4, E5, E6) |
| https://github.com/vllm-project/vllm | Dependency (primary engine) | FP8 KV, skip-layers flag, built-in TurboQuant presets as a fallback | Evaluated independently by the vLLM team (E1, E2) |
| https://github.com/sgl-project/sglang | Dependency (Blackwell, optional) | NVFP4 KV, TriAttention backend | E3, E6 |
| https://github.com/NVIDIA/TensorRT-LLM | Dependency (T1 only, optional) | TriAttention compression config | E5 |
| https://github.com/NVIDIA/kvpress | Dependency (arm A8 only, optional) | KVzap press for Hugging Face pipelines | E9 |
| https://github.com/vllm-project/llm-compressor | Tool (optional) | KV-scale calibration for FP8 KV | Recommended by the vLLM docs (S12) |
| https://github.com/huggingface/Math-Verify and https://github.com/NVIDIA/RULER | Eval tools | Math answer checking; long-context retrieval tasks | Standard, engine-agnostic scoring |
| https://github.com/TheTom/turboquant_plus | Borrow ideas | Keep K at higher precision than V; protect the first and last layers; judge by generations and fidelity checks, not perplexity alone; probe the position sensitivity of eviction. The repo also ships a fidelity tool (REFRACT): evaluate it as an extra signal | Credible research (E8); its V3 and longctx parts are experimental (E7) |
| https://github.com/scos-lab/turboquant | Borrow idea | Per-layer K/V norm profiling of the target model as a risk flag | Measured on Qwen2.5 only; Llama untested; reference code |
| https://github.com/onur-gokyildiz-bhi/tq-kv | Borrow idea (parked) | Quantizing keys before RoPE | Its own results show large accuracy loss at high compression; research note only |
| https://github.com/Zefan-Cai/KVCache-Factory | Borrow (optional research arm) | SnapKV, PyramidKV and AdaKV baselines on LongBench and needle tests | Evaluation playground, batch size 1; never a production recommendation |
| https://github.com/TheTom/vllm-turboquant | Leave | None | Narrow validated envelope; reasoning, multi-needle and 128K+ untested (E7); upstream vLLM already ships TurboQuant (E2) |
| https://github.com/aivrar/multi-turboquant | Leave | None | Launcher and planner; its 80x preset has no task-accuracy test |
| https://github.com/AmesianX/TurboQuant | Leave | None | Archived by its author |

### 3.2 How the ideas merge
Merge at seams, not by merging code. Do not fork-merge these repositories or copy code between them. The combined community stacks (see Leave) depend on glue that the engine teams have not validated, and the one failure found in the official plugin was silent (E6). Connect the ideas at four seams instead, and let the harness check each one:

| Layer | Seam | Ideas merged here | Lives in kvpipe |
|---|---|---|---|
| L1 Precision | Engine flag (kv_dtype, skip-layers) | FP8 and NVFP4 from the engine teams; K-over-V asymmetry and boundary-layer protection from turboquant_plus | Arms A1, A1b, A2, A7 |
| L2 Eviction | Official plugin or config (budget, calibration, prompt protection) | TriAttention for generated tokens; KVzap as the Hugging Face alternative | Arms R1, A3 to A5, C1, A8 |
| L3 Verification | Our harness | Golden and integrity checks (E8); eviction canary (E6); position probes; K/V norm profile (scos-lab); reproduction protocol (E4) | run/, stats.py, profile.py |
| L4 Tiering (later) | Engine feature, not custom code | TensorRT-LLM cold-page tier instead of longctx-style retrieval (E5, E7) | Out of scope for v1; keep a documented hook |

Compatibility matrix. configs/compat.yaml is generated from this table and read by supports():

| Combination | vLLM | SGLang | TensorRT-LLM |
|---|---|---|---|
| FP8 KV alone | Validated (E1) | Supported; the baseline in E3 | Documented |
| NVFP4 KV alone | Not verified | Experimental, validated by the SGLang team (E3) | Documented; check the current quantization page |
| TriAttention alone | Official plugin (E6) | Official backend (E6) | Native, B200 (E5) |
| FP8 KV + TriAttention | Untested together | Untested together | Separate settings; not listed as validated |
| NVFP4 KV + TriAttention | Not available | Untested (arm A6) | Untested |
| TurboQuant preset + TriAttention | Untested; experimental lane only (rung 6) | Not available | Not available |

Merge ladder. Attempt one rung at a time; every rung must pass all gates in section 7 before the next starts:
1. A1: FP8 KV alone.
2. R1: TriAttention alone on BF16 KV (reproduces the paper).
3. A3 and A4: FP8 KV plus TriAttention.
4. A2: NVFP4 KV alone on Blackwell, then A6: NVFP4 plus TriAttention.
5. A1b: boundary-layer protection, only if a rung misses a gate by a small margin.
6. Experimental, only if the owner asks: a TurboQuant preset plus TriAttention. It is untested in every source reviewed, so it needs a smoke test, the full gates and an owner sign-off.

Expected shape of the final pipeline (to be confirmed by the gates, not assumed): FP8 KV for every workload (NVFP4 on Blackwell if it passes), plus TriAttention only on long-reasoning traffic, and nothing else.

## 4. Decision model
Hardware tiers, chosen by `kvpipe doctor` (the owner can override):

| Tier | Hardware | Engines to use | Note |
|---|---|---|---|
| T1 | Datacenter Blackwell (SM100/103, e.g. B200) | TensorRT-LLM for TriAttention; SGLang for NVFP4; vLLM | Only tier where TRT-LLM TriAttention is supported (E5) |
| T2 | Workstation or consumer Blackwell (SM120/121, e.g. RTX 5090, RTX PRO 6000, GB10) | vLLM (FP8, TriAttention plugin); SGLang (NVFP4) | TRT-LLM TriAttention not supported (E5) |
| T3 | Hopper (H100, H200) | vLLM | FP8 KV is validated here (E1) |
| T4 | Ada or Ampere (e.g. RTX 4090, RTX 3090, A100) | vLLM | FP8 KV validation covers Hopper and Blackwell only: test it, do not assume |

Arms (candidate configurations). The Expected column is a hypothesis from section 3, not a result.

| Arm | Configuration | Role | Expected |
|---|---|---|---|
| A0 | BF16 KV | Reference | Baseline |
| A1 | FP8 KV | Default recommendation | 2x capacity, within about 1 to 2 points (E1) |
| A2 | NVFP4 KV (SGLang) | Blackwell upgrade | About 1.78x over A1, within about 1.6 points of A1 (E3) |
| R1 | BF16 KV + TriAttention budget 3072 | Reproduces the paper; not a deployment candidate | Parity with A0 on AIME25 (E4) |
| A3 | A1 + TriAttention budget 4096 | Reasoning candidate | Within about 2.5 points (E4) |
| A4 | A1 + TriAttention budget 3072 | Reasoning candidate (target) | Parity on AIME25 (E4); the FP8 combination is untested |
| A5 | A1 + TriAttention budget 2048 | Negative control | About 8 points lower on AIME25 (E4) |
| A6 | A2 + TriAttention (SGLang backend) | Unverified combination | Unknown; smoke test only |
| A7 | TurboQuant 4bit_nc (vLLM) | Fallback if more than 2x is needed without eviction | 1 to 4 points loss, about 25% less throughput (E2) |
| A8 | KVzap via kvpress (Hugging Face) | Offline document-QA research arm | 2x to 4x, negligible loss (E9) |
| C1 | A1 + TriAttention budget 256 | Canary: proves eviction is active | Large accuracy drop |

Constraints on all arms:
• Hold model weights identical inside every comparison. NVFP4 KV needs an FP8-weight checkpoint (E3), so re-run A0 and A1 on that checkpoint for the A2 comparison.
• Eviction arms must protect prompt tokens and apply to long-reasoning workloads only, never to document QA (E7).

## 5. Architecture and repo layout
```
kvpipe/
  pyproject.toml            # lockfile, ruff, pytest
  PLAN.md, PLAN_STATE.md    # this plan and the progress log
  configs/arms/*.yaml       # one file per arm (Appendix A)
  configs/suites/*.yaml     # eval suites, sample counts, seeds
  configs/gates.yaml        # thresholds approved at CP0
  src/kvpipe/
    doctor.py               # hardware and engine discovery, tier selection
    config.py               # pydantic models for arms, suites, gates
    engines/                # base.py, vllm.py, sglang.py, trtllm.py, stub.py
    calib.py                # TriAttention calibration wrapper and cache
    data/                   # dataset loaders with pinned revisions
    run/                    # sampler.py, scoring.py, integrity.py, perf.py
    stats.py                # paired bootstrap, verdicts
    decide.py               # gates and selection
    report.py               # report.md, report.json, claims audit
    cli.py                  # doctor | calibrate | run | report | audit | regress
  third_party/              # pinned clones and patches
  artifacts/                # raw generations, logs, metrics (gitignored)
  tests/                    # unit tests, golden prompts, tiny-model integration tests
```

Key interfaces:
• Engine.supports(arm, env) returns (ok, reason). Engine.launch(arm) returns a handle with kv_capacity_tokens(), metrics(), eviction_events() (received and applied, or None) and stop().
• Every evaluation talks to an OpenAI-compatible endpoint, so evals are engine-agnostic. Run one server at a time and confirm GPU memory is released between arms.
• Results are append-only JSONL keyed by (arm, suite, item, sample, seed), so any run can resume.
• stub.py is a fake OpenAI-compatible server with canned answers, used to test the sampler, scorers, stats and report without a GPU.

## 6. Build phases
### Phase 0: Preflight
• P0.1 Create the repo and PLAN_STATE.md.
• P0.2 Discover the environment and save raw output under artifacts/env/: nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv, nvcc --version, python --version, pip list filtered for vllm, sglang, tensorrt_llm, torch and flash-attn, and df -h.
• P0.3 Clone WeianMao/triattention into third_party/ and pin the commit. Read its README, docs/reproduction.md and the calibration instructions. Check whether shipped stats exist for the reference model.
• P0.4 Freshness check: reopen every URL in Appendix C. If a fact in section 3 changed, record it in PLAN_STATE.md and tell the owner.
• CP0 (ask the owner): target model(s) (default reference model: Qwen3-8B, the model with the most published TriAttention evidence); workload mix (long reasoning, document QA, chat, agents); maximum context and typical output length; latency targets (p95 TPOT and TTFT); accuracy margin (standard 2 points, strict 1 point); GPU count and budget in GPU hours; permission for downloads; whether a Hugging Face token is available (GPQA is gated).
• Done when: the environment JSON is saved, third-party repos are pinned and the owner's answers are recorded.

### Phase 1: Scaffold and doctor
• P1.1 Create the skeleton from section 5 with a lockfile, ruff and pytest.
• P1.2 Write pydantic schemas for arms, suites and gates, and validate the YAML files against them.
• P1.3 Implement kvpipe doctor: parse the P0.2 output into env.json, map compute capability to a tier (T1 to T4), list installed engines and versions, and call supports() for every arm, printing the reason when an arm cannot run.
• P1.4 Encode these support rules in supports(): TriAttention on TensorRT-LLM needs compute capability 10.0, 10.3 or 10.7 and a GQA, MHA or MQA model (no MLA, no hybrid); TurboQuant needs standard attention (no hybrid, no sliding window); NVFP4 KV needs SM12x or SM100/103 and a GQA or sparse-MLA model; FP8 KV on T4 is marked experimental.
• P1.5 Unit-test the tier mapping and support rules with recorded fixtures for B200, RTX 5090, H100 and RTX 4090.
• Done when: tests pass and kvpipe doctor on the real machine matches reality (the owner confirms the tier).

### Phase 2: vLLM adapter and baselines (A0, A1)
• P2.1 Implement engines/vllm.py: build the command, start the server, poll /health, stop cleanly (kill the process group), capture logs to artifacts, parse the startup line that reports KV cache capacity in tokens, scrape /metrics, and assert from the startup log that the effective KV dtype equals the arm's intended dtype.
• P2.2 Define A0 (--kv-cache-dtype auto) and A1 (fp8, plus --kv-cache-dtype-skip-layers sliding_window when the model config has sliding-window layers). Use identical --gpu-memory-utilization and --max-model-len for every arm.
• P2.3 Write 20 deterministic golden prompts (temperature 0, known answers: arithmetic, facts, formatting, one multilingual line, one code snippet) in tests/golden.jsonl.
• P2.4 If A1 later fails gate G2, calibrate KV scales (llm-compressor dataset calibration, per-attention-head if the backend supports it; see S12) and re-run.
• Done when: both arms serve, pass 20 of 20 golden prompts, and the measured capacity ratio A1 over A0 is between 1.8x and 2.2x. If not, investigate (KV scales, skipped layers, engine version) before continuing.

### Phase 3: Evaluation harness
• P3.1 Dataset loaders with pinned revisions: AIME 2024, AIME 2025, MATH-500, GPQA-Diamond (gated), and a long-context retrieval suite (openai/mrcr, or a synthetic needle suite at four lengths and five depths, sized below the model's native context). Reuse the data sources named in the official docs/reproduction.md. Add a custom suite loader for the owner's prompts (JSONL): with reference answers use exact or rubric matching; without them use greedy-decode agreement with A0 plus the integrity checks and the review file.
• P3.2 Async sampler with retries, streaming (records TTFT and TPOT), per-item JSONL cache, resume and fixed seeds. Reasoning protocol from E4: thinking mode on, temperature 0.6, top-p 0.95, 32K max tokens, at least 8 samples per problem (16 for AIME).
• P3.3 Scorers: Math-Verify (or equivalent) for math; exact match for multiple choice and retrieval.
• P3.4 Integrity checks on every response: empty output, finish reason (length vs stop), answer-extraction success, replacement characters and runs of question marks, n-gram repetition ratio and language drift. Write 20 random generations per arm to artifacts/review/ARM.md for the owner to read.
• P3.5 stats.py: paired bootstrap over problems (10,000 resamples), 95% CI of (arm minus reference), and the PASS, FAIL or INCONCLUSIVE verdict from section 7. Add power planning: from the pilot variance, choose samples per problem so the CI half-width is at most the margin, capped by the GPU budget; if the cap binds, report INCONCLUSIVE instead of PASS.
• P3.6 Known-answer test: A0 on the reference model, AIME25, paper protocol. Expect about 40.8 for Qwen3-8B (E4).
• P3.7 Stub-engine tests for the sampler, scorers, stats and integrity checks. A unit test must show the integrity checker flags every deliberately corrupted output.
• CP1: show the owner the baseline results and the review file.
• Done when: the AIME25 baseline is within 6 points of 40.8 (otherwise debug the prompt template and answer parsing against the official repo's eval script; do not widen the tolerance), a rerun with a different seed agrees within the CI, and the stub tests pass.

### Phase 4: Performance harness
• P4.1 Load generator: concurrency sweep (1, 4, 16, 64, up to the engine limit) with prompts drawn from the suites' real length distributions, plus a forced-long generation case (about 32K output tokens, using min_tokens or ignore_eos) for eviction arms.
• P4.2 Per run, record output tokens per second, TTFT and TPOT p50, p95 and p99, peak running requests, OOM or preemption counts, KV capacity in tokens, and KV bytes per token computed from the model config and dtype (cross-check against the engine's capacity).
• P4.3 Repeat each point three times; report mean and standard deviation.
• Done when: the A0 and A1 performance table exists and repeats agree within 5%.

### Phase 5: TriAttention arms (R1, A3, A4, A5, C1)
• P5.1 calib.py: wrap the official calibrator; cache stats by (model revision, repo commit); use shipped stats only if the model matches exactly.
• P5.2 vLLM plugin launcher: set TRIATTN_RUNTIME_SPARSE_STATS_PATH and TRIATTN_RUNTIME_KV_BUDGET; pass --enforce-eager, disable prefix caching and apply the README's batched-token cap. Find the plugin's prompt-protection switch in the README and turn it on.
• P5.3 eviction_events(): parse the plugin's log line that reports received and applied events. Fail the run if applied is 0 for any request longer than the budget.
• P5.4 T1 only: TensorRT-LLM adapter using KvCacheConfig(use_kv_cache_manager_v2=True) and TriAttentionKvCacheCompressionConfig(budget, beta=128, eviction_mode='union', calibration_path). Confirm eviction by checking that KV pages in use plateau during a forced-long generation.
• P5.5 Canary C1: on a 16K+ token generation task, budget 256 must score clearly below budget 4096 (paired CI excludes zero; expect more than 10 points). Otherwise treat eviction as inactive and stop.
• P5.6 Reproduction check: R1 vs A0 on AIME25 with the paper protocol agrees within 5 points, and budget 2048 scores clearly below budget 3072 (E4 expects about 8 points).
• CP2: show the owner the reproduction and canary results.
• Done when: all five arms run end to end, eviction is verified active, and capacity and throughput are measured, not assumed.

### Phase 6: Borrowed ideas and optional arms
Each item sits behind a feature flag; skip it when supports() says the arm cannot run.
• P6.1 profile.py (idea from scos-lab): measure the per-layer ratio of key norms to value norms for the target model with a short Hugging Face forward pass (about 100 prompts) and write artifacts/profile/MODEL.json. This is an informational risk flag for K-quantizing arms, not a gate.
• P6.2 Arm A1b (idea from turboquant_plus; it has no row in the section 4 table, so create configs/arms/a1b_fp8_boundary.yaml): FP8 KV with the first 2 and last 2 layers kept in higher precision through the skip-layers flag, if the flag accepts layer indices (confirm in S12 and vllm serve --help; otherwise record unsupported). Run it only if A1 misses G2 by a small margin.
• P6.3 Position probes for eviction arms: (a) the needle suite at 10%, 50% and 90% depth in the prompt, which must match A1 because prompt tokens are protected; (b) a generated-memory probe: the prompt asks for a random code on the first line, then about 10K tokens of filler computation, then the code again. Report exact recall for budgets 256, 2048, 3072 and 4096.
• P6.4 Fidelity signal beyond accuracy: greedy-decode agreement and mean token-level logprob divergence against A0 on a 200-prompt held-out set. Optionally evaluate the fidelity tool shipped in turboquant_plus as an extra signal; do not make it a dependency.
• P6.5 SGLang adapter and arm A2: --kv-cache-dtype nvfp4 on an FP8-weight checkpoint, attention backends as in S3, --enable-metrics. Then the A6 smoke test (TriAttention SGLang backend with NVFP4 KV): golden prompts plus the canary, recording pass or fail. Never recommend A6 without passing every gate.
• P6.6 Arm A7: turboquant_4bit_nc on vLLM with an architecture guard that rejects hybrid and sliding-window models.
• P6.7 Arm A8: KVzap via kvpress for Hugging Face-only workloads, evaluated on the retrieval suite and reported in a separate section.
• P6.8 Research arm, only if the owner asks: SnapKV, PyramidKV and AdaKV baselines from KVCache-Factory on LongBench and needle tests at batch size 1, for comparison with TriAttention. Never a production recommendation.
• P6.9 Only if the owner runs llama.cpp: a fork arm with q8_0 K and turbo4 V. Verification must include reading generations (E8).
• Done when: each module or arm is implemented and tested, or recorded as unsupported with a reason.

### Phase 7: Decision and report
• P7.1 decide.py: evaluate gates G1 to G7 for each arm. Among passing arms choose the highest measured compression and break ties on throughput. Always name a safe fallback (default A1).
• P7.2 report.py: write report.md and report.json with one row per arm: measured KV compression versus A0, accuracy deltas with CIs, integrity rates, throughput, TTFT and TPOT, and the verdicts. Add a diagnostics section (position probes, K/V profile, fidelity signal) and the sample plan for every INCONCLUSIVE result.
• P7.3 kvpipe audit: fail if any number in the report lacks a run ID and an artifact path.
• P7.4 Generate deploy/recommended.sh and deploy/fallback.sh for the chosen arms.
• CP3: the owner reviews the report and signs off on the recommendation.
• Done when: kvpipe audit passes and the owner approves.

### Phase 8: Hardening and rollout
• P8.1 kvpipe regress: re-run the golden prompts, the canary and a reduced suite after any engine, model or dataset change, and compare with the stored report.
• P8.2 Production monitoring: KV cache usage, eviction events received vs applied, integrity failure rate on sampled outputs, answer-parse rate, p95 TPOT, and OOM or preemption counts.
• P8.3 Canary rollout: send 5% of traffic to the recommended arm. Roll back automatically if the integrity failure rate exceeds baseline by 0.5 points or p95 TPOT breaches the target for 15 minutes.
• P8.4 Docs: README.md (quickstart), RUNBOOK.md (silent no-op, OOM, version drift) and DECISIONS.md (why these arms and gates).
• CP4: the owner approves rollout.
• Done when: kvpipe regress passes on a clean machine and the docs are reviewed.

## 7. Gates and statistics
Defaults below; the owner confirms them at CP0. A compression arm is compared with A1, and A1 itself with A0, always on identical weights.

| Gate | Check | Pass rule |
|---|---|---|
| G1 Integrity | Golden prompts, corrupt or empty outputs, answer extraction | 20 of 20 golden; corrupt or empty rate at most reference plus 0.5 points; extraction rate at least reference minus 1 point |
| G2 Reasoning accuracy | AIME24, AIME25, MATH-500 and GPQA-Diamond, each and pooled | CI lower bound of (arm minus reference) at least minus margin (2.0 points standard, 1.0 strict) |
| G3 Long-context retrieval | Only if the workload has long prompts | Same rule with a 1 point margin at every tested length; eviction arms with prompt protection off fail automatically |
| G4 Eviction active | Eviction arms only | Applied events above 0 on over-budget requests, and canary C1 shows a clear drop |
| G5 Capacity | Measured KV capacity gain versus A0 | Report it; an arm counts as a compression arm only at 1.8x or more |
| G6 Performance | The owner's concurrency | p95 TPOT within target; throughput at least 70% of A1; no OOM or preemption |
| G7 Stability | Three repeats | Standard deviation of the pooled score at most 1.5 points, otherwise add samples |

Verdict rule: PASS if the CI lower bound is at least minus margin. FAIL if the CI upper bound is below minus margin. Otherwise INCONCLUSIVE: add samples, and never report it as a pass. The owner may relax to a 90% CI for screening runs only.

Compression accounting. Report two measured parts and their product. (a) Dtype gain: engine-reported KV capacity in tokens at equal memory settings, divided by A0's. (b) Eviction ratio: the average over requests of total tokens divided by peak resident tokens (prompt tokens are never evicted). Never quote nominal ratios.

Diagnostics (reported, not gated): position probes (P6.3), K/V norm profile (P6.1) and fidelity signal (P6.4). The owner decides at CP3 whether a diagnostic blocks a recommendation, for example when the workload needs late recall of early outputs.

Power planning figure (the author's estimate; replace it with pilot data from P3.5): about 32 samples per problem, pooled over AIME 2024 and 2025, gives a CI half-width near 2 points. The paper's 8 samples are a floor for reproduction, not enough for a 2-point non-inferiority claim.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| TriAttention budget silently not applied (E6) | G4, canary C1, applied-events check |
| AIME has only 30 problems per year, so comparisons are noisy | Paired bootstrap, pooled suites, power planning, INCONCLUSIVE verdict |
| Corrupt output hidden by perplexity or speed checks (E8) | G1, golden prompts, review files read by the owner |
| Eviction forgets early generated facts | Generated-memory probe (P6.3); limit eviction to workloads that do not need late recall of early outputs |
| Eager mode and no prefix caching hurt multi-turn latency (E6) | Measure TTFT in Phase 4; report in G6 |
| FP8 regressions on head_dim 256 or sliding-window models (E1) | Skip-layers option; prefill benchmark in Phase 4 |
| Hybrid and MLA models unsupported (E2, E5) | supports() guards with explicit reasons |
| Version drift in engines or the plugin | Pinned lockfile; kvpipe regress |
| Prompt-format mismatch or contamination skews absolute scores | Compare arms against A0 on the same model; known-answer test in P3.6 |
| GPU budget overrun | Pilot-first rule; budget confirmed at CP0 |
| Calibration script executes remote code | Rule 6; owner-approved models only |

## 9. Out of scope and do not do
Do not implement or recommend: TurboQuant 3-bit presets for reasoning (E2); V3 and longctx in production (E7); eviction of prompt tokens for document QA (E7); any report that multiplies nominal ratios; any repository marked Leave in section 3.1.
Out of scope for v1: training or fine-tuning, MLA and hybrid-model support, multi-node serving, Apple Silicon.

## Appendix A: Arm config example
```yaml
# configs/arms/a4_fp8_tri3072.yaml
id: A4
engine: vllm                  # vllm, sglang, trtllm or stub
model: Qwen/Qwen3-8B
model_revision: COMMIT_SHA    # pin it
weights_dtype: bf16
kv_dtype: fp8                 # auto, fp8, nvfp4 or turboquant_4bit_nc
skip_layers: auto             # adds sliding_window when the model has such layers
compression:
  method: triattention
  budget: 3072
  protect_prompt: true
  stats_path: artifacts/calib/qwen3-8b/REPO_COMMIT.pt
serve:
  gpu_memory_utilization: 0.90
  max_model_len: FROM_MODEL_CONFIG
  extra_args: [--enforce-eager, --no-enable-prefix-caching]
```

## Appendix B: Launch templates
```bash
# A0 and A1 on vLLM
vllm serve MODEL --kv-cache-dtype auto
vllm serve MODEL --kv-cache-dtype fp8        # add --kv-cache-dtype-skip-layers sliding_window if the model needs it

# A4 on vLLM with the official TriAttention plugin (follow its README for install and the prompt-protection switch)
export TRIATTN_RUNTIME_SPARSE_STATS_PATH=STATS.pt
export TRIATTN_RUNTIME_KV_BUDGET=3072
vllm serve MODEL --enforce-eager --no-enable-prefix-caching    # FP8 KV together with the plugin is untested: gate it

# A2 on SGLang (Blackwell, experimental)
python -m sglang.launch_server --model-path FP8_WEIGHT_MODEL --kv-cache-dtype nvfp4 --prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha --enable-metrics

# A7 fallback on vLLM
vllm serve MODEL --kv-cache-dtype turboquant_4bit_nc
# T1 only: TensorRT-LLM with TriAttention (B200 is the validated target); confirm the import paths in S7
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, TriAttentionKvCacheCompressionConfig
llm = LLM(model='MODEL', backend='pytorch',
          kv_cache_config=KvCacheConfig(use_kv_cache_manager_v2=True),
          kv_cache_compression_config=TriAttentionKvCacheCompressionConfig(
              budget=3072, beta=128, eviction_mode='union', calibration_path='CALIB.pt'))
```

## Appendix C: Sources
Source codes used in section 3. Repository links are in section 3.1.

| Code | Source | URL |
|---|---|---|
| S1 | vLLM TurboQuant study (accuracy and performance) | https://vllm.ai/blog/2026-05-11-turboquant |
| S2 | vLLM FP8 KV-cache and attention quantization | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| S3 | SGLang NVFP4 KV cache | https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache/ |
| S4 | TriAttention paper | https://arxiv.org/abs/2604.04921 |
| S5 | TriAttention repo pages: README, issue 24, release v0.2.0, reproduction doc | https://github.com/WeianMao/triattention (also /issues/24, /releases/tag/v0.2.0, /blob/main/docs/reproduction.md) |
| S6 | TensorRT-LLM KV cache compression docs | https://nvidia.github.io/TensorRT-LLM/features/kv-cache-compression.html |
| S7 | TensorRT-LLM TriAttention example | https://github.com/NVIDIA/TensorRT-LLM/blob/main/examples/kv_cache_compression/triattention.md |
| S8 | TheTom TriAttention V3 paper | https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/triattention-v3.md |
| S9 | TheTom vllm-turboquant usage notes | https://github.com/TheTom/vllm-turboquant/blob/feature/turboquant_plus/USAGE.md |
| S10 | TheTom asymmetric K/V paper | https://github.com/TheTom/turboquant_plus/blob/main/docs/papers/asymmetric-kv-compression.md |
| S11 | KVzap paper | https://arxiv.org/abs/2601.07891 |
| S12 | vLLM quantized KV cache docs | https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/ |
| S13 | TurboQuant paper (background) | https://arxiv.org/abs/2504.19874 |
