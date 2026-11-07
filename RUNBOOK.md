# kvpipe Runbook: Operations & Troubleshooting

## 1. Silent Eviction No-Op (TriAttention Plugin)
**Symptom:** Eviction arm runs, but KV cache capacity or memory usage shows no compression, or logs show `applied events: 0`.  
**Root Cause:** Prefix caching or CUDA graph capture can cause eviction hooks to be bypassed.  
**Resolution:**
1. Check `artifacts/report.json` for `G4_EvictionActive` status.
2. Ensure `--enforce-eager` and `--no-enable-prefix-caching` are passed to `vllm serve`.
3. Confirm `TRIATTN_RUNTIME_SPARSE_STATS_PATH` and `TRIATTN_RUNTIME_KV_BUDGET` environment variables are exported in the serving process environment.
4. Run Canary C1 (`budget=256`). If Canary accuracy does not drop by >10 points, eviction is confirmed inactive.

## 2. Out of Memory (OOM) During Long Contexts
**Symptom:** CUDA OOM error during prefill or long generation.  
**Root Cause:** A large batched prefill chunk can overshoot the KV budget before the periodic eviction hook triggers.  
**Resolution:**
1. Set `--max-num-batched-tokens 1024` to cap chunk sizes.
2. Ensure `--gpu-memory-utilization` is held constant (default `0.90`).
3. For multi-turn chat workloads, increase the budget from 2048/3072 to 8192/12000 (`TRIATTN_RUNTIME_KV_BUDGET`).

## 3. Version Drift & Dependency Pinning
**Symptom:** Regression failure in `kvpipe regress` after package or driver updates.  
**Resolution:**
1. Check pinned submodule SHA for TriAttention in `third_party/triattention` (pinned at `a4bc3c8f709db60f016ef42c3feb290fd0c00c1b`).
2. Run `kvpipe regress --arm A1` to verify golden prompt deterministic outputs.

## 4. Production Canary Rollout & Automatic Rollback
**Traffic Allocation:** Route 5% of production traffic to the recommended arm.  
**Rollback Triggers (P8.3):**
- Roll back immediately if the sampled output integrity failure rate exceeds baseline by **0.5 points**.
- Roll back if p95 TPOT breaches the latency target threshold for **15 consecutive minutes**.
- To revert, switch traffic to `deploy/fallback.sh`.
