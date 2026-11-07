#!/usr/bin/env bash
set -euo pipefail
# kvpipe Recommended Deployment: Arm A4
TRIATTN_RUNTIME_SPARSE_STATS_PATH=third_party/triattention/triattention/calibration/for_aime25_experiment/qwen3_8b.pt TRIATTN_RUNTIME_KV_BUDGET=3072 TRIATTN_RUNTIME_PROTECT_PREFILL=true vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window --enforce-eager --no-enable-prefix-caching --max-num-batched-tokens 1024
