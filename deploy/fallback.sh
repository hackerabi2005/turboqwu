#!/usr/bin/env bash
set -euo pipefail
# kvpipe Safe Fallback Deployment: Arm A1
vllm serve Qwen/Qwen3-8B --kv-cache-dtype fp8 --kv-cache-dtype-skip-layers sliding_window
