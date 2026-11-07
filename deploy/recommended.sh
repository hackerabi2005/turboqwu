#!/usr/bin/env bash
set -euo pipefail
# kvpipe Recommended Deployment: Arm A6
vllm serve Qwen/Qwen3-8B-FP8 --kv-cache-dtype nvfp4
