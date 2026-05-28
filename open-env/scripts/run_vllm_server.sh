#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
    uv run vllm serve Qwen/Qwen3-1.7B \
        --max-model-len 3072 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}'
