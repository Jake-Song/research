#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
    uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --max-model-len 32000 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}'
