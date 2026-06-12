#!/usr/bin/env bash
set -euo pipefail

# /v1/completions doesn't accept thinking_token_budget upstream; patch it in.
uv run python "$(dirname "$0")/patch_vllm_thinking_budget.py"

CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
    uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
        --max-model-len 32000 \
        --logprobs-mode processed_logprobs \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        --weight-transfer-config '{"backend":"nccl"}'
