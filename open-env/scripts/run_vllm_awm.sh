#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a && source .env && set +a
fi

uv run hf auth login --token "${HF_TOKEN}"

# /v1/completions doesn't accept thinking_token_budget upstream; patch it in.
uv run python "$(dirname "$0")/patch_vllm_thinking_budget.py"

CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_SERVER_DEV_MODE=1 \
    uv run vllm serve Jakemu/Qwen3-4B-Thinking-awm-async-grpo-100 \
        --tensor-parallel-size 4 \
        --max-model-len 32768 \
        --logprobs-mode processed_logprobs \
        --reasoning-parser deepseek_r1 \
        --reasoning-config '{"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}' \
        --weight-transfer-config '{"backend":"nccl"}' \
        --override-generation-config '{"temperature":1}' \
               
        # Qwen3 thinking-mode recommended sampling. The rollout request omits
        # top_p/top_k/min_p/presence_penalty so these server defaults apply; it
        # always sends temperature, so during training the trainer's
        # AsyncGRPOConfig.temperature overrides the 0.6 here (set there too).
