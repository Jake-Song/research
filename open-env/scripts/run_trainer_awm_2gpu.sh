#!/usr/bin/env bash
# 2-GPU layout: single trainer process on GPU 1 (vLLM owns GPU 0). No FSDP
# sharding -> the single-GPU FP8 accelerate config. Mirrors run_trainer_awm.sh
# but for one trainer GPU. Pass trainer flags (--env-url, --max-steps, ...) as args.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES=1 \
    uv run accelerate launch \
        --config_file "$REPO_ROOT/open-env/configs/accelerate_fp8_single_gpu.yaml" \
        "$REPO_ROOT/open-env/openenv_awm_async_grpo.py" "$@"
