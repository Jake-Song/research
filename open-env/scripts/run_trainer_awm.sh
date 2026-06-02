#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CUDA_VISIBLE_DEVICES=1 \
    uv run accelerate launch \
        --config_file "$REPO_ROOT/open-env/configs/accelerate_fp8_single_gpu.yaml" \
        "$REPO_ROOT/open-env/openenv_awm_async_grpo.py" "$@"
