#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=1 uv run accelerate launch "$SCRIPT_DIR/openenv_2048_async_grpo.py" "$@"
