#!/usr/bin/env bash
set -euo pipefail

uvx hf auth login
uv venv
uv pip install -e ../trl[vllm]
uv pip install -U transformers
uv pip install -U torch==2.11.0
uv pip install wandb bitsandbytes kernels==0.14.0 torchao
uv pip install -e ../OpenEnv/envs/agent_world_model_env
uv pip install python-dotenv
