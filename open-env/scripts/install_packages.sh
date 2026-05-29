#!/usr/bin/env bash
set -euo pipefail

uvx hf auth login
uv venv
uv pip install "trl[vllm]"
uv pip install -U transformers
uv pip install wandb
uv pip install -e ../OpenEnv/envs/agent_world_model_env
uv pip install python-dotenv
