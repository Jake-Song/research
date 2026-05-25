#!/usr/bin/env bash
# Launch async GRPO training on a 2-GPU Runpod pod.
# GPU 0 runs the vLLM server; GPU 1 runs AsyncGRPOTrainer.
#
# Required env vars (set in Runpod pod secrets):
#   HF_TOKEN         - HuggingFace token (for model pull + push_to_hub)
#   WANDB_API_KEY    - Weights & Biases API key
#
# Usage:
#   bash open-env/run_async_grpo_runpod.sh
#   bash open-env/run_async_grpo_runpod.sh --model-id Qwen/Qwen3-4B  # extra args forwarded to trainer

set -euo pipefail

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-3072}"
LOG_DIR="${LOG_DIR:-/workspace/logs}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
if [[ "$GPU_COUNT" -lt 2 ]]; then
  echo "ERROR: need >=2 GPUs, found $GPU_COUNT" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Install deps
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd "$REPO_ROOT"

echo "[setup] installing python deps"
uv pip install --system \
  "trl[vllm]>=0.25" \
  peft \
  accelerate \
  liger-kernel \
  wandb \
  "transformers>=4.57" \
  "git+https://huggingface.co/spaces/Jakemu/openspiel_env"

# ---------------------------------------------------------------------------
# Hugging Face + wandb auth
# ---------------------------------------------------------------------------
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
wandb login "$WANDB_API_KEY" >/dev/null

# ---------------------------------------------------------------------------
# Start vLLM server on GPU 0
# ---------------------------------------------------------------------------
VLLM_LOG="$LOG_DIR/vllm_server.log"
echo "[vllm] starting server (GPU 0) -> $VLLM_LOG"

CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
  vllm serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --logprobs-mode processed_logprobs \
    --weight-transfer-config '{"backend":"nccl"}' \
    >"$VLLM_LOG" 2>&1 &
VLLM_PID=$!

cleanup() {
  echo "[cleanup] killing vLLM server (pid $VLLM_PID)"
  kill "$VLLM_PID" 2>/dev/null || true
  wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Wait for server to accept requests (max ~10 minutes).
echo "[vllm] waiting for server on :$VLLM_PORT"
for _ in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then
    echo "[vllm] server is up"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM server died during startup. Last 50 lines:" >&2
    tail -n 50 "$VLLM_LOG" >&2
    exit 1
  fi
  sleep 5
done

if ! curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null; then
  echo "ERROR: vLLM server failed to become healthy in 10 minutes" >&2
  tail -n 50 "$VLLM_LOG" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Launch trainer on GPU 1
# ---------------------------------------------------------------------------
echo "[train] launching AsyncGRPOTrainer (GPU 1)"

CUDA_VISIBLE_DEVICES=1 \
  accelerate launch "$REPO_ROOT/open-env/openenv_2048_async_grpo.py" \
    --model-id "$MODEL_ID" \
    --vllm-server-host 127.0.0.1 \
    --vllm-server-port "$VLLM_PORT" \
    "$@"

echo "[done] training finished"
