#!/usr/bin/env bash
# Deploy async GRPO 2048 training to a fresh RunPod pod via runpodctl.
#
# Creates a 2-GPU pod, uploads the trainer + launcher scripts, and kicks off
# `run_async_grpo_runpod.sh` over SSH. The pod keeps running after training so
# you can inspect logs / checkpoints — `runpodctl pod stop <id>` when done.
#
# Required env vars (on this machine):
#   RUNPOD_API_KEY   - runpodctl auth (or run `runpodctl doctor` first)
#   HF_TOKEN         - forwarded into the pod
#   WANDB_API_KEY    - forwarded into the pod
#
# Tunables:
#   GPU_ID            (default "NVIDIA GeForce RTX 4090")
#   GPU_COUNT         (default 2)
#   IMAGE             (default runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404)
#   CONTAINER_DISK_GB (default 50)
#   VOLUME_GB         (default 100)
#   POD_NAME          (default async-grpo-2048-<timestamp>)
#   SSH_KEY           (default ~/.ssh/id_ed25519 — must be registered via `runpodctl ssh add-key`)

set -euo pipefail

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (or run 'runpodctl doctor' first)}"
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${WANDB_API_KEY:?WANDB_API_KEY must be set}"

GPU_ID="${GPU_ID:-NVIDIA GeForce RTX 4090}"
GPU_COUNT="${GPU_COUNT:-2}"
IMAGE="${IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-50}"
VOLUME_GB="${VOLUME_GB:-100}"
POD_NAME="${POD_NAME:-async-grpo-2048-$(date +%Y%m%d-%H%M%S)}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
command -v runpodctl >/dev/null || { echo "ERROR: runpodctl not installed" >&2; exit 1; }
command -v jq        >/dev/null || { echo "ERROR: jq not installed"        >&2; exit 1; }
[[ -f "$SSH_KEY" ]]             || { echo "ERROR: SSH key $SSH_KEY not found" >&2; exit 1; }

for f in run_async_grpo_runpod.sh openenv_2048_async_grpo.py; do
  [[ -f "$SCRIPT_DIR/$f" ]] || { echo "ERROR: missing $SCRIPT_DIR/$f" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# Create pod
# ---------------------------------------------------------------------------
ENV_JSON=$(jq -n \
  --arg hf "$HF_TOKEN" \
  --arg wb "$WANDB_API_KEY" \
  '{HF_TOKEN: $hf, WANDB_API_KEY: $wb}')

echo "[runpod] creating pod $POD_NAME ($GPU_COUNT x $GPU_ID)"
CREATE_OUT=$(runpodctl pod create \
  --name "$POD_NAME" \
  --image "$IMAGE" \
  --gpu-id "$GPU_ID" \
  --gpu-count "$GPU_COUNT" \
  --container-disk-in-gb "$CONTAINER_DISK_GB" \
  --volume-in-gb "$VOLUME_GB" \
  --volume-mount-path /workspace \
  --ports "22/tcp" \
  --env "$ENV_JSON" \
  -o json)

POD_ID=$(echo "$CREATE_OUT" | jq -r '.id // .data.id // empty')
if [[ -z "$POD_ID" ]]; then
  echo "ERROR: couldn't parse pod id from runpodctl output:" >&2
  echo "$CREATE_OUT" >&2
  exit 1
fi
echo "[runpod] pod id: $POD_ID"

cleanup_hint() {
  echo
  echo "Pod $POD_ID is still running. Manage with:"
  echo "  runpodctl pod get    $POD_ID"
  echo "  runpodctl ssh info   $POD_ID"
  echo "  runpodctl pod stop   $POD_ID"
  echo "  runpodctl pod delete $POD_ID"
}
trap cleanup_hint EXIT

# ---------------------------------------------------------------------------
# Wait until SSH is reachable
# ---------------------------------------------------------------------------
echo "[runpod] waiting for pod to be RUNNING"
for _ in $(seq 1 60); do
  STATUS=$(runpodctl pod get "$POD_ID" -o json | jq -r '.desiredStatus // .status // empty')
  echo "  status=$STATUS"
  [[ "$STATUS" == "RUNNING" ]] && break
  sleep 5
done

echo "[runpod] resolving SSH endpoint"
SSH_INFO=$(runpodctl ssh info "$POD_ID" -o json)
SSH_HOST=$(echo "$SSH_INFO" | jq -r '.host // .ip // empty')
SSH_PORT=$(echo "$SSH_INFO" | jq -r '.port // 22')
SSH_USER=$(echo "$SSH_INFO" | jq -r '.user // "root"')

if [[ -z "$SSH_HOST" ]]; then
  echo "ERROR: couldn't parse SSH host. Raw output:" >&2
  echo "$SSH_INFO" >&2
  exit 1
fi

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -p "$SSH_PORT")

echo "[runpod] waiting for sshd on $SSH_USER@$SSH_HOST:$SSH_PORT"
for _ in $(seq 1 60); do
  if ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" true 2>/dev/null; then
    echo "[runpod] ssh is up"
    break
  fi
  sleep 5
done

# ---------------------------------------------------------------------------
# Upload code + launch training
# ---------------------------------------------------------------------------
REMOTE_DIR=/workspace/open-env
ssh "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" "mkdir -p $REMOTE_DIR"

echo "[runpod] uploading scripts"
scp "${SSH_OPTS[@]}" \
  "$SCRIPT_DIR/run_async_grpo_runpod.sh" \
  "$SCRIPT_DIR/openenv_2048_async_grpo.py" \
  "$SSH_USER@$SSH_HOST:$REMOTE_DIR/"

echo "[runpod] launching training (Ctrl-C detaches; training keeps running)"
ssh -t "${SSH_OPTS[@]}" "$SSH_USER@$SSH_HOST" \
  "cd /workspace && bash $REMOTE_DIR/run_async_grpo_runpod.sh $*"
