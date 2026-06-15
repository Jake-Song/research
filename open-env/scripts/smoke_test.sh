#!/usr/bin/env bash
# Smoke test for the AWM async-GRPO stack. Run AFTER the env server and vLLM
# server are up (see run_vllm_awm.sh and the env-server command in the trainer
# docstring). It checks, in order:
#   1. AWM env server  /health
#   2. vLLM server     /health, /v1/models, and /metrics (the prometheus/fastapi
#      regression — must be 200, not 500)
#   3. vLLM can generate a tool-calling response
#   4. the trainer runs through one optimizer step on the trainer GPU without
#      OOM-ing (memory-light overrides), which is the real pre-flight for a run
#
# Overridable via env vars: ENV_URL, VLLM_HOST, VLLM_PORT, TRAINER_GPU,
# SMOKE_TIMEOUT (seconds to wait for the first trainer step, default 900).
set -uo pipefail

ENV_URL="${ENV_URL:-http://localhost:8899}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM="http://${VLLM_HOST}:${VLLM_PORT}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

pass() { printf '\033[32m[PASS]\033[0m %s\n' "$1"; }
fail() { printf '\033[31m[FAIL]\033[0m %s\n' "$1"; exit 1; }

echo "== 1. AWM env server ($ENV_URL) =="
curl -fsS -m 5 "$ENV_URL/health" >/dev/null 2>&1 \
  && pass "env server /health reachable" \
  || fail "env server not reachable at $ENV_URL — start it (uvicorn ... --port 8899)"

echo "== 2. vLLM server ($VLLM) =="
curl -fsS -m 5 "$VLLM/health" >/dev/null 2>&1 || fail "vLLM /health unreachable at $VLLM"
pass "vLLM /health reachable"

MODEL=$(curl -fsS -m 5 "$VLLM/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null) \
  || fail "vLLM /v1/models did not return a model"
pass "vLLM serving model: $MODEL"

# /metrics regression check: the prometheus-fastapi-instrumentator vs FastAPI
# 0.137 bug returns 500 here. Anything but 200 means the fastapi<0.137 pin is
# not in effect on this server.
CODE=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$VLLM/metrics")
[ "$CODE" = "200" ] && pass "vLLM /metrics returned 200 (fastapi pin OK)" \
  || fail "vLLM /metrics returned $CODE — apply fastapi<0.137 (see install_packages.sh)"

echo "== 3. vLLM generation + tool-calling =="
RESP=$(curl -fsS -m 90 "$VLLM/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @- <<JSON
{"model":"$MODEL",
 "messages":[{"role":"user","content":"Use the list_tools tool to see what is available."}],
 "tools":[{"type":"function","function":{"name":"list_tools",
   "description":"List the available MCP tools.","parameters":{"type":"object","properties":{}}}}],
 "max_tokens":1024,"temperature":0.6}
JSON
) || fail "vLLM /v1/chat/completions request failed"
echo "$RESP" | python3 -c '
import sys, json
d = json.load(sys.stdin)
ch = d["choices"][0]; m = ch["message"]
tcs = [t["function"]["name"] for t in (m.get("tool_calls") or [])]
print("  finish_reason:", ch["finish_reason"])
print("  tool_calls:", tcs)
print("  content[:160]:", (m.get("content") or "")[:160].replace(chr(10), " "))
' || fail "could not parse chat/completions response"
pass "vLLM produced a response"

echo "== 4. Trainer one-step (OOM / loop check) on GPU ${TRAINER_GPU:-1} =="
LOG="$(mktemp)"; OUT="$(mktemp -d)"
echo "  log: $LOG"
setsid env CUDA_VISIBLE_DEVICES="${TRAINER_GPU:-1}" \
  uv run accelerate launch \
    --config_file "$REPO_ROOT/open-env/configs/accelerate_fp8_single_gpu.yaml" \
    "$REPO_ROOT/open-env/openenv_awm_async_grpo.py" \
    --env-url "$ENV_URL" \
    --vllm-server-host "$VLLM_HOST" --vllm-server-port "$VLLM_PORT" \
    --num-generations 2 --gradient-accumulation-steps 1 \
    --max-turns 3 --max-completion-length 512 --thinking-token-budget 256 \
    --output-dir "$OUT" --logging-steps 1 --save-steps 1000000 \
    --wandb-name awm-smoke-test \
    >"$LOG" 2>&1 &
PID=$!

status="timeout"
deadline=$(( $(date +%s) + ${SMOKE_TIMEOUT:-900} ))
while :; do
  if grep -qiE "out of memory|outofmemoryerror|cuda error|cublas" "$LOG"; then status="oom"; break; fi
  if grep -qE "(reward|'loss'|train_runtime)" "$LOG"; then status="ok"; break; fi
  if grep -qE "Traceback \(most recent call last\)" "$LOG"; then status="error"; break; fi
  kill -0 "$PID" 2>/dev/null || { status="exited"; break; }
  [ "$(date +%s)" -ge "$deadline" ] && { status="timeout"; break; }
  sleep 5
done

# Tear down the whole accelerate process group.
kill -TERM -"$PID" 2>/dev/null; sleep 3; kill -KILL -"$PID" 2>/dev/null

case "$status" in
  ok)      pass "trainer reached a training step with no OOM";;
  oom)     echo "--- log tail ---"; tail -n 25 "$LOG"
           fail "trainer OOM — lower --thinking-token-budget/--max-completion-length, use --optim paged_adamw_8bit, or move to FSDP2";;
  error)   echo "--- log tail ---"; tail -n 30 "$LOG"; fail "trainer crashed (traceback above)";;
  exited)  echo "--- log tail ---"; tail -n 30 "$LOG"; fail "trainer exited before logging a step";;
  timeout) echo "--- log tail ---"; tail -n 20 "$LOG"
           fail "trainer did not reach a step within ${SMOKE_TIMEOUT:-900}s (env bottleneck or hang)";;
esac

rm -f "$LOG"; rm -rf "$OUT"
echo
pass "all smoke checks passed"
