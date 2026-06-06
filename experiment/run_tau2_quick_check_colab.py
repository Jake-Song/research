# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "vllm",
# ]
# ///
"""Install Tau2, launch Qwen with vLLM, and run a five-task Colab smoke test."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TAU2_REPOSITORY = "https://github.com/sierra-research/tau2-bench.git"
TAU2_TAG = "v1.0.0"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SERVED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
USER_MODEL = "gpt-4.1"
RUN_NAME = "tau2-qwen-airline-smoke"
OUTPUT_DIR = Path("/content/tau2_quick_check")
PORT = 8000
NUM_TASKS = 5
SEED = 20260606
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.9


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def install_tau2(tau2_dir: Path) -> None:
    run(["apt-get", "update"])
    run(["apt-get", "install", "-y", "portaudio19-dev"])
    run(["uv", "sync", "--extra", "voice"], cwd=tau2_dir)


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"vLLM exited with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                models = json.load(response)
            if any(item["id"] == SERVED_MODEL for item in models["data"]):
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit("Timed out waiting for vLLM.")


if not os.environ.get("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY for the hosted user simulator.")

output_dir = OUTPUT_DIR.resolve()
tau2_dir = output_dir / "tau2-bench"
uv_cache_dir = output_dir / "uv-cache"
server_log_path = output_dir / "vllm.log"
results_path = tau2_dir / "data" / "simulations" / RUN_NAME / "results.json"

if results_path.exists():
    raise SystemExit(f"Output already exists: {results_path}")
output_dir.mkdir(parents=True, exist_ok=True)
os.environ["UV_CACHE_DIR"] = str(uv_cache_dir)

if not (tau2_dir / ".git").exists():
    if tau2_dir.exists():
        shutil.rmtree(tau2_dir)
    run(
        [
            "git",
            "clone",
            "--branch",
            TAU2_TAG,
            "--depth",
            "1",
            TAU2_REPOSITORY,
            str(tau2_dir),
        ]
    )

install_tau2(tau2_dir)
run(["uv", "run", "tau2", "check-data"], cwd=tau2_dir)

base_url = f"http://127.0.0.1:{PORT}/v1"
server_log = server_log_path.open("w", encoding="utf-8")
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        MODEL,
        "--served-model-name",
        SERVED_MODEL,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "hermes",
    ],
    stdout=server_log,
    stderr=subprocess.STDOUT,
    text=True,
)

try:
    print(f"Starting vLLM for {MODEL}. Logs: {server_log_path}")
    wait_for_server(base_url, server)
    run(
        [
            "uv",
            "run",
            "tau2",
            "run",
            "--domain",
            "airline",
            "--agent-llm",
            f"openai/{SERVED_MODEL}",
            "--agent-llm-args",
            json.dumps(
                {
                    "api_base": base_url,
                    "api_key": "local",
                    "temperature": 0,
                }
            ),
            "--user-llm",
            USER_MODEL,
            "--user-llm-args",
            '{"temperature":0}',
            "--num-trials",
            "1",
            "--num-tasks",
            str(NUM_TASKS),
            "--seed",
            str(SEED),
            "--max-concurrency",
            "1",
            "--timeout",
            "600",
            "--verbose-logs",
            "--llm-log-mode",
            "all",
            "--save-to",
            RUN_NAME,
        ],
        cwd=tau2_dir,
    )
    if not results_path.is_file():
        raise SystemExit(f"Tau2 completed without producing {results_path}.")
    print(f"Tau2 results: {results_path}")
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    server_log.close()
