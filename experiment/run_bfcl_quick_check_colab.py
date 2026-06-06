# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "bfcl-eval==2026.3.23",
#   "soundfile>=0.13.1",
#   "vllm",
# ]
# ///
"""Launch Qwen with vLLM and run the 100-case BFCL quick check."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import bfcl_eval
from bfcl_eval.constants.category_mapping import VERSION_PREFIX


BFCL_MODEL = "Qwen/Qwen3-4B-FC"
SERVED_MODEL = "Qwen/Qwen3-4B"
CATEGORIES = ("multi_turn_base", "irrelevance")
CASES_PER_CATEGORY = 50
SEED = 20260606

MODEL = SERVED_MODEL
RUN_NAME = "base"
OUTPUT_DIR = Path("/content/bfcl_quick_check")
PORT = 8000
NUM_THREADS = 1
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: int = 900) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(f"vLLM exited with code {process.returncode}.")
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    raise SystemExit("Timed out waiting for vLLM.")


def category_ids(category: str) -> list[str]:
    data_file = (
        Path(bfcl_eval.__file__).resolve().parent
        / "data"
        / f"{VERSION_PREFIX}_{category}.json"
    )
    with data_file.open(encoding="utf-8") as lines:
        return [json.loads(line)["id"] for line in lines if line.strip()]


def select_ids() -> dict[str, list[str]]:
    rng = random.Random(SEED)
    return {
        category: sorted(rng.sample(category_ids(category), CASES_PER_CATEGORY))
        for category in CATEGORIES
    }


def run(command: list[str], env: dict[str, str]) -> None:
    print("$", " ".join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def score(run_dir: Path, category: str) -> dict:
    files = list(
        (run_dir / "score").rglob(f"{VERSION_PREFIX}_{category}_score.json")
    )
    if len(files) != 1:
        raise SystemExit(f"Expected one {category} score file, found {len(files)}.")
    with files[0].open(encoding="utf-8") as lines:
        return json.loads(next(line for line in lines if line.strip()))


if NUM_THREADS < 1:
    raise SystemExit("NUM_THREADS must be at least 1.")

run_dir = (OUTPUT_DIR / RUN_NAME).resolve()
if run_dir.exists():
    raise SystemExit(f"Output already exists: {run_dir}")
run_dir.mkdir(parents=True)

(run_dir / "test_case_ids_to_generate.json").write_text(
    json.dumps(select_ids(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

base_url = f"http://127.0.0.1:{PORT}/v1"
server_log = (run_dir / "vllm.log").open("w", encoding="utf-8")
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
        "--override-generation-config",
        json.dumps(
            {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "min_p": MIN_P,
            }
        ),
    ],
    stdout=server_log,
    stderr=subprocess.STDOUT,
    text=True,
)

env = os.environ.copy()
env.update(
    {
        "BFCL_PROJECT_ROOT": str(run_dir),
        "REMOTE_OPENAI_BASE_URL": base_url,
        "REMOTE_OPENAI_API_KEY": "EMPTY",
        "REMOTE_OPENAI_TOKENIZER_PATH": SERVED_MODEL,
    }
)

try:
    print(f"Starting vLLM for {MODEL}. Logs: {run_dir / 'vllm.log'}")
    wait_for_server(base_url, server)

    run(
        [
            sys.executable,
            "-m",
            "bfcl_eval",
            "generate",
            "--model",
            BFCL_MODEL,
            "--run-ids",
            "--skip-server-setup",
            "--temperature",
            str(TEMPERATURE),
            "--num-threads",
            str(NUM_THREADS),
            "--result-dir",
            "result",
        ],
        env,
    )
    run(
        [
            sys.executable,
            "-m",
            "bfcl_eval",
            "evaluate",
            "--model",
            BFCL_MODEL,
            "--test-category",
            ",".join(CATEGORIES),
            "--result-dir",
            "result",
            "--score-dir",
            "score",
            "--partial-eval",
        ],
        env,
    )

    summary = {category: score(run_dir, category) for category in CATEGORIES}
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for category, result in summary.items():
        print(
            f"{category}: {result['accuracy']:.2%} "
            f"({result['correct_count']}/{result['total_count']})"
        )
    print(f"Artifacts: {run_dir}")
finally:
    server.terminate()
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait()
    server_log.close()
