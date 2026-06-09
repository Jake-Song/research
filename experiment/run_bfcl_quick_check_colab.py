# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "bfcl-eval==2026.3.23",
#   "soundfile>=0.13.1",
#   "vllm",
# ]
# ///
"""Evaluate a Google Drive Qwen checkpoint with the 100-case BFCL quick check."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
import shutil
from pathlib import Path

import bfcl_eval
from bfcl_eval.constants.category_mapping import VERSION_PREFIX


BFCL_MODEL = "Qwen/Qwen3-4B-Think-FC"
SERVED_MODEL = "Qwen/Qwen3-4B"
MODEL = SERVED_MODEL
CATEGORIES = ("multi_turn_base", "irrelevance")
CASES_PER_CATEGORY = 50
SEED = 20260606

RUN_NAME = "think"
OUTPUT_DIR = Path("/content/bfcl_quick_check")
PORT = 8000
NUM_THREADS = 100
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 65536
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
MIN_P = 0


def log_tail(log_path: Path, line_count: int = 50) -> str:
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        lines = log_file.readlines()
    return "".join(lines[-line_count:]).rstrip()


def wait_for_server(
    base_url: str,
    process: subprocess.Popen,
    log_path: Path,
    server_log,
    timeout: int = 900,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            server_log.flush()
            details = log_tail(log_path)
            raise SystemExit(
                f"vLLM exited with code {process.returncode}.\n"
                f"Last lines from {log_path}:\n{details}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2)
    server_log.flush()
    details = log_tail(log_path)
    raise SystemExit(
        f"Timed out waiting for vLLM.\nLast lines from {log_path}:\n{details}"
    )


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


def run(command: list[str], env: dict[str, str], log_path: Path) -> None:
    print("$", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            raise SystemExit(f"Could not start command: {exc}") from exc

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()

    if return_code:
        details = log_tail(log_path)
        raise SystemExit(
            f"Command exited with code {return_code}.\n"
            f"Last lines from {log_path}:\n{details}"
        )


def score(run_dir: Path, category: str) -> dict:
    files = list(
        (run_dir / "score").rglob(f"{VERSION_PREFIX}_{category}_score.json")
    )
    if len(files) != 1:
        raise SystemExit(f"Expected one {category} score file, found {len(files)}.")
    with files[0].open(encoding="utf-8") as lines:
        return json.loads(next(line for line in lines if line.strip()))


# if NUM_THREADS < 1:
#     raise SystemExit("NUM_THREADS must be at least 1.")

run_dir = (OUTPUT_DIR / RUN_NAME).resolve()
if run_dir.exists():
    print(f"Output already exists, removing: {run_dir}")
    shutil.rmtree(run_dir)
run_dir.mkdir(parents=True)

(run_dir / "test_case_ids_to_generate.json").write_text(
    json.dumps(select_ids(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

base_url = f"http://127.0.0.1:{PORT}/v1"
server_log_path = run_dir / "vllm.log"
server_log = server_log_path.open("w", encoding="utf-8")

env = os.environ.copy()
env.update(
    {
        "BFCL_PROJECT_ROOT": str(run_dir),
        "REMOTE_OPENAI_BASE_URL": base_url,
        "REMOTE_OPENAI_API_KEY": "EMPTY",
        "REMOTE_OPENAI_TOKENIZER_PATH": SERVED_MODEL,
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
        "OPENAI_API_KEY": "EMPTY"
    }
)

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
        json.dumps(
            {
                "rope_type": "yarn",
                "factor": 2.0,
                "original_max_position_embeddings": 32768,
            }
        ),
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
    env=env,
    stdout=server_log,
    stderr=subprocess.STDOUT,
    text=True,
)

try:
    print(f"Starting vLLM for {MODEL}. Logs: {server_log_path}")
    wait_for_server(base_url, server, server_log_path, server_log)

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
        run_dir / "generate.log",
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
        run_dir / "evaluate.log",
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
