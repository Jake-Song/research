# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "vllm",
# ]
# ///
"""Colab: self-serve a Qwen3-4B-Thinking checkpoint and run MCPMark filesystem/easy-10.

Standalone variant of run_mcpmark_quick_check.py for a single-GPU Colab runtime: it
starts vLLM itself (with tool-calling + the thinking reasoning parser enabled),
waits for it, runs the 10 filesystem/easy MCPMark tasks, then stops vLLM. MCPMark is
cloned at a pinned commit into ~/.cache/research/mcpmark and installed into its own
uv venv so its deps never clash with vLLM's.

Colab setup (run once in a cell):
    !apt-get -qq install -y unzip nodejs npm
    !pip install -q uv
    !uv run experiment/run_mcpmark_quick_check_colab.py --run-name base
    # trained checkpoint (HF id, local path, or Drive path):
    !uv run experiment/run_mcpmark_quick_check_colab.py \
        --run-name awm-trained --model <hf-id-or-path>

MCPMark drives the model with the native OpenAI tools API against the real
filesystem MCP tools (npx @modelcontextprotocol/server-filesystem) — a different
tool interface than AWM's call_tool wrapper, so this measures OOD transfer.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

MCPMARK_REPO = "https://github.com/eval-sys/mcpmark"
MCPMARK_SHA = "cd45b7f57923b9b3985467f5139927575f83141c"

CACHE_ROOT = Path.home() / ".cache" / "research" / "mcpmark"
CLONE_DIR = CACHE_ROOT / "repo"
VENV_DIR = CACHE_ROOT / "venv"
TEST_ROOT = CACHE_ROOT / "test_environments"

SERVED_MODEL = "Qwen/Qwen3-4B-Thinking-2507"  # stable alias; keep constant across checkpoints
MODEL_KEY = "qwen3-4b-thinking-awm"  # MCPMark short name
MCP_SERVICE = "filesystem"
TASK_SUITE = "easy"
EXPECTED_TASKS = 10

PORT = 8000
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768

RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab: self-serve a checkpoint and run MCPMark filesystem/easy-10.",
    )
    parser.add_argument("--run-name", required=True, help="Artifact name, e.g. base or awm-trained.")
    parser.add_argument(
        "--model",
        default=SERVED_MODEL,
        help="Checkpoint to serve: HF id, local path, or Drive path. Defaults to the base model.",
    )
    parser.add_argument("--tasks", default="all", help='"all", "category", or "category/task".')
    parser.add_argument("--k", type=int, default=1, help="Runs per task (default: 1 => pass@1).")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-task agent timeout (s).")
    parser.add_argument("--port", type=int, default=PORT, help="Port for the local vLLM server.")
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION,
        help="vLLM GPU memory fraction.",
    )
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN, help="vLLM max model length.")
    parser.add_argument(
        "--output-root", type=Path, default=Path("/content/mcpmark_quick_check"),
        help="Parent directory for per-run artifacts.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not RUN_NAME_RE.fullmatch(args.run_name):
        raise SystemExit(
            "--run-name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )
    if args.k < 1:
        raise SystemExit("--k must be at least 1.")


def check_prerequisites() -> None:
    missing = [tool for tool in ("git", "uv", "npx", "wget", "unzip") if shutil.which(tool) is None]
    if missing:
        raise SystemExit(
            f"Missing required tools on PATH: {', '.join(missing)}. In Colab run: "
            "!apt-get -qq install -y unzip nodejs npm && pip install -q uv"
        )


def log_tail(log_path: Path, line_count: int = 50) -> str:
    with log_path.open(encoding="utf-8", errors="replace") as log_file:
        return "".join(log_file.readlines()[-line_count:]).rstrip()


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None,
        log_path: Path | None = None) -> None:
    print("$", " ".join(command), flush=True)
    log_file = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        process = subprocess.Popen(
            command, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            if log_file:
                log_file.write(line)
        return_code = process.wait()
    finally:
        if log_file:
            log_file.close()
    if return_code:
        where = f" See {log_path}." if log_path else ""
        raise SystemExit(f"Command failed with exit code {return_code}.{where}")


def wait_for_server(base_url: str, process: subprocess.Popen, log_path: Path, timeout: int = 1200) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise SystemExit(
                f"vLLM exited with code {process.returncode}.\n"
                f"Last lines from {log_path}:\n{log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(3)
    raise SystemExit(f"Timed out waiting for vLLM.\nLast lines from {log_path}:\n{log_tail(log_path)}")


def ensure_mcpmark() -> Path:
    """Clone MCPMark at the pinned SHA and install it into an isolated venv. Idempotent."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if not (CLONE_DIR / ".git").is_dir():
        run(["git", "clone", MCPMARK_REPO, str(CLONE_DIR)])
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=CLONE_DIR, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != MCPMARK_SHA:
        run(["git", "fetch", "--depth", "1", "origin", MCPMARK_SHA], cwd=CLONE_DIR)
        run(["git", "checkout", "--force", MCPMARK_SHA], cwd=CLONE_DIR)
    venv_python = VENV_DIR / "bin" / "python"
    if not venv_python.is_file():
        run(["uv", "venv", str(VENV_DIR)])
        run(["uv", "pip", "install", "--python", str(venv_python), "-e", str(CLONE_DIR)])
    return venv_python


def ensure_model_registered() -> None:
    """Idempotently add the served model to MCPMark's MODEL_CONFIGS (openai/ prefix + base_url)."""
    config_path = CLONE_DIR / "src" / "model_config.py"
    text = config_path.read_text(encoding="utf-8")
    if f'"{MODEL_KEY}"' in text:
        return
    entry = (
        f'        "{MODEL_KEY}": {{\n'
        f'            "provider": "openai",\n'
        f'            "api_key_var": "OPENAI_API_KEY",\n'
        f'            "base_url_var": "OPENAI_BASE_URL",\n'
        f'            "litellm_input_model_name": "openai/{SERVED_MODEL}",\n'
        f'        }},\n'
    )
    marker = "    MODEL_CONFIGS = {\n"
    if marker not in text:
        raise SystemExit(f"Could not find MODEL_CONFIGS marker in {config_path}.")
    config_path.write_text(text.replace(marker, marker + entry, 1), encoding="utf-8")


def write_mcp_env(base_url: str) -> None:
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    (CLONE_DIR / ".mcp_env").write_text(
        f"OPENAI_BASE_URL={base_url}\nOPENAI_API_KEY=EMPTY\nFILESYSTEM_TEST_ROOT={TEST_ROOT}\n",
        encoding="utf-8",
    )


def collect_task_results(exp_dir: Path) -> dict[str, dict]:
    """Read every meta.json under exp_dir into {task_name: {...}}; a task passes if any run passed."""
    results: dict[str, dict] = {}
    for meta_path in sorted(exp_dir.rglob("meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        task_name = meta.get("task_name")
        if not task_name:
            continue
        exec_result = meta.get("execution_result", {})
        success = bool(exec_result.get("success"))
        prior = results.get(task_name)
        if prior is None or (success and not prior["success"]):
            results[task_name] = {
                "success": success,
                "turn_count": meta.get("turn_count"),
                "verification_error": exec_result.get("verification_error"),
                "meta_file": str(meta_path.relative_to(exp_dir)),
            }
    return results


def main() -> None:
    args = parse_args()
    validate_args(args)
    check_prerequisites()

    run_dir = args.output_root.resolve() / args.run_name
    if run_dir.exists():
        print(f"Output already exists, removing: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    venv_python = ensure_mcpmark()
    ensure_model_registered()

    base_url = f"http://127.0.0.1:{args.port}/v1"
    server_log_path = run_dir / "vllm.log"
    server_log = server_log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", args.model,
            "--served-model-name", SERVED_MODEL,
            "--port", str(args.port),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--max-model-len", str(args.max_model_len),
            "--enable-auto-tool-choice",
            "--tool-call-parser", "hermes",
            "--reasoning-parser", "deepseek_r1",
            "--reasoning-config",
            json.dumps({"reasoning_start_str": "<think>", "reasoning_end_str": "</think>"}),
        ],
        stdout=server_log, stderr=subprocess.STDOUT, text=True,
    )
    try:
        print(f"Starting vLLM for {args.model} (alias {SERVED_MODEL}). Logs: {server_log_path}")
        wait_for_server(base_url, server, server_log_path)
        write_mcp_env(base_url)

        exp_dir = CLONE_DIR / "results" / args.run_name
        if exp_dir.exists():
            shutil.rmtree(exp_dir)
        env = os.environ.copy()
        env.update({
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": "EMPTY",
            "FILESYSTEM_TEST_ROOT": str(TEST_ROOT),
        })
        run(
            [
                str(venv_python), "-m", "pipeline",
                "--mcp", MCP_SERVICE,
                "--task-suite", TASK_SUITE,
                "--tasks", args.tasks,
                "--k", str(args.k),
                "--models", MODEL_KEY,
                "--exp-name", args.run_name,
                "--timeout", str(args.timeout),
                "--output-dir", str(CLONE_DIR / "results"),
            ],
            cwd=CLONE_DIR, env=env, log_path=run_dir / "pipeline.log",
        )

        task_results = collect_task_results(exp_dir)
        if not task_results:
            raise SystemExit(f"No task results under {exp_dir}. See {run_dir / 'pipeline.log'}.")
        if args.tasks == "all" and len(task_results) != EXPECTED_TASKS:
            print(f"WARNING: expected {EXPECTED_TASKS} tasks, found {len(task_results)}.")

        total = len(task_results)
        solved = sum(1 for r in task_results.values() if r["success"])
        summary = {
            "pass_at_1": solved / total if total else 0.0,
            "solved": solved,
            "total": total,
            "tasks": task_results,
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        metadata = {
            "mcpmark_repo": MCPMARK_REPO,
            "mcpmark_sha": MCPMARK_SHA,
            "served_model": SERVED_MODEL,
            "model": args.model,
            "model_key": MODEL_KEY,
            "endpoint": base_url,
            "mcp_service": MCP_SERVICE,
            "task_suite": TASK_SUITE,
            "tasks": args.tasks,
            "k": args.k,
            "timeout": args.timeout,
        }
        (run_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print(f"\nMCPMark {MCP_SERVICE}/{TASK_SUITE} quick check ({args.run_name})")
        print(f"- pass@1: {summary['pass_at_1']:.2%} ({solved}/{total})")
        for task_name, result in sorted(task_results.items()):
            print(f"  {'PASS' if result['success'] else 'FAIL'}  {task_name}")
        print(f"Artifacts: {run_dir}")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        server_log.close()


if __name__ == "__main__":
    main()
