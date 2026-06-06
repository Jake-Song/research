# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "bfcl-eval==2026.3.23",
#   "soundfile>=0.13.1",
# ]
# ///
"""Run the 100-case BFCL quick check against an existing vLLM endpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import bfcl_eval
from bfcl_eval.constants.category_mapping import VERSION_PREFIX


BFCL_VERSION = "2026.3.23"
BFCL_MODEL = "Qwen/Qwen3-4B-Instruct-2507-FC"
SERVED_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 20260606
CASES_PER_CATEGORY = 50
CATEGORIES = ("multi_turn_base", "irrelevance")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 50 multi_turn_base and 50 irrelevance BFCL cases against "
            "an existing OpenAI-compatible vLLM endpoint."
        )
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Unique artifact name, for example base, step-12, or step-24.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default="EMPTY",
        help="API key for the endpoint. It is not written to run metadata.",
    )
    parser.add_argument(
        "--tokenizer-path",
        default=SERVED_MODEL,
        help="Hugging Face ID or local path used by BFCL for prompt tokenization.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of concurrent BFCL inference requests.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "bfcl_quick_check_runs",
        help="Parent directory for per-run artifacts.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not RUN_NAME_RE.fullmatch(args.run_name):
        raise SystemExit(
            "--run-name must start with an alphanumeric character and contain "
            "only letters, numbers, dots, underscores, or hyphens."
        )
    if args.num_threads < 1:
        raise SystemExit("--num-threads must be at least 1.")


def get_endpoint_models(base_url: str, api_key: str) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"vLLM endpoint is not ready at {base_url}: {exc}") from exc

    model_ids = [
        item.get("id")
        for item in payload.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if SERVED_MODEL not in model_ids:
        available = ", ".join(model_ids) if model_ids else "<none>"
        raise SystemExit(
            f"Endpoint must serve the model alias {SERVED_MODEL!r}; found: {available}. "
            "Start vLLM with --served-model-name "
            f"{SERVED_MODEL}."
        )
    return model_ids


def load_category_ids(category: str) -> list[str]:
    package_dir = Path(bfcl_eval.__file__).resolve().parent
    data_path = package_dir / "data" / f"{VERSION_PREFIX}_{category}.json"
    if not data_path.is_file():
        raise SystemExit(f"BFCL dataset file is missing: {data_path}")

    ids = []
    with data_path.open(encoding="utf-8") as data_file:
        for line_number, line in enumerate(data_file, 1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Invalid JSON in {data_path} at line {line_number}: {exc}"
                ) from exc
            if isinstance(entry.get("id"), str):
                ids.append(entry["id"])

    if len(ids) < CASES_PER_CATEGORY:
        raise SystemExit(
            f"BFCL category {category!r} has {len(ids)} entries; "
            f"{CASES_PER_CATEGORY} are required."
        )
    if len(ids) != len(set(ids)):
        raise SystemExit(f"BFCL category {category!r} contains duplicate IDs.")
    return ids


def select_test_ids() -> dict[str, list[str]]:
    rng = random.Random(SEED)
    selected = {}
    for category in CATEGORIES:
        ids = load_category_ids(category)
        selected[category] = sorted(rng.sample(ids, CASES_PER_CATEGORY))
    return selected


def run_command(
    command: list[str], env: dict[str, str], log_path: Path
) -> None:
    print(f"$ {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise SystemExit(
            f"Command failed with exit code {return_code}. See {log_path}."
        )


def read_score_header(score_root: Path, category: str) -> dict:
    matches = list(score_root.rglob(f"{VERSION_PREFIX}_{category}_score.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one score file for {category!r}, found {len(matches)}."
        )
    with matches[0].open(encoding="utf-8") as score_file:
        for line in score_file:
            if line.strip():
                header = json.loads(line)
                return {
                    "accuracy": header["accuracy"],
                    "correct_count": header["correct_count"],
                    "total_count": header["total_count"],
                    "score_file": str(matches[0]),
                }
    raise SystemExit(f"Score file is empty: {matches[0]}")


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_dir = args.output_root.resolve() / args.run_name
    if run_dir.exists():
        raise SystemExit(
            f"Run directory already exists: {run_dir}. Use a new --run-name."
        )

    model_ids = get_endpoint_models(args.base_url, args.api_key)
    selected_ids = select_test_ids()

    run_dir.mkdir(parents=True)
    ids_path = run_dir / "test_case_ids_to_generate.json"
    ids_path.write_text(
        json.dumps(selected_ids, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "bfcl_eval_version": BFCL_VERSION,
        "bfcl_data_prefix": VERSION_PREFIX,
        "bfcl_model_handler": BFCL_MODEL,
        "served_model": SERVED_MODEL,
        "endpoint": args.base_url,
        "endpoint_models": model_ids,
        "tokenizer_path": args.tokenizer_path,
        "seed": SEED,
        "cases_per_category": CASES_PER_CATEGORY,
        "categories": list(CATEGORIES),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env.update(
        {
            "BFCL_PROJECT_ROOT": str(run_dir),
            "REMOTE_OPENAI_BASE_URL": args.base_url,
            "REMOTE_OPENAI_API_KEY": args.api_key,
            "REMOTE_OPENAI_TOKENIZER_PATH": args.tokenizer_path,
        }
    )

    generate_command = [
        sys.executable,
        "-m",
        "bfcl_eval",
        "generate",
        "--model",
        BFCL_MODEL,
        "--run-ids",
        "--skip-server-setup",
        "--num-threads",
        str(args.num_threads),
        "--result-dir",
        "result",
    ]
    run_command(generate_command, env, run_dir / "generate.log")

    evaluate_command = [
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
    ]
    run_command(evaluate_command, env, run_dir / "evaluate.log")

    summary = {
        category: read_score_header(run_dir / "score", category)
        for category in CATEGORIES
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nQuick-check scores")
    for category, score in summary.items():
        print(
            f"- {category}: {score['accuracy']:.2%} "
            f"({score['correct_count']}/{score['total_count']})"
        )
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
