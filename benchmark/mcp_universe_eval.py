"""OOD transfer eval — MCP-Universe, faithful to the benchmark's native harness.

Third off-distribution benchmark from experiment/awm_transfer_experiment_plan.md
(after BFCLv3 and tau2). Unlike the on-distribution AWM eval, this one does NOT
reuse the AWM training prompt / `list_tools`+`call_tool` wrapper — it drives the
model through MCP-Universe's OWN function_call agent and real MCP servers, so the
harness can be calibrated to the *published* Qwen3-4B number (the plan's sanity
gate). "Eval parity" here = base (C0) and AWM-trained (C1) run through the exact
same config; only the served weights behind the vLLM endpoint change.

We reuse the upstream per-domain benchmark configs verbatim (agent instruction,
servers, task list) and override only the `llm` doc so it points at our vLLM
checkpoint with fixed decoding. Output mirrors experiment/awm_eval_ondist.py's
{"summary", "tasks"} shape so a later aggregator treats every OOD eval the same.

Needs MCP-Universe **editable-installed from a clone** (not a plain wheel): its
BenchmarkRunner resolves task-JSON paths against the installed `mcpuniverse`
package dir, so the config data, the task data, and this script must all see the
same tree. Also needs MCP-Universe's live MCP servers + per-domain API keys
(finance/yfinance, GOOGLE_MAPS_API_KEY); put keys in a `.env` (the package calls
`load_dotenv()`) or `os.environ`.

Colab (single GPU) — run this script per model as a subprocess, exactly like
awm_eval_ondist.py. In a notebook:

    !git clone https://github.com/SalesforceAIResearch/MCP-Universe.git
    !uv pip -q install -e MCP-Universe        # editable: keeps config+task paths consistent
    # ...set per-domain API keys, then serve the checkpoint on port 8000:
    #   vllm serve <model> --served-model-name m --enable-auto-tool-choice \
    #     --tool-call-parser hermes --reasoning-parser deepseek_r1 --max-model-len 24576
    !python experiment/mcp_universe_eval.py --model m \
        --domains financial_analysis --limit 3 --output base.json
    # then serve the finetuned checkpoint and run again with --output ft.json

The default --endpoint (http://localhost:8000/v1) matches that vLLM serve, so on
Colab you only pass --model / --domains / --output.
"""

import argparse
import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def upstream_config(domain: str) -> Path:
    """Path to MCP-Universe's own per-domain benchmark config (installed pkg)."""
    try:
        import mcpuniverse
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "MCP-Universe is not installed. Clone and editable-install it:\n"
            "    git clone https://github.com/SalesforceAIResearch/MCP-Universe.git\n"
            "    uv pip install -e MCP-Universe"
        ) from e

    path = Path(mcpuniverse.__file__).parent / "benchmark" / "configs" / "mcpuniverse" / f"{domain}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No MCP-Universe config for domain '{domain}' at {path}. The installed "
            "mcpuniverse package is missing its config/task data — install it as an "
            "*editable* clone (`uv pip install -e MCP-Universe`), not a plain wheel, "
            "so BenchmarkRunner can also resolve the task JSONs."
        )
    return path


def build_domain_config(domain: str, args) -> str:
    """Copy the upstream domain config, repoint its llm at our vLLM checkpoint,
    optionally cap tasks, and write it to a temp file. Task paths inside stay
    relative — BenchmarkRunner resolves them against its own configs dir, not
    this file's location — so the temp file can live anywhere."""
    docs = list(yaml.safe_load_all(upstream_config(domain).read_text()))
    for doc in docs:
        kind = str(doc.get("kind", "")).lower()
        if kind == "llm":
            # Preserve the doc's name (agents reference it) and openai type;
            # swap in our checkpoint + fixed decoding for C0/C1 parity.
            doc["spec"]["config"] = {
                "model_name": args.model,
                "base_url": args.endpoint,
                "api_key": os.environ.get("OPENAI_API_KEY", "EMPTY"),
                "temperature": args.temperature,
                "max_completion_tokens": args.max_tokens,
            }
        elif kind == "benchmark" and args.limit:
            doc["spec"]["tasks"] = doc["spec"]["tasks"][: args.limit]

    fd, path = tempfile.mkstemp(prefix=f"mcpu_{domain}_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump_all(docs, f, sort_keys=False)
    return path


def reduce_results(domain: str, results) -> list[dict]:
    """Flatten a domain's BenchmarkResult(s) into per-task records. A task passes
    iff all of its evaluators pass (EvaluationResult.passed)."""
    records = []
    for br in results:
        for task_path, res in br.task_results.items():
            evals = res.get("evaluation_results", []) or []
            total = len(evals)
            passed = sum(bool(getattr(e, "passed", False)) for e in evals)
            records.append({
                "domain": domain,
                "task": Path(task_path).stem,
                "success": total > 0 and passed == total,
                "evaluators_passed": passed,
                "evaluators_total": total,
                "trace_id": br.task_trace_ids.get(task_path),
            })
    return records


async def run_domain(domain: str, args) -> list[dict]:
    from mcpuniverse.benchmark.runner import BenchmarkRunner
    from mcpuniverse.tracer.collectors import MemoryCollector

    config_path = build_domain_config(domain, args)
    try:
        runner = BenchmarkRunner(config_path)
        results = await runner.run(trace_collector=MemoryCollector())
        return reduce_results(domain, results)
    finally:
        os.unlink(config_path)


async def main(args):
    # MCP-Universe's openai LLM type reads these; set them so the client hits vLLM.
    os.environ["OPENAI_BASE_URL"] = args.endpoint
    os.environ.setdefault("OPENAI_API_KEY", "EMPTY")

    print(f"Running MCP-Universe {args.domains} against {args.model} @ {args.endpoint}")
    records: list[dict] = []
    for domain in args.domains:
        try:
            recs = await run_domain(domain, args)
        except Exception as e:  # one dead domain (server/keys) must not sink the rest
            recs = [{"domain": domain, "task": None, "success": False,
                     "evaluators_passed": 0, "evaluators_total": 0, "trace_id": None,
                     "error": f"{type(e).__name__}: {e}"}]
        succ = sum(r["success"] for r in recs)
        print(f"  {domain}: {succ}/{len(recs)} solved")
        records.extend(recs)

    per_domain = {}
    for d in args.domains:
        drecs = [r for r in records if r["domain"] == d]
        n = len(drecs)
        per_domain[d] = round(sum(r["success"] for r in drecs) / n, 4) if n else 0.0

    n = len(records)
    successes = sum(r["success"] for r in records)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": args.endpoint,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "domains": args.domains,
        "num_tasks": n,
        "successes": successes,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "per_domain_success": per_domain,
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "tasks": records}, indent=2))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="MCP-Universe OOD eval (native harness, C0/C1 parity)")
    p.add_argument("--model", required=True, help="served model name on the vLLM endpoint")
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    p.add_argument("--domains", nargs="+", default=["financial_analysis", "location_navigation"])
    p.add_argument("--limit", type=int, default=0, help="cap tasks per domain (0 = all)")
    p.add_argument("--temperature", type=float, default=1.0, help="held fixed for C0/C1 parity")
    p.add_argument("--max-tokens", type=int, default=10000, help="max_completion_tokens (upstream default)")
    p.add_argument("--output", default="mcp_universe_result.json")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
