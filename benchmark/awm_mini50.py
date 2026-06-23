"""AWM-mini-50: native Agent World Model quick benchmark.

Runs a fixed, seeded subset of AWM tasks (one task each from N distinct
scenarios) against an OpenAI-compatible chat endpoint, scores them with the
deterministic *code* verifier, and writes a JSON report.

Setup:
    # Terminal 1 - AWM env server (from the OpenEnv checkout)
    cd ~/OpenEnv
    PYTHONPATH=src:envs uv run uvicorn \
        envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2 - run the benchmark (agent model = OpenAI-compatible endpoint)
    export ENDPOINT_URL="https://YOUR_ENDPOINT/v1"
    export OPENAI_API_KEY="..."
    export AWM_EXAMPLE_AGENT_MODEL="gpt-5"
    PYTHONPATH=$HOME/OpenEnv/envs uv run python benchmark/awm_mini50.py

Connect to the hosted Space instead of a local server with
``--base-url https://chilled-agent-world-model-env.hf.space``.
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv, AWMObservation
from agent_world_model_env.server.prompts import DEFAULT_SYSTEM_PROMPT

load_dotenv()


def parse_tool_call(content: str) -> dict | None:
    """Extract the first <tool_call> JSON block from LLM output."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        data = data[0] if data else None
    if not isinstance(data, dict) or "name" not in data:
        return None
    return data


def format_tools(tools) -> str:
    """Format Tool objects into a readable string for the LLM."""
    lines = [f"Available MCP Tools ({len(tools)} tools):", "=" * 60]
    for i, t in enumerate(tools, 1):
        lines.append(f"{i}. {t.name}")
        lines.append(f"   Description: {t.description}")
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        if props:
            lines.append("   Parameters:")
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                lines.append(
                    f"     - {pname}: {pinfo.get('type', 'any')}{req} — {pinfo.get('description', '')}"
                )
        else:
            lines.append("   Parameters: None")
        lines.append("")
    return "\n".join(lines)


async def run_episode(env, llm, model, scenario, task_idx, args) -> dict:
    """Run one task end-to-end and return its per-task record."""
    t0 = time.perf_counter()
    reset: StepResult[AWMObservation] = await env.reset(scenario=scenario, task_idx=task_idx)
    task = reset.observation.task

    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    tool_calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    last_content = ""

    for _ in range(args.max_turns):
        response = await llm.chat.completions.create(
            model=model,
            messages=messages,
            temperature=args.temperature,
            max_completion_tokens=args.max_tokens,
        )
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens or 0
            completion_tokens += response.usage.completion_tokens or 0
        content = response.choices[0].message.content or ""
        last_content = content
        messages.append({"role": "assistant", "content": content})

        tc = parse_tool_call(content)
        if not tc:
            break  # no tool call -> treat as final answer

        name = tc["name"]
        arguments = tc.get("arguments") or {}
        tool_calls += 1

        if name == "list_tools":
            res = await env.step(ListToolsAction())
            tool_response = format_tools(res.observation.tools)
        elif name == "call_tool":
            tool_name = arguments.get("tool_name", "")
            inner = arguments.get("arguments", "{}")
            if isinstance(inner, str):
                try:
                    inner = json.loads(inner)
                except json.JSONDecodeError:
                    inner = {}
            if not isinstance(inner, dict):
                inner = {}
            res = await env.step(CallToolAction(tool_name=tool_name, arguments=inner))
            obs = res.observation
            if getattr(obs, "tool_result", None) is not None:
                tool_response = (
                    obs.tool_result
                    if isinstance(obs.tool_result, str)
                    else json.dumps(obs.tool_result, ensure_ascii=False)
                )
            elif getattr(obs, "error", None):
                tool_response = f"Error: {obs.error}"
            else:
                tool_response = json.dumps(obs.model_dump(), ensure_ascii=False)
        else:
            tool_response = f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'."

        messages.append({"role": "user", "content": f"Tool response:\n{tool_response}"})

    # Deterministic code verifier.
    verify: StepResult[AWMObservation] = await env.step(
        CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": "code", "final_answer": last_content},
        )
    )
    reward = verify.reward
    reward_type = verify.observation.reward_type

    await env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))

    return {
        "scenario": scenario,
        "task_idx": task_idx,
        "reward": reward,
        "reward_type": reward_type,
        "success": reward_type == "complete",
        "tool_calls": tool_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "latency_s": round(time.perf_counter() - t0, 3),
    }


async def main(args):
    model = args.model or os.environ.get("AWM_EXAMPLE_AGENT_MODEL")
    if not model:
        raise SystemExit("Set --model or AWM_EXAMPLE_AGENT_MODEL.")
    llm = AsyncOpenAI(
        base_url=os.environ["ENDPOINT_URL"],
        api_key=os.environ["OPENAI_API_KEY"],
    )

    async with AWMEnv(base_url=args.base_url) as env:
        listing = await env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = listing.observation.scenarios

        # Fixed-seed selection: N distinct scenarios, one task each.
        rng = random.Random(args.seed)
        chosen = rng.sample(scenarios, min(args.num_scenarios, len(scenarios)))
        plan = [(s["name"], rng.randrange(len(s["tasks"]))) for s in chosen]

        print(f"Running {len(plan)} tasks against {model} @ {args.base_url}")
        records = []
        for i, (scenario, task_idx) in enumerate(plan, 1):
            try:
                rec = await run_episode(env, llm, model, scenario, task_idx, args)
            except Exception as e:  # keep the run going across 50 tasks
                rec = {
                    "scenario": scenario,
                    "task_idx": task_idx,
                    "reward": None,
                    "reward_type": "error",
                    "success": False,
                    "tool_calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "latency_s": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            records.append(rec)
            mark = "✓" if rec["success"] else "·"
            print(f"  [{i}/{len(plan)}] {mark} {scenario}#{task_idx} "
                  f"reward={rec['reward']} type={rec['reward_type']} "
                  f"tools={rec['tool_calls']} {rec['latency_s']}s")

    n = len(records)
    reward_type_counts: dict[str, int] = {}
    for r in records:
        reward_type_counts[r["reward_type"]] = reward_type_counts.get(r["reward_type"], 0) + 1
    successes = sum(r["success"] for r in records)
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "base_url": args.base_url,
        "seed": args.seed,
        "num_scenarios": n,
        "max_turns": args.max_turns,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "success_rate": round(successes / n, 4) if n else 0.0,
        "reward_type_counts": reward_type_counts,
        "avg_tool_calls": round(sum(r["tool_calls"] for r in records) / n, 2) if n else 0.0,
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in records),
        "total_completion_tokens": sum(r["completion_tokens"] for r in records),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "tasks": records}, indent=2))

    print("\n=== AWM-mini-50 summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {out}")


def parse_args():
    p = argparse.ArgumentParser(description="AWM-mini-50 quick benchmark")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    p.add_argument("--base-url", default=os.environ.get("AWM_BASE_URL", "http://localhost:8899"))
    p.add_argument("--model", default=None, help="agent model (default AWM_EXAMPLE_AGENT_MODEL)")
    p.add_argument("--num-scenarios", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--output", default=str(Path(__file__).parent / "results" / f"awm_mini50_{ts}.json"))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
