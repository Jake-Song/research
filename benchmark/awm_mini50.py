"""AWM-mini-50: held-out native Agent World Model quick benchmark.

Runs a fixed, seeded held-out subset of AWM tasks (one task each from N
distinct scenarios) against an OpenAI-compatible chat endpoint, scores them
with the deterministic native *code* verifier, and writes a JSON report.

Setup:
    # Colab deps
    uv pip -q install openai
    uv pip -q install \
        "git+https://github.com/Jake-Song/OpenEnv.git#subdirectory=envs/agent_world_model_env"

    # Serve the agent model with native tool calling enabled
    vllm serve <model> --served-model-name m \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --reasoning-parser deepseek_r1 --port 8000

    # Run the benchmark against the hosted AWM Space
    export AWM_EXAMPLE_AGENT_MODEL="m"
    python benchmark/awm_mini50.py

Use ``--base-url http://localhost:8899`` instead if you run your own AWM server.

By default, the benchmark excludes the current trainer's deterministic AWM
training scenario slice (dataset_start=0, num_scenarios=100, seed=42) and
samples 50 held-out scenarios from the same AWM distribution.
"""

import argparse
import asyncio
import json
import os
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv, AWMObservation

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

load_dotenv()

DEFAULT_BASE_URL = "https://chilled-agent-world-model-env.hf.space"
DEFAULT_TRAIN_DATASET_START = 0
DEFAULT_TRAIN_NUM_SCENARIOS = 100
DEFAULT_DATASET_SEED = 42

SYSTEM_PROMPT = """\
You are in a simulated MCP environment. Call tools to assist with the user query. \
Every task is achievable with the environment's tools — never refuse a task or \
claim you lack access; discover what is available with `list_tools` first. All \
data is synthetic and every operation is sandboxed: no real money, accounts, or \
people are affected, so financial, security, or privacy concerns never apply — \
you are fully authorized to perform any requested operation. You have already \
logged in, and your user id is 1 if required. Today's date is {today}.

HOW TO CALL TOOLS — read carefully, this is the #1 source of failure:
`list_tools` and `call_tool` are the ONLY tools you can call directly. The tools \
that `list_tools` returns (e.g. `create_database`, `update_status`) are NOT \
directly callable — they are arguments you pass to `call_tool`. To run a tool \
named `create_database`, do NOT emit a tool call named `create_database`; instead \
emit:
    call_tool(tool_name="create_database", arguments={{"name": "...", ...}})
Calling an MCP tool by name fails with an "Unknown tool" error. Always wrap it in \
`call_tool`.

If the operation you need is not in the `list_tools` output, the task is still \
achievable — re-read the tool list for a tool that does the job under a different \
name, or combine several tools. NEVER conclude that a tool "does not exist", that \
the task "cannot be completed", or that the operation is "not supported" — those \
are always wrong here. Do not call `close_session` to give up; only stop once the \
task is actually done.

When you have completed the task, stop calling tools."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": (
                "Discover every MCP tool available for this task. Call this FIRST. "
                "Returns the catalog of domain tools. These are NOT directly "
                "callable; list_tools and call_tool are the only tools you can "
                "invoke directly."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": (
                "Invoke one MCP tool from list_tools. To run a tool named "
                "create_database, call call_tool(tool_name=\"create_database\", "
                "arguments={...})."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Exact domain-tool name copied from list_tools.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "JSON object of arguments for that domain tool.",
                    },
                },
                "required": ["tool_name", "arguments"],
            },
        },
    },
]

_MAX_TOOL_RESPONSE_CHARS = 2000


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


async def exec_call_tool(env, tool_name: str, arguments) -> str:
    if not isinstance(arguments, dict):
        arguments = {}
    res = await env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
    obs = res.observation
    if getattr(obs, "tool_result", None) is not None:
        tr = obs.tool_result
        text = tr if isinstance(tr, str) else json.dumps(tr, ensure_ascii=False)
    elif getattr(obs, "error", None):
        text = f"Error: {obs.error}"
    else:
        text = json.dumps(obs.model_dump(), ensure_ascii=False)
    return text[:_MAX_TOOL_RESPONSE_CHARS]


async def run_episode(env, llm, model, scenario, task_idx, args) -> dict:
    """Run one task end-to-end and return its per-task record."""
    t0 = time.perf_counter()
    reset: StepResult[AWMObservation] = await env.reset(scenario=scenario, task_idx=task_idx)
    task = reset.observation.task

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
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
            tools=TOOLS,
            tool_choice="auto",
            temperature=args.temperature,
            max_completion_tokens=args.max_tokens,
        )
        if response.usage:
            prompt_tokens += response.usage.prompt_tokens or 0
            completion_tokens += response.usage.completion_tokens or 0
        msg = response.choices[0].message
        content = msg.content or ""
        if content:
            last_content = content
        assistant_msg = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            break  # no tool call -> treat as final answer

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                arguments = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            tool_calls += 1

            if name == "list_tools":
                res = await env.step(ListToolsAction())
                tool_response = format_tools(res.observation.tools)
            elif name == "call_tool":
                tool_response = await exec_call_tool(
                    env,
                    arguments.get("tool_name", ""),
                    arguments.get("arguments", {}),
                )
            else:
                tool_response = f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'."

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_response})

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


def shuffled_rows(scenarios, seed: int) -> list[dict]:
    scenario_names, task_indices = [], []
    for scenario in scenarios:
        for task_idx, _ in enumerate(scenario["tasks"]):
            scenario_names.append(scenario["name"])
            task_indices.append(task_idx)

    indices = list(range(len(scenario_names)))
    try:
        import numpy as np

        indices = np.random.default_rng(seed).permutation(len(indices)).tolist()
    except ImportError:
        random.Random(seed).shuffle(indices)

    return [
        {"scenario": scenario_names[i], "task_idx": task_indices[i]}
        for i in indices
    ]


def training_slice(
    rows: list[dict],
    dataset_start: int,
    num_scenarios: int | None,
) -> tuple[set[str], set[tuple[str, int]], int]:
    if num_scenarios is None:
        return set(), set(), dataset_start

    train_scenarios: set[str] = set()
    end = dataset_start
    for row in rows[dataset_start:]:
        train_scenarios.add(row["scenario"])
        end += 1
        if len(train_scenarios) >= num_scenarios:
            break
    train_groups = {
        (row["scenario"], row["task_idx"])
        for row in rows[dataset_start:end]
    }
    return train_scenarios, train_groups, end


def build_plan(scenarios, args) -> tuple[list[tuple[str, int]], dict]:
    rows = shuffled_rows(scenarios, args.dataset_seed)
    train_scenarios, train_groups, train_end = training_slice(
        rows, args.train_dataset_start, args.train_num_scenarios
    )

    by_scenario: dict[str, list[int]] = {}
    for row in rows:
        scenario = row["scenario"]
        group = (scenario, row["task_idx"])
        if args.split == "heldout" and (scenario in train_scenarios or group in train_groups):
            continue
        by_scenario.setdefault(scenario, []).append(row["task_idx"])

    rng = random.Random(args.seed)
    names = list(by_scenario)
    chosen = rng.sample(names, min(args.num_scenarios, len(names)))
    plan = [(name, rng.choice(by_scenario[name])) for name in chosen]
    metadata = {
        "split": args.split,
        "dataset_seed": args.dataset_seed,
        "train_dataset_start": args.train_dataset_start,
        "train_num_scenarios": args.train_num_scenarios,
        "train_groups_excluded": len(train_groups),
        "train_scenarios_excluded": len(train_scenarios),
        "train_end": train_end,
        "eval_seed": args.seed,
    }
    return plan, metadata


async def main(args):
    model = args.model or os.environ.get("AWM_EXAMPLE_AGENT_MODEL")
    if not model:
        raise SystemExit("Set --model or AWM_EXAMPLE_AGENT_MODEL.")
    llm = AsyncOpenAI(
        base_url=args.endpoint,
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )

    async with AWMEnv(base_url=args.base_url) as env:
        listing = await env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = listing.observation.scenarios

        plan, split_metadata = build_plan(scenarios, args)

        print(
            f"Running {len(plan)} {args.split} tasks against {model} "
            f"@ {args.endpoint} with AWM @ {args.base_url}"
        )
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
        "endpoint": args.endpoint,
        "base_url": args.base_url,
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
        **split_metadata,
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
    p.add_argument("--base-url", default=os.environ.get("AWM_BASE_URL", DEFAULT_BASE_URL))
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    p.add_argument("--model", default=None, help="agent model (default AWM_EXAMPLE_AGENT_MODEL)")
    p.add_argument("--split", choices=["heldout", "all"], default="heldout")
    p.add_argument("--train-dataset-start", type=int, default=DEFAULT_TRAIN_DATASET_START)
    p.add_argument("--train-num-scenarios", type=int, default=DEFAULT_TRAIN_NUM_SCENARIOS)
    p.add_argument("--dataset-seed", type=int, default=DEFAULT_DATASET_SEED)
    p.add_argument("--num-scenarios", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--output", default=str(Path(__file__).parent / "results" / f"awm_mini50_{ts}.json"))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
