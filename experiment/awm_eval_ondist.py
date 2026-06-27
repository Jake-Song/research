"""On-distribution AWM eval — faithful to the async-GRPO training rollout path.

Step 1 of the "training failure vs transfer gap" diagnosis: eval a model on the
*same AWM tasks it was trained on*, scored with the deterministic code verifier.
Run the identical harness for the base and the finetuned checkpoint; if finetuned
does not beat base here, the training produced no usable signal (the eval-side
BFCL no-improvement is then a training problem, not just a domain-transfer gap).

Faithful to training (open-env/openenv_awm_async_grpo.py):
  - same custom SYSTEM_PROMPT
  - `list_tools` / `call_tool` as NATIVE tool-calling tools (tools= passed, the
    model emits <tool_call> parsed by vLLM's tool parser into message.tool_calls)
  - Qwen3 thinking mode, temperature 1.0

Setup (Colab): serve the model with vLLM exposing the tool + reasoning parsers,
then point --endpoint at it. The AWM env runs on the hosted HF Space by default
(no local env server needed).

    vllm serve <model> --served-model-name m \
        --enable-auto-tool-choice --tool-call-parser hermes \
        --reasoning-parser deepseek_r1 --max-model-len 24576 \
        --gpu-memory-utilization 0.92 &

    python experiment/awm_eval_ondist.py --model m \
        --endpoint http://localhost:8000/v1 \
        --base-url https://chilled-agent-world-model-env.hf.space \
        --output base.json   # then again for the finetuned model
"""

import argparse
import asyncio
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI
from openenv.core.client_types import StepResult
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from agent_world_model_env import AWMEnv, AWMObservation

# Exact training system prompt (openenv_awm_async_grpo.py:93).
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

# The two native tools, descriptions copied from the trainer's tool docstrings so
# the rendered tool schema matches what the model was trained against.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_tools",
            "description": (
                "Discover every MCP tool available for this task. Call this FIRST. "
                "Returns the catalog of domain tools (e.g. create_database, "
                "update_status) that actually perform the work. These are NOT "
                "directly callable — list_tools and call_tool are the only tools "
                "you can invoke directly; run each catalog tool by passing its "
                "name to call_tool, not by emitting a tool call of that name."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_tool",
            "description": (
                "Invoke one MCP tool from list_tools. This is the ONLY way to run "
                "them. To run a tool named create_database, do NOT emit a tool "
                "call named create_database; instead call call_tool(tool_name="
                "\"create_database\", arguments={...}). Emitting a tool call named "
                "after the domain tool fails with an \"Unknown tool\" error."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Exact domain-tool name, copied verbatim from list_tools.",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "JSON object of arguments for that tool; {} if it takes none.",
                    },
                },
                "required": ["tool_name", "arguments"],
            },
        },
    },
]

_MAX_TOOL_RESPONSE_CHARS = 2000


def format_tools(tools) -> str:
    """Format Tool objects into a readable string (copied from the trainer)."""
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
    t0 = time.perf_counter()
    reset: StepResult[AWMObservation] = await env.reset(scenario=scenario, task_idx=task_idx)
    task = reset.observation.task

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
        {"role": "user", "content": task},
    ]
    tool_calls = 0
    last_text = ""

    for _ in range(args.max_turns):
        resp = await llm.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=args.temperature,
            max_completion_tokens=args.max_tokens,
        )
        msg = resp.choices[0].message
        if msg.content:
            last_text = msg.content
        # Echo the assistant turn back as a clean message (role/content/tool_calls
        # only) — drop vLLM-specific fields like reasoning_content so the chat
        # template on the next turn isn't confused.
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            break  # no tool call -> final answer

        for tc in msg.tool_calls:
            tool_calls += 1
            name = tc.function.name
            try:
                fargs = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                fargs = {}
            if name == "list_tools":
                res = await env.step(ListToolsAction())
                content = format_tools(res.observation.tools)
            elif name == "call_tool":
                content = await exec_call_tool(
                    env, fargs.get("tool_name", ""), fargs.get("arguments", {})
                )
            else:
                content = f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'."
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

    verify: StepResult[AWMObservation] = await env.step(
        CallToolAction(
            tool_name="verify",
            arguments={"verifier_mode": args.verifier, "final_answer": last_text},
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
        "latency_s": round(time.perf_counter() - t0, 2),
    }


async def main(args):
    tasks = json.loads(Path(args.tasks).read_text())
    llm = AsyncOpenAI(base_url=args.endpoint, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    sem = asyncio.Semaphore(args.concurrency)
    done = 0

    async def worker(i, t):
        nonlocal done
        sc, ti = t["scenario"], t["task_idx"]
        async with sem:
            try:
                # One env session per task — sessions are stateful, so concurrent
                # tasks must not share a connection (mirrors the trainer's
                # one-env-per-inflight-slot model).
                async with AWMEnv(base_url=args.base_url) as env:
                    rec = await run_episode(env, llm, args.model, sc, ti, args)
            except Exception as e:  # keep the sweep going
                rec = {"scenario": sc, "task_idx": ti, "reward": None,
                       "reward_type": "error", "success": False, "tool_calls": 0,
                       "latency_s": None, "error": f"{type(e).__name__}: {e}"}
        rec["train_solve_rate"] = t.get("train_solve_rate")
        done += 1
        mark = "✓" if rec["success"] else "·"
        print(f"  [{done}/{len(tasks)}] {mark} {sc}#{ti} "
              f"reward={rec['reward']} type={rec['reward_type']} tools={rec['tool_calls']}")
        return i, rec

    print(f"Running {len(tasks)} tasks against {args.model} @ {args.endpoint} "
          f"(concurrency={args.concurrency})")
    results = await asyncio.gather(*(worker(i, t) for i, t in enumerate(tasks)))
    records = [rec for _, rec in sorted(results, key=lambda x: x[0])]

    n = len(records)
    succ = sum(r["success"] for r in records)
    rtc: dict[str, int] = {}
    for r in records:
        rtc[r["reward_type"]] = rtc.get(r["reward_type"], 0) + 1
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "endpoint": args.endpoint,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "num_tasks": n,
        "success_rate": round(succ / n, 4) if n else 0.0,
        "successes": succ,
        "reward_type_counts": rtc,
    }
    Path(args.output).write_text(json.dumps({"summary": summary, "tasks": records}, indent=2))
    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nWrote {args.output}")


def parse_args():
    p = argparse.ArgumentParser(description="On-distribution AWM eval (faithful to training)")
    p.add_argument("--model", required=True, help="served model name on the vLLM endpoint")
    p.add_argument("--endpoint", default=os.environ.get("ENDPOINT_URL", "http://localhost:8000/v1"))
    p.add_argument("--base-url", default="https://chilled-agent-world-model-env.hf.space")
    p.add_argument("--tasks", default=str(Path(__file__).parent / "awm_eval_tasks.json"))
    p.add_argument("--verifier", default="code", choices=["sql", "code"],
                   help="code = deterministic only (default); sql = code-augmented LLM judge (matches training reward)")
    p.add_argument("--concurrency", type=int, default=50, help="tasks run in parallel")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--output", default="awm_eval_result.json")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
