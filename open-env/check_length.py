"""Measure token lengths of AWM GRPO dataset prompts (after the list_tools turn).

The model only sees the 3 native wrapper tools (list_tools, call_tool, submit) at
turn 0, but the real token cost shows up once it calls list_tools and the env
returns the formatted MCP tool descriptions for the scenario. This script
reproduces that turn: native tools as tool schemas, plus an assistant list_tools
call and the env's list_tools output appended as a tool message.
"""

from __future__ import annotations

import argparse
import inspect

from datasets import Dataset
from transformers import AutoTokenizer

from agent_world_model_env import AWMEnv
from openenv.core.env_server.mcp_types import CallToolAction

from openenv_awm_async_grpo import AWMEnvironment, SYSTEM_PROMPT


def build_dataset(env_url: str, dataset_size: int) -> Dataset:
    """List AWM scenarios/tasks and build the GRPO prompt dataset."""
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    prompts, scenario_names = [], []
    for scenario in scenarios:
        for task_idx, task in enumerate(scenario["tasks"]):
            prompts.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]
            )
            scenario_names.append(scenario["name"])

    return Dataset.from_dict(
        {"prompt": prompts[:dataset_size], "scenario": scenario_names[:dataset_size]}
    )


def prompt_token_length(
    tokenizer, messages: list[dict], tools: list, list_tools_output: str
) -> int:
    # Reproduce the post-list_tools turn: the assistant calls list_tools and the
    # env returns the formatted MCP tool descriptions as a tool message.
    messages = messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "list_tools", "arguments": {}}}
            ],
        },
        {"role": "tool", "name": "list_tools", "content": list_tools_output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools or None,  # `or None`: avoid empty-tools boilerplate
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,  # matches chat_template_kwargs in training
    )
    return len(tokenizer.encode(text, add_special_tokens=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url", default="http://localhost:8899")
    parser.add_argument("--dataset-size", type=int, default=3000)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    args = parser.parse_args()

    dataset = build_dataset(args.env_url, args.dataset_size)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # The 3 native wrapper tools (list_tools, call_tool, submit) become the tool
    # schemas, exactly as AsyncRolloutWorker exposes them.
    environment = AWMEnvironment(args.env_url)
    tools = [
        member
        for name, member in inspect.getmembers(environment, predicate=inspect.ismethod)
        if name != "reset" and not name.startswith("_")
    ]

    # list_tools output is per-scenario (a fresh sub-env per scenario), so reset
    # once per scenario and cache the formatted tool string.
    list_tools_by_scenario: dict[str, str] = {}

    lengths: list[int] = []
    longest_idx = 0
    for i, messages in enumerate(dataset["prompt"]):
        scenario = dataset["scenario"][i]
        if scenario not in list_tools_by_scenario:
            environment.reset(scenario=scenario, task_idx=0)
            list_tools_by_scenario[scenario] = environment.list_tools()
        n = prompt_token_length(tokenizer, messages, tools, list_tools_by_scenario[scenario])
        lengths.append(n)
        if n >= lengths[longest_idx]:
            longest_idx = i

    lengths.sort()
    n = len(lengths)
    p95 = lengths[int(0.95 * (n - 1))] if n else 0

    print(f"Prompts: {n}")
    print(f"Model:   {args.model_id}")
    print(f"Min:     {lengths[0]}")
    print(f"Max:     {lengths[-1]}")
    print(f"Mean:    {sum(lengths) / n:.1f}")
    print(f"Median:  {lengths[n // 2]}")
    print(f"P95:     {p95}")
    print()
    longest = dataset["prompt"][longest_idx]
    print(f"Longest prompt ({lengths[-1]} tokens) — user task excerpt:")
    task = longest[1]["content"]
    print(task[:500] + ("..." if len(task) > 500 else ""))


if __name__ == "__main__":
    main()
