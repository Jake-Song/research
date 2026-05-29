"""Measure token lengths of AWM GRPO dataset prompts (initial turn only)."""

from __future__ import annotations

import argparse

from datasets import Dataset
from transformers import AutoTokenizer

from agent_world_model_env import AWMEnv
from agent_world_model_env.server.prompts import DEFAULT_SYSTEM_PROMPT
from openenv.core.env_server.mcp_types import CallToolAction


def build_dataset(env_url: str, dataset_size: int) -> Dataset:
    """List AWM scenarios/tasks and build the GRPO prompt dataset."""
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    prompts = []
    for scenario in scenarios:
        for task_idx, task in enumerate(scenario["tasks"]):
            prompts.append(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]
            )

    prompts = prompts[:dataset_size]
    return Dataset.from_dict({"prompt": prompts})


def prompt_token_length(tokenizer, messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
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

    lengths: list[int] = []
    longest_idx = 0
    for i, messages in enumerate(dataset["prompt"]):
        n = prompt_token_length(tokenizer, messages)
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
