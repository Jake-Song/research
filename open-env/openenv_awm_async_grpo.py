"""GRPO training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is an agentic environment: the model discovers MCP tools, calls them over
several turns, and a verifier scores the final outcome (complete=1.0,
incomplete=0.1, format_error=-1.0). Because the reward only exists after a
multi-turn rollout, this uses TRL's experimental `rollout_func` together with
`trl.experimental.openenv.generate_rollout_completions` - the same pattern as
TRL's BrowserGym OpenEnv example. Each turn re-templates the running
conversation as the "prompt" and keeps only the generated assistant tokens as
the "completion"; tool-response tokens live in the next turn's prompt and never
enter `completion_ids`, so no env_mask is needed.

Two-GPU cloud setup with vLLM in TRL server mode:

    # Terminal 1 - AWM env server on CPU (or set --env-url to a hosted HF Space)
    PYTHONPATH=src:envs uv run uvicorn \
      envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2 - vLLM server on GPU 0
    CUDA_VISIBLE_DEVICES=0 uv run trl vllm-serve \
      --model Qwen/Qwen3-1.7B --port 8000 --max_model_len 10000

    # Terminal 3 - trainer on GPU 1 (sql verifier needs an LLM judge)
    export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
    CUDA_VISIBLE_DEVICES=1 uv run accelerate launch open-env/openenv_awm_grpo.py \
      --env-url http://localhost:8899

The verifier runs in "sql" mode, which calls the external LLM judge configured
via the OPENENV_AWM_LLM_* env vars (the env's reset() reads them automatically).

Caveats:
- Rollouts hit the AWM env server synchronously and run multiple agent turns
  each, so generation throughput is bottlenecked by the env, not the GPU.
"""

from __future__ import annotations

import argparse
import json
import re

import huggingface_hub
import wandb
from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.openenv import generate_rollout_completions

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from agent_world_model_env import AWMEnv
from agent_world_model_env.server.prompts import DEFAULT_SYSTEM_PROMPT
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Tool-call parsing helpers (copied from awm_example_usage.py)
# ---------------------------------------------------------------------------


def parse_tool_call(content: str) -> dict | None:
    """Extract the first <tool_call> block from LLM output."""
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


# ---------------------------------------------------------------------------
# Dataset + scenario lookup
# ---------------------------------------------------------------------------

# rollout_func only receives the `prompt` column, so we recover the
# (scenario, task_idx) needed to reset the env by looking up the task text.
# Task descriptions are detailed enough to be unique across scenarios.
TASK_LOOKUP: dict[str, tuple[str, int]] = {}


def build_dataset(env_url: str, dataset_size: int) -> Dataset:
    """List AWM scenarios/tasks and build the GRPO prompt dataset."""
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    prompts = []
    for scenario in scenarios:
        for task_idx, task in enumerate(scenario["tasks"]):
            TASK_LOOKUP[task] = (scenario["name"], task_idx)
            prompts.append(
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]
            )

    prompts = prompts[:dataset_size]
    return Dataset.from_dict({"prompt": prompts})


# ---------------------------------------------------------------------------
# Multi-turn rollout
# ---------------------------------------------------------------------------

_MAX_TOOL_RESPONSE_CHARS = 4000


def execute_tool_call(env: AWMEnv, tc: dict) -> str:
    """Run a parsed list_tools / call_tool action and return a text response."""
    name = tc["name"]
    arguments = tc.get("arguments") or {}

    if name == "list_tools":
        result = env.step(ListToolsAction())
        return format_tools(result.observation.tools)

    if name == "call_tool":
        tool_name = arguments.get("tool_name", "")
        inner_args = arguments.get("arguments", "{}")
        if isinstance(inner_args, str):
            try:
                inner_args = json.loads(inner_args)
            except json.JSONDecodeError:
                inner_args = {}
        if not isinstance(inner_args, dict):
            inner_args = {}

        result = env.step(CallToolAction(tool_name=tool_name, arguments=inner_args))
        obs = result.observation
        if getattr(obs, "tool_result", None) is not None:
            tool_result = obs.tool_result
            return tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
        if getattr(obs, "error", None):
            return f"Error: {obs.error}"
        return json.dumps(obs.model_dump(), ensure_ascii=False)

    return f"Error: Unknown tool '{name}'. Use 'list_tools' or 'call_tool'."


def rollout_once(trainer, env: AWMEnv, scenario: str, task_idx: int, task: str, max_steps: int) -> dict:
    """Run one AWM episode and collect GRPO training data."""
    env.reset(scenario=scenario, task_idx=task_idx, verifier_mode="sql")
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []

    for _ in range(max_steps):
        prompt_text = trainer.processing_class.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **trainer.chat_template_kwargs,
        )
        out = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(out["prompt_ids"])
        completion_ids.extend(out["completion_ids"])
        logprobs.extend(out["logprobs"])

        content = out.get("text") or trainer.processing_class.decode(
            out["completion_ids"], skip_special_tokens=True
        )
        messages.append({"role": "assistant", "content": content})

        tc = parse_tool_call(content)
        if tc is None:
            break  # no tool call => final answer

        tool_response = execute_tool_call(env, tc)[:_MAX_TOOL_RESPONSE_CHARS]
        messages.append({"role": "user", "content": f"Tool response:\n{tool_response}"})

    result = env.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "sql"}))
    reward = float(result.reward or 0.0)
    env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "reward": reward,
    }


def make_rollout_func(env_url: str, max_steps: int):
    def rollout_func(prompts: list, trainer) -> dict:
        # GRPO's sampler already repeats each prompt num_generations times, so
        # iterate 1:1 and run one independent episode per prompt.
        env = AWMEnv(base_url=env_url).sync()
        prompt_ids, completion_ids, logprobs, rewards = [], [], [], []
        with env:
            for prompt in prompts:
                task = prompt[-1]["content"]
                scenario, task_idx = TASK_LOOKUP[task]
                episode = rollout_once(trainer, env, scenario, task_idx, task, max_steps)
                prompt_ids.append(episode["prompt_ids"])
                completion_ids.append(episode["completion_ids"])
                logprobs.append(episode["logprobs"])
                rewards.append(episode["reward"])

        return {
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "env_reward": rewards,
        }

    return rollout_func


def task_reward(completions, **kwargs):
    """Reward = the AWM verifier reward forwarded from rollout_func."""
    return kwargs["env_reward"]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for AWM agent tasks.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env-url", default="http://localhost:8899")
    parser.add_argument("--output-dir", default="Qwen3-1.7B-awm-grpo")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--wandb-project", default="openenv-awm")
    parser.add_argument("--wandb-name", default="awm-grpo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    huggingface_hub.login()
    wandb.login()
    wandb.init(project=args.wandb_project, name=args.wandb_name)

    dataset = build_dataset(args.env_url, args.dataset_size)

    grpo_config = AsyncGRPOConfig(
        # Training schedule / optimization
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        optim="adamw_torch",
        max_grad_norm=1.0,

        # GRPO configuration
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        log_completions=True,
        num_completions_to_print=2,
        chat_template_kwargs={"enable_thinking": False},
        weight_sync_steps=1,
        max_staleness=4,

        # vLLM (async => server mode on a separate GPU)
        vllm_server_base_url=f"http://{args.vllm_server_host}:{args.vllm_server_port}",

        # Precision
        bf16=True,

        # Logging / reporting
        output_dir=args.output_dir,
        report_to="wandb",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Memory
        gradient_checkpointing=True,

        # Hub
        push_to_hub=args.push_to_hub,
    )

    trainer = AsyncGRPOTrainer(
        model=args.model_id,
        reward_funcs=[task_reward],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=make_rollout_func(args.env_url, args.num_generations),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Upload model")


if __name__ == "__main__":
    main()
