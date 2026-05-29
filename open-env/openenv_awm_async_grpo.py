"""Async GRPO training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is an agentic environment: the model discovers MCP tools, calls them over
several turns, and a verifier scores the final outcome (complete=1.0,
incomplete=0.1, format_error=-1.0). Because the reward only exists after a
multi-turn rollout, this uses TRL's `AsyncGRPOTrainer` with an
`environment_factory`: the trainer creates one `AWMEnvironment` per inflight
slot, calls `reset(**row)` before each rollout, and exposes the env's public
methods (`list_tools`, `call_tool`, `submit`) as native tool-calling tools. The
worker drives the multi-turn loop and feeds tool results back automatically.

Reward funcs don't receive the env instance, only the completion (which includes
tool messages). So `submit` runs the AWM verifier and returns the reward as a
JSON string; `task_reward` reads it back out of the `submit` tool message.

Two-GPU cloud setup with vLLM serving + NCCL weight transfer:

    # Terminal 1 - AWM env server on CPU (or set --env-url to a hosted HF Space)
    PYTHONPATH=src:envs uv run uvicorn \
      envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2 - vLLM server on GPU 0
    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
      uv run vllm serve Qwen/Qwen3-1.7B \
        --max-model-len 10000 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}'

    # Terminal 3 - trainer on GPU 1 (sql verifier needs an LLM judge)
    export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
    CUDA_VISIBLE_DEVICES=1 uv run accelerate launch open-env/openenv_awm_async_grpo.py \
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

import huggingface_hub
import wandb
from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from agent_world_model_env import AWMEnv
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """\
You are in an MCP environment. Call tools to assist with the user query. You \
have already logged in, and your user id is 1 if required.

Use `list_tools` to discover the environment's available tools, then `call_tool` \
to invoke a specific tool by name with its arguments. Call `list_tools` first. \
When you have completed the task, call `submit` to finish and be evaluated."""


# ---------------------------------------------------------------------------
# Tool-formatting helper
# ---------------------------------------------------------------------------


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
# Environment wrapper (created once per inflight slot by AsyncGRPOTrainer)
# ---------------------------------------------------------------------------

_MAX_TOOL_RESPONSE_CHARS = 4000


class AWMEnvironment:
    """AWM env exposed to AsyncGRPOTrainer as a set of tool-calling tools."""

    def __init__(self, env_url: str):
        self.env = AWMEnv(base_url=env_url).sync()

    def reset(self, scenario: str, task_idx: int, **kwargs) -> None:
        # kwargs absorbs the other dataset-row columns (prompt, task, ...).
        # The sql verifier's LLM judge is configured via OPENENV_AWM_LLM_* env
        # vars, which the env's reset() reads automatically.
        self.env.reset(scenario=scenario, task_idx=task_idx, verifier_mode="sql")

    def list_tools(self) -> str:
        """List the MCP tools available in the current environment.

        Returns:
            A human-readable description of every available tool.
        """
        result = self.env.step(ListToolsAction())
        return format_tools(result.observation.tools)

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call one MCP tool by name.

        Args:
            tool_name: The name of the tool to call (from list_tools).
            arguments: A JSON object of arguments for the tool.

        Returns:
            The tool's text response.
        """
        if not isinstance(arguments, dict):
            arguments = {}
        result = self.env.step(CallToolAction(tool_name=tool_name, arguments=arguments))
        obs = result.observation
        if getattr(obs, "tool_result", None) is not None:
            tool_result = obs.tool_result
            text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
        elif getattr(obs, "error", None):
            text = f"Error: {obs.error}"
        else:
            text = json.dumps(obs.model_dump(), ensure_ascii=False)
        return text[:_MAX_TOOL_RESPONSE_CHARS]

    def submit(self) -> str:
        """Finish the task and run the verifier. Call this when you are done.

        Returns:
            A JSON string containing the verifier reward.
        """
        result = self.env.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "sql"}))
        reward = float(result.reward or 0.0)
        self.env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))
        return json.dumps({"reward": reward})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(env_url: str, dataset_size: int) -> Dataset:
    """List AWM scenarios/tasks and build the GRPO prompt dataset."""
    env = AWMEnv(base_url=env_url).sync()
    with env:
        result = env.step(CallToolAction(tool_name="__list_scenarios__", arguments={}))
        scenarios = result.observation.scenarios

    prompts, scenario_names, task_indices = [], [], []
    for scenario in scenarios:
        for task_idx, task in enumerate(scenario["tasks"]):
            prompts.append(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                ]
            )
            scenario_names.append(scenario["name"])
            task_indices.append(task_idx)

    return Dataset.from_dict(
        {
            "prompt": prompts[:dataset_size],
            "scenario": scenario_names[:dataset_size],
            "task_idx": task_indices[:dataset_size],
        }
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------


def task_reward(completions, **kwargs):
    """Reward = the AWM verifier reward returned by the `submit` tool."""
    rewards = []
    for completion in completions:
        reward = 0.0
        for msg in completion:
            if msg.get("role") == "tool" and msg.get("name") == "submit":
                try:
                    reward = float(json.loads(msg["content"])["reward"])
                except Exception:
                    reward = 0.0
        rewards.append(reward)
    return rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async GRPO training for AWM agent tasks.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--env-url", default="http://localhost:8899")
    parser.add_argument("--output-dir", default="Qwen3-1.7B-awm-async-grpo")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=10)
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
    parser.add_argument("--wandb-name", default="awm-async-grpo")
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
        max_tool_calling_iterations=args.max_turns,
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
        environment_factory=lambda: AWMEnvironment(args.env_url),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Upload model")


if __name__ == "__main__":
    main()
