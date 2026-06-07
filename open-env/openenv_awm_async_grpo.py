"""Async GRPO training for the Agent World Model (AWM) multi-turn MCP agent.

AWM is an agentic environment: the model discovers MCP tools, calls them over
several turns, and a verifier scores the final outcome (complete=1.0,
incomplete=0.1, format_error=-1.0). Because the reward only exists after a
multi-turn rollout, this uses TRL's `AsyncGRPOTrainer` with an
`environment_factory`: the trainer creates one `AWMEnvironment` per inflight
slot, calls `reset(**row)` before each rollout, and exposes the env's public
methods (`list_tools`, `call_tool`) as native tool-calling tools. The worker
drives the multi-turn loop and feeds tool results back automatically.

Scoring is handled out-of-band by `AWMRolloutWorker`, a subclass of
`AsyncRolloutWorker` that overrides `_generate_one` to call `_score_rollout`
on the slot's env immediately after each rollout completes (while the env DB
state is still valid). The reward is stored by completion identity and retrieved
by `_verifier_reward`. The model never sees the reward — it is not a tool.

Eight-GPU cloud setup: 1 vLLM inference GPU + 7 FSDP2 trainer GPUs, with NCCL
weight transfer. The rollout worker only runs on rank 0, so the 7 trainer ranks
share the single vLLM server. See open-env/scripts/run_vllm_awm.sh and
open-env/scripts/run_trainer_awm.sh; the trainer launch uses the FSDP2 accelerate
config at open-env/configs/fsdp2.yaml.

    # Terminal 1 - AWM env server on CPU (or set --env-url to a hosted HF Space)
    PYTHONPATH=src:envs uv run uvicorn \
      envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899

    # Terminal 2 - vLLM server on GPU 0
    bash open-env/scripts/run_vllm_awm.sh

    # Terminal 3 - 7 FSDP2 trainers on GPUs 1-7 (sql verifier needs an LLM judge)
    export OPENENV_AWM_LLM_BASE_URL=... OPENENV_AWM_LLM_API_KEY=... OPENENV_AWM_LLM_MODEL=...
    bash open-env/scripts/run_trainer_awm.sh --env-url http://localhost:8899

A single trainer GPU still works via:
    CUDA_VISIBLE_DEVICES=1 uv run accelerate launch open-env/openenv_awm_async_grpo.py \
      --env-url http://localhost:8899

The verifier runs in "sql" mode, which calls the external LLM judge configured
via the OPENENV_AWM_LLM_* env vars (the env's reset() reads them automatically).

Caveats:
- Rollouts hit the AWM env server synchronously and run multiple agent turns
  each, so generation throughput is bottlenecked by the env, not the GPU.
"""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import asyncio
import json

from datasets import Dataset
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from agent_world_model_env import AWMEnv
from dotenv import load_dotenv
load_dotenv()


SYSTEM_PROMPT = """\
You are in an MCP environment. Call tools to assist with the user query. You \
have already logged in, and your user id is 1 if required.

Use `list_tools` to discover the environment's available tools, then `call_tool` \
to invoke a specific tool by name with its arguments. Call `list_tools` first. \
When you have completed the task, stop calling tools."""


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

_MAX_TOOL_RESPONSE_CHARS = 2000


class AWMEnvironment:
    """AWM env exposed to AsyncGRPOTrainer as a set of tool-calling tools."""

    def __init__(self, env_url: str):
        # Default message_timeout_s is 60s; the sql verifier's LLM judge can take
        # longer, so bump it to avoid spurious TimeoutErrors during scoring.
        self.env = AWMEnv(base_url=env_url, message_timeout_s=300.0).sync()

    def reset(self, scenario: str, task_idx: int, **kwargs) -> None:
        # kwargs absorbs the other dataset-row columns (prompt, task, ...).
        # The sql verifier's LLM judge is configured via OPENENV_AWM_LLM_* env
        # vars, which the env's reset() reads automatically.
        self.env.reset(
            scenario=scenario, 
            task_idx=task_idx, 
            verifier_mode="sql", 
            llm_base_url=os.environ.get("OPENENV_AWM_LLM_BASE_URL"), 
            llm_api_key=os.environ.get("OPENENV_AWM_LLM_API_KEY"), 
            llm_model=os.environ.get("OPENENV_AWM_LLM_MODEL")
        )

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

    def _score_rollout(self) -> float:
        """Run the verifier on the finished episode. Not exposed to the model.

        A scoring failure (e.g. the LLM judge timing out) must not crash the
        rollout worker — _generate_loop re-raises any task exception — so on
        error we return 0.0 and always close the session.
        """
        try:
            r = self.env.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "sql"}))
            return float(r.reward or 0.0)
        except Exception:
            return 0.0
        finally:
            self.env.step(CallToolAction(tool_name="done", arguments={"keep_session": False}))


# ---------------------------------------------------------------------------
# Rollout worker — scores each episode out-of-band, reward never in context
# ---------------------------------------------------------------------------


class AWMRolloutWorker(AsyncRolloutWorker):
    """AsyncRolloutWorker subclass that scores AWM rollouts outside the model.

    After each rollout completes (while the slot's env still holds the final
    DB state), _generate_one calls env._score_rollout() and stores the reward
    keyed by id(completion). _verifier_reward retrieves it at scoring time.
    The model has no submit tool and never sees the reward value.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rollout_rewards: dict[int, float] = {}
        self.reward_funcs = [self._verifier_reward]
        self.reward_func_names = ["task_reward"]

    async def _generate_one(self, prompt, tool_dict):
        out = await super()._generate_one(prompt, tool_dict)
        completion = out[0]
        env = tool_dict["call_tool"].__self__
        
        self._rollout_rewards[id(completion)] = await asyncio.to_thread(env._score_rollout)
        return out

    def _verifier_reward(self, completions, **kwargs):
        return [self._rollout_rewards.pop(id(c), 0.0) for c in completions]


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
    """Placeholder reward func required by AsyncGRPOTrainer's constructor.

    Actual scoring is done by AWMRolloutWorker._verifier_reward, which
    replaces this in self.reward_funcs after the worker is constructed.
    """
    return [0.0] * len(completions)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async GRPO training for AWM agent tasks.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--env-url", default="http://localhost:8899")
    parser.add_argument("--output-dir", default="Qwen/Qwen3-4B-Instruct-2507-awm-async-grpo")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=7e-7)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-timeout", type=float, default=1200.0)
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--wandb-project", default="openenv-awm")
    parser.add_argument("--wandb-name", default="awm-async-grpo")
    return parser.parse_args()


def main() -> None:
    import huggingface_hub
    import wandb

    from transformers import AutoTokenizer
    from trl.chat_template_utils import qwen3_instruct_2507_chat_template

    args = parse_args()
    huggingface_hub.login()
    wandb.login()
    wandb.init(project=args.wandb_project, name=args.wandb_name)

    dataset = build_dataset(args.env_url, args.dataset_size)

    # Point the trainer at our subclass so it instantiates AWMRolloutWorker
    # instead of the base. The trainer still handles all weight-metadata and
    # tokenizer setup; we just swap the class before it calls AsyncRolloutWorker().
    from trl.experimental.async_grpo import async_grpo_trainer
    async_grpo_trainer.AsyncRolloutWorker = AWMRolloutWorker

    # Qwen3-4B-Instruct-2507 ships its own chat template, which doesn't byte-for-byte
    # match any template TRL knows, so add_response_schema() inside AsyncRolloutWorker
    # would raise. Swap in TRL's bundled qwen3-instruct-2507 template (same <tool_call>
    # format) so the schema is recognized, and pass the tokenizer to the trainer.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.chat_template = qwen3_instruct_2507_chat_template

    grpo_config = AsyncGRPOConfig(
        model_init_kwargs={"attn_implementation": "flash-attention_3"},
        # Training schedule / optimization
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        warmup_steps=args.warmup_steps,
        optim=args.optim,
        max_grad_norm=1.0,

        # GRPO configuration
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_turns,
        # Sequence-level importance sampling (GSPO), matching the AWM paper: one
        # length-normalized ratio per rollout instead of raw per-token ratios.
        importance_sampling_level="sequence",
        loss_type="grpo",
        epsilon_high=0.28,  # DAPO-style high clip for more exploration
        beta=0.001,         # KL coefficient (TRL default is 0.0, i.e. no KL)
        log_completions=True,
        num_completions_to_print=2,
        # chat_template_kwargs={"enable_thinking": False},
        weight_sync_steps=1,
        max_staleness=4,

        # vLLM (async => server mode on a separate GPU)
        vllm_server_base_url=f"http://{args.vllm_server_host}:{args.vllm_server_port}",
        # How long the trainer waits on an empty rollout queue before stopping the
        # epoch (also the vLLM-server-ready timeout). Rollouts hit the AWM env
        # synchronously over multiple turns, so the default 240s can starve.
        vllm_server_timeout=args.vllm_server_timeout,

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
        processing_class=tokenizer,
        environment_factory=lambda: AWMEnvironment(args.env_url),
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Upload model")


if __name__ == "__main__":
    main()
