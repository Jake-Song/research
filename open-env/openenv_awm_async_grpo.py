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
import logging
from datetime import date

from datasets import Dataset
from trl.chat_template_utils import parse_response
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker

from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
from agent_world_model_env import AWMEnv
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


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
MESSAGE_TIMEOUT_S = 600.0
# reward_type strings the AWM env assigns to tool-call format violations
# (mirrors FORMAT_ERROR_TYPES in agent_world_model_env/server/awm_environment.py).
# The paper terminates the rollout with r_t = -1.0 on any such violation.
# Only truly malformed actions abort the rollout with -1.0. tool_not_found and
# invalid_args (wrong tool name, schema misses like a missing required property)
# are recoverable: the model sees the error text in the tool message and can
# retry, instead of dying on calls that were 90% correct.
_FORMAT_ERROR_REWARD_TYPES = {"invalid_action"}


class AWMEnvironment:
    """AWM env exposed to AsyncGRPOTrainer as a set of tool-calling tools."""

    def __init__(self, env_url: str):
        # Default message_timeout_s is 60s; the sql verifier's LLM judge can take
        # longer, so bump it to avoid spurious TimeoutErrors during scoring.
        self.env = AWMEnv(base_url=env_url, message_timeout_s=MESSAGE_TIMEOUT_S).sync()
        # Set by call_tool when a tool call hits a format violation; the rollout
        # worker checks it to early-terminate the rollout with reward -1.0.
        self.format_violation = False
        self.scenario = None
        self.task_idx = None

    def reset(self, scenario: str, task_idx: int, **kwargs) -> None:
        # kwargs absorbs the other dataset-row columns (prompt, task, ...).
        # The sql verifier's LLM judge is configured via OPENENV_AWM_LLM_* env
        # vars, which the env's reset() reads automatically.
        self.format_violation = False
        self.scenario = scenario
        self.task_idx = task_idx
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
        if getattr(obs, "reward_type", None) in _FORMAT_ERROR_REWARD_TYPES:
            self.format_violation = True
        if getattr(obs, "tool_result", None) is not None:
            tool_result = obs.tool_result
            text = tool_result if isinstance(tool_result, str) else json.dumps(tool_result, ensure_ascii=False)
        elif getattr(obs, "error", None):
            text = f"Error: {obs.error}"
        else:
            text = json.dumps(obs.model_dump(), ensure_ascii=False)
        return text[:_MAX_TOOL_RESPONSE_CHARS]

    def _score_rollout(self) -> tuple[float, str]:
        """Run the verifier on the finished episode. Not exposed to the model.

        Uses the "sql" verifier only — the code-augmented LLM-as-Judge that the
        AWM paper defines as the single outcome reward R_τ ∈ {1.0, 0.1, 0.0}.
        (Pure "code" verification is deliberately excluded: the paper shows it
        produces false negatives on partial/transient executions.)

        Status comes from the server's reward_type, not from the reward value:
        the env returns reward 0.0 for server-side scoring failures
        (judge_error, no_verifier, server_error), which would otherwise be
        indistinguishable from a judged outcome. Judged outcomes (complete=1.0,
        incomplete=0.1, agent_error=0.0) keep the server's reward; scoring
        failures get 0.1 — the incomplete baseline — because a server failure
        is not the model's fault and must not score below group-mates whose
        episodes were judged incomplete.

        A scoring failure must also not crash the rollout worker —
        _generate_loop re-raises any task exception — so client-side errors
        (e.g. an HTTP timeout talking to the env) are caught and reported as
        "env_error:<ExceptionType>" with reward 0.1. close_session is guarded
        too: an exception in the finally block would replace the return value
        and propagate past _generate_one's narrow `except RuntimeError`.

        Returns:
            (reward, status) where status is "complete", "incomplete",
            "agent_error", a server-side failure reward_type (e.g.
            "judge_error"), or "env_error:<ExceptionType>".
        """
        try:
            r = self.env.step(CallToolAction(tool_name="verify", arguments={"verifier_mode": "sql"}))
            status = r.observation.reward_type
            if status in ("complete", "incomplete", "agent_error"):
                return float(r.reward or 0.0), status
            return 0.1, status or "env_server_error"
        except Exception as e:
            return 0.1, f"env_server_error:{type(e).__name__}"
        finally:
            try:
                self._close_session()
            except Exception:
                logger.warning("close_session failed after scoring", exc_info=True)

    def _close_session(self) -> None:
        """End the episode without running the verifier (used by early-terminate)."""
        self.env.step(CallToolAction(tool_name="done", arguments={"keep_session": True}))


# ---------------------------------------------------------------------------
# Rollout worker — scores each episode out-of-band, reward never in context
# ---------------------------------------------------------------------------

# Set in main() to <output_dir>/rollouts.jsonl; the worker appends one JSON
# line per finished rollout. Only rank 0 runs the worker, so no write races.
TRAJECTORY_FILE = None

_REWARD_EMA_ALPHA = 0.1  # ~20-group (10-step) smoothing window


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
        self._reward_ema = None

    async def _score_group(self, group):
        # Attach a reward EMA to every sample so it shows up as a W&B curve.
        samples = await super()._score_group(group)
        if not samples:
            return samples
        group_reward = sum(s.metrics["reward"] for s in samples) / len(samples)
        self._reward_ema = (
            group_reward
            if self._reward_ema is None
            else _REWARD_EMA_ALPHA * group_reward + (1 - _REWARD_EMA_ALPHA) * self._reward_ema
        )
        for s in samples:
            s.metrics["reward_ema"] = self._reward_ema
        return samples

    async def _generate_one(self, prompt, tool_dict):
        # Reimplements AsyncRolloutWorker._generate_one's multi-turn loop so we can
        # early-terminate on a tool-call format violation (the base loop has no such
        # hook and would otherwise run every turn to completion). On the first
        # format error we keep the partial completion and force reward -1.0, matching
        # the AWM paper's step-level rule; otherwise we score normally via the judge.
        env = tool_dict["call_tool"].__self__
        completion, completion_ids, completion_logprobs, tool_mask = [], [], [], []
        tool_call_count = 0
        tool_failure_count = 0
        iteration_num = 0
        max_iterations = self.max_tool_calling_iterations
        prompt_ids = self.tokenizer.apply_chat_template(
            prompt,
            return_dict=False,
            add_generation_prompt=True,
            tools=self.tools or None,  # `or None`: Llama bug: it renders tool boilerplate for tools=[]
            chat_template=self.chat_template,
            **self.chat_template_kwargs,
        )
        while True:
            turn_ids, turn_logprobs = await self._generate_one_turn(prompt_ids)
            assistant_message = parse_response(self.tokenizer, turn_ids)
            completion.append(assistant_message)
            completion_ids.extend(turn_ids)
            completion_logprobs.extend(turn_logprobs)
            tool_mask.extend(self._turn_mask(turn_ids))
            tool_calls = assistant_message.get("tool_calls")
            if tool_calls is None or (max_iterations is not None and iteration_num >= max_iterations):
                # Normal termination: score the finished episode with the LLM judge.
                try:
                    reward, status = await asyncio.to_thread(env._score_rollout)
                except RuntimeError:
                    # The worker loop is shutting down (its default executor is closed,
                    # so to_thread can't submit) — typically at a worker stop/restart or
                    # phase boundary while this rollout is still in flight. Not a server
                    # error: the env is fine, the rollout just got caught in teardown and
                    # its reward is discarded anyway; swallow it so the worker isn't
                    # marked failed and check_health doesn't abort the whole run.
                    reward, status = 0.1, "rollout_error"
                self._rollout_rewards[id(completion)] = reward
                self._save_trajectory(env, prompt, completion, reward, status)
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count

            tool_messages, n_calls, n_failures = self._execute_tool_calls(tool_calls, tool_dict)
            tool_call_count += n_calls
            tool_failure_count += n_failures
            completion.extend(tool_messages)
            suffix_ids = self._get_tool_suffix_ids(tool_messages)
            completion_ids.extend(suffix_ids)
            completion_logprobs.extend([0.0] * len(suffix_ids))
            tool_mask.extend([0] * len(suffix_ids))
            if env.format_violation:
                # Step-level format violation -> early-terminate with r_t = -1.0.
                # Keep the partial completion so the advantage applies to the tokens
                # generated up to the violation; skip the judge entirely.
                self._rollout_rewards[id(completion)] = -1.0
                env.close_session()
                self._save_trajectory(env, prompt, completion, -1.0, "format_violation")
                return completion, completion_ids, completion_logprobs, tool_mask, tool_call_count, tool_failure_count
            prompt_ids = prompt_ids + turn_ids + suffix_ids
            iteration_num += 1

    def _save_trajectory(self, env, prompt, completion, reward, status):
        if TRAJECTORY_FILE is None:
            return
        record = {
            "scenario": env.scenario,
            "task_idx": env.task_idx,
            "reward": reward,
            "status": status,
            "prompt": prompt,
            "completion": completion,
        }
        with open(TRAJECTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _verifier_reward(self, completions, **kwargs):
        rewards = []
        for c in completions:
            if id(c) not in self._rollout_rewards:
                logger.warning("verifier reward missing for completion id=%d; defaulting to 0.0", id(c))
            rewards.append(self._rollout_rewards.pop(id(c), 0.0))
        return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def build_dataset(env_url: str, dataset_size: int, dataset_start: int = 0) -> Dataset:
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
                    {"role": "system", "content": SYSTEM_PROMPT.format(today=date.today().isoformat())},
                    {"role": "user", "content": task},
                ]
            )
            scenario_names.append(scenario["name"])
            task_indices.append(task_idx)

    # Shuffle before truncating: the scenario list is ordered, so taking the
    # first dataset_size rows would both bias the dataset to early scenarios and
    # iterate one scenario at a time (making per-step reward track scenario
    # difficulty instead of training progress).
    dataset = Dataset.from_dict(
        {
            "prompt": prompts,
            "scenario": scenario_names,
            "task_idx": task_indices,
        }
    ).shuffle(seed=42)
    # The fixed seed makes the shuffled order identical across runs, so a
    # warm-started run can continue at index dataset_start (= sum of prior
    # runs' dataset sizes) instead of replaying the same rows.
    end = min(dataset_start + dataset_size, len(dataset))
    return dataset.select(range(dataset_start, end))


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
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-4B",
        help="Base HF model, or a previous run's output dir / checkpoint-N dir / Hub"
        " repo to warm-start from (continual training). Use a fresh --output-dir.",
    )
    parser.add_argument("--env-url", default="http://localhost:8899")
    parser.add_argument("--output-dir", default="Qwen3-4B-Thinking-awm-async-grpo")
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument(
        "--dataset-start",
        type=int,
        default=0,
        help="Start index into the shuffled dataset; for continual training set"
        " this to the sum of previous runs' dataset sizes.",
    )
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-completion-length", type=int, default=1536)
    parser.add_argument("--thinking-token-budget", type=int, default=1280)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=7e-7)
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--save-total-limit", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--vllm-server-host", default="127.0.0.1")
    parser.add_argument("--vllm-server-port", type=int, default=8000)
    parser.add_argument("--vllm-server-timeout", type=float, default=1200.0)
    parser.add_argument("--push-to-hub", action="store_true", default=True)
    parser.add_argument("--no-push-to-hub", dest="push_to_hub", action="store_false")
    parser.add_argument("--wandb-project", default="openenv-awm-thinking")
    parser.add_argument("--wandb-name", default="awm-thinking-async-grpo")
    return parser.parse_args()


def main() -> None:
    import huggingface_hub
    import wandb

    from transformers import AutoTokenizer
    from trl.chat_template_utils import qwen3_chat_template

    args = parse_args()
    huggingface_hub.login()
    wandb.login()
    wandb.init(project=args.wandb_project, name=args.wandb_name)

    global TRAJECTORY_FILE
    os.makedirs(args.output_dir, exist_ok=True)
    TRAJECTORY_FILE = os.path.join(args.output_dir, "rollouts.jsonl")

    dataset = build_dataset(args.env_url, args.dataset_size, args.dataset_start)

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
    tokenizer.chat_template = qwen3_chat_template

    grpo_config = AsyncGRPOConfig(
        model_init_kwargs={"attn_implementation": "flash_attention_3"},
        # Training schedule / optimization
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        lr_scheduler_type="constant",
        optim=args.optim,
        # Pre-clip grad norms run 0-10 in practice; clipping at 1.0 was scaling
        # most updates down 5-10x. Clip only the rare outliers (~35).
        max_grad_norm=10.0,

        # GRPO configuration
        # Qwen3 thinking-mode recommended sampling temperature. Used both to
        # sample rollouts (sent in the vLLM request, overriding the server) and
        # to scale the training-loss logits, keeping the two consistent.
        # top_p/top_k/min_p/presence_penalty aren't exposed here — they're set on
        # the vLLM server in scripts/run_vllm_awm.sh.
        temperature=0.6,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        thinking_token_budget=args.thinking_token_budget,
        max_tool_calling_iterations=args.max_turns,
        # Sequence-level importance sampling (GSPO), matching the AWM paper: one
        # length-normalized ratio per rollout instead of raw per-token ratios.
        importance_sampling_level="sequence_token",
        loss_type="grpo",
        epsilon_high=0.28,  # DAPO-style high clip for more exploration
        # No KL penalty: the async trainer has no reference model, and old_log_probs
        # are vLLM sampling logprobs, not reference logprobs. GSPO with beta=0.
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
