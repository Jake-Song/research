# /// script
# dependencies = [
#     "git+https://huggingface.co/spaces/Jakemu/openspiel_env",
#     "transformers==5.2.0",
#     "trl[vllm]==1.4.0",
#     "vllm>=0.17.1",
#     "wandb",
#     "openspiel-env @ git+https://huggingface.co/spaces/Jakemu/openspiel_env",
# ]
# ///

"""Async GRPO training for OpenSpiel 2048 strategy generation.

Two-GPU cloud setup: vLLM server on GPU 0, AsyncGRPOTrainer on GPU 1.
NCCL is used to push updated trainer weights into the live vLLM server.

    # Terminal 1 - vLLM server on GPU 0
    CUDA_VISIBLE_DEVICES=0 VLLM_SERVER_DEV_MODE=1 \
      uv run vllm serve Qwen/Qwen3.5-2B \
        --max-model-len 3072 \
        --logprobs-mode processed_logprobs \
        --weight-transfer-config '{"backend":"nccl"}'

    # Terminal 2 - trainer on GPU 1
    CUDA_VISIBLE_DEVICES=1 \
      HF_TOKEN=... WANDB_API_KEY=... \
      uv run accelerate launch open-env/openenv_2048_async_grpo.py

The VLLM_SERVER_DEV_MODE flag, processed_logprobs mode, and NCCL
weight-transfer config are all required by AsyncGRPOTrainer - without them
the server can't accept weight updates and async generation will drift away
from the training policy.

Caveats:
- strategy_succeeds() hits an external HF Space synchronously; under async
  generation it becomes the rollout-throughput bottleneck.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
from datasets import Dataset

from openspiel_env import OpenSpielAction, OpenSpielEnv, OpenSpielObservation
from trl.experimental.async_grpo import AsyncGRPOTrainer, AsyncGRPOConfig


# ---------------------------------------------------------------------------
# 2048 environment wrapper
# ---------------------------------------------------------------------------


class Env2048:
    def __init__(self, base_url: str):
        self.client = OpenSpielEnv(base_url=base_url).sync()

    def reset(self, **kwargs) -> None | str:
        result = self.client.reset()
        self.legal_actions = result.observation.legal_actions
        self.reward = 0.0
        self.done = False
        return result.observation

    def move(self, action_id: int) -> str:
        result = self.client.step(OpenSpielAction(action_id=action_id, game_name="2048"))
        self.result = result
        self.legal_actions = result.observation.legal_actions
        self.reward = result.reward
        self.done = result.done
        return result.observation


def convert_to_board(current_state):
    n = len(current_state)
    size = int(np.sqrt(n))
    board = np.array_split(np.array(current_state, dtype=int), size)
    board = [x.tolist() for x in board]
    return board, size


# ---------------------------------------------------------------------------
# Strategy execution with timeout
# ---------------------------------------------------------------------------

_STRATEGY_TIMEOUT_S = 5.0


def execute_strategy(env: Env2048, strategy: Callable, current_state: OpenSpielObservation):
    """Run strategy on the env until done or timeout. Thread-safe (no SIGALRM)."""
    assert callable(strategy)

    deadline = time.monotonic() + _STRATEGY_TIMEOUT_S
    steps = 0
    total_reward = 0

    while not current_state.done:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {_STRATEGY_TIMEOUT_S}s.")

        board, size = convert_to_board(current_state.info_state)
        action = strategy(board)
        try:
            action = int(action)
        except Exception:
            return steps, False, current_state.info_state

        steps += 1
        if type(action) is not int or action not in current_state.legal_actions:
            return steps, max(itertools.chain.from_iterable(board)) == 2048, current_state.info_state

        current_state = env.move(action_id=action)

        if env.reward is not None:
            total_reward += env.reward

    return steps, max(itertools.chain.from_iterable(board)) == 2048, current_state.info_state


# ---------------------------------------------------------------------------
# Generated-code safety helpers
# ---------------------------------------------------------------------------


def check_python_modules(code_string):
    try:
        tree = ast.parse(code_string)
    except SyntaxError as e:
        return False, {"error": f"Syntax Error: {e}"}

    stdlib_imports = set()
    non_stdlib_imports = set()
    relative_imports = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if base_module in sys.stdlib_module_names:
                    stdlib_imports.add(base_module)
                else:
                    non_stdlib_imports.add(base_module)
        elif isinstance(node, ast.ImportFrom):
            if node.level is not None and node.level > 0:
                relative_imports += 1
            if node.module:
                base_module = node.module.split(".")[0]
                if base_module in sys.stdlib_module_names:
                    stdlib_imports.add(base_module)
                else:
                    non_stdlib_imports.add(base_module)

    info = {
        "stdlib": list(stdlib_imports),
        "non_stdlib": list(non_stdlib_imports),
        "relative_imports": relative_imports,
    }
    ok = len(non_stdlib_imports) == 0 and relative_imports == 0
    return ok, info


def create_locked_down_function(code_string):
    restricted_globals = {"__builtins__": __builtins__}
    local_namespace = {}
    exec(code_string, restricted_globals, local_namespace)

    for name, obj in local_namespace.items():
        if callable(obj):
            return obj
    raise ValueError("No function defined in the provided code.")


def extract_function(text):
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board):"):
            return fx
    return None


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


PROMPT = """
Create a new short 2048 strategy using only native Python code.
You are given a list of list of numbers for the current board state.
Output one action for "0", "1", "2", "3" on what is the optimal next step.
Output your new short function in backticks using the format below:
```python
def strategy(board):
    return "0" # Example
```
All helper functions should be inside def strategy. Only output the short function `strategy`.
""".strip()


def function_works(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is None:
            scores.append(-2.0)
            continue
        ok, info = check_python_modules(function)
        if "error" in info:
            scores.append(-2.0)
        else:
            try:
                create_locked_down_function(function)
                scores.append(1.0)
            except Exception:
                scores.append(-0.5)
    return scores


def no_cheating(completions, **kwargs):
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is not None:
            ok, info = check_python_modules(function)
            scores.append(1.0 if ok else -20.0)
        else:
            scores.append(-1.0)
    return scores


_STRATEGY_PARALLELISM = 8


def make_strategy_succeeds(env_url: str):
    def score_one(completion):
        response = completion[0]["content"]
        function = extract_function(response)

        if function is None:
            return 0
        ok, info = check_python_modules(function)
        if "error" in info:
            return 0
        try:
            new_strategy = create_locked_down_function(function)
        except Exception:
            return 0

        env2048 = None
        try:
            env2048 = Env2048(base_url=env_url).sync()
            current_state = env2048.reset()
            steps, if_done, info_state = execute_strategy(env2048, new_strategy, current_state)
            return 20.0 if if_done else 2.0
        except TimeoutError as e:
            print(f"Exception = {str(e)}")
            return -1.0
        except Exception as e:
            print(f"Exception = {str(e)}")
            return -3.0
        finally:
            if env2048 is not None:
                env2048.client.close()

    def strategy_succeeds(completions, **kwargs):
        with ThreadPoolExecutor(max_workers=_STRATEGY_PARALLELISM) as pool:
            return list(pool.map(score_one, completions))

    return strategy_succeeds


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async GRPO training for 2048 strategies.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--env-url", default="https://jakemu-openspiel-env.hf.space")
    parser.add_argument("--output-dir", default="Qwen3.5-2B-2048-async-grpo")
    parser.add_argument("--dataset-size", type=int, default=3000)
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
    parser.add_argument("--wandb-project", default="openenv-2048")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    dataset = Dataset.from_dict(
        {"prompt": [[{"role": "user", "content": PROMPT}] for _ in range(args.dataset_size)]}
    )

    grpo_config = AsyncGRPOConfig(
        # Training schedule / optimization
        #use_liger_kernel=True,
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
        weight_sync_steps=10,
        max_staleness=3,

        # vLLM (async => server mode on a separate GPU)
        vllm_server_base_url=f"http://{args.vllm_server_host}:{args.vllm_server_port}",

        # Precision (bf16; quantization is incompatible with NCCL weight transfer)
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
        reward_funcs=[
            function_works,
            no_cheating,
            make_strategy_succeeds(args.env_url),
        ],
        train_dataset=dataset,
        args=grpo_config,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="Upload model")


if __name__ == "__main__":
    main()
