"""Offline mock-up test for openenv_awm_async_grpo.py.

Exercises the new out-of-band scoring path (submit removed, AWMRolloutWorker
scores each rollout and feeds the reward to _verifier_reward) without vLLM,
GPUs, or a running AWM env server.

Run with:

    uv run python open-env/test_awm_worker_mock.py
"""

from __future__ import annotations

import asyncio
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openenv_awm_async_grpo as awm
from trl.experimental.async_grpo.async_rollout_worker import AsyncRolloutWorker


class FakeObs:
    def __init__(self, reward):
        self.reward = reward


class FakeEnvClient:
    """Stand-in for AWMEnv(...).sync(): records steps, returns fixed rewards."""

    def __init__(self, code_reward=1.0):
        self._code_reward = code_reward
        self.calls = []

    def step(self, action):
        mode = action.arguments.get("verifier_mode")
        self.calls.append((action.tool_name, mode))
        if action.tool_name == "verify" and mode == "code":
            return FakeObs(self._code_reward)
        return FakeObs(None)  # done


def make_env(code_reward=1.0):
    """Build an AWMEnvironment without connecting to a server."""
    env = awm.AWMEnvironment.__new__(awm.AWMEnvironment)
    env.env = FakeEnvClient(code_reward)
    return env


def discover_tools(env):
    """Replicate TRL AsyncRolloutWorker's tool-discovery filter."""
    names = []
    for name, member in inspect.getmembers(env, predicate=inspect.ismethod):
        if name == "reset":
            continue
        if not name.startswith("_"):
            names.append(name)
    return names


def test_submit_not_a_tool():
    names = discover_tools(make_env())
    assert "submit" not in names, names
    assert "_score_rollout" not in names, names
    assert set(names) == {"list_tools", "call_tool"}, names


def test_score_rollout_returns_code_reward_and_calls_done():
    env = make_env(code_reward=1.0)
    reward = env._score_rollout(final_answer="my answer")
    assert reward == 1.0, reward
    assert env.env.calls == [
        ("verify", "code"),
        ("done", None),
    ], env.env.calls


def test_worker_reward_plumbing():
    worker = awm.AWMRolloutWorker.__new__(awm.AWMRolloutWorker)
    worker._rollout_rewards = {}

    fake_completion = [{"role": "assistant", "content": "done"}]

    async def fake_super_generate_one(self, prompt, tool_dict):
        return (fake_completion, [1, 2], [0.0, 0.0], [1, 1], 0, 0)

    orig = AsyncRolloutWorker._generate_one
    AsyncRolloutWorker._generate_one = fake_super_generate_one
    try:
        env = make_env(code_reward=1.0)
        tool_dict = {"list_tools": env.list_tools, "call_tool": env.call_tool}
        out = asyncio.run(worker._generate_one(["prompt"], tool_dict))
    finally:
        AsyncRolloutWorker._generate_one = orig

    completion = out[0]
    assert completion is fake_completion
    assert worker._rollout_rewards[id(completion)] == 1.0, worker._rollout_rewards
    # the completion was passed through as the _score_rollout final_answer
    assert env.env.calls[0] == ("verify", "code"), env.env.calls

    rewards = worker._verifier_reward([completion])
    assert rewards == [1.0], rewards
    assert worker._rollout_rewards == {}, "reward should be popped after read"


def test_missing_reward_defaults_to_zero():
    worker = awm.AWMRolloutWorker.__new__(awm.AWMRolloutWorker)
    worker._rollout_rewards = {}
    unknown = [{"role": "assistant", "content": "x"}]
    assert worker._verifier_reward([unknown]) == [0.0]


def test_reward_func_names_stable_for_wandb():
    # __init__ runs super().__init__ (needs vLLM), so check the override intent
    # via source: it must keep the wandb key stable and swap in _verifier_reward.
    src = inspect.getsource(awm.AWMRolloutWorker.__init__)
    assert 'self.reward_func_names = ["task_reward"]' in src, src
    assert "self.reward_funcs = [self._verifier_reward]" in src, src


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
