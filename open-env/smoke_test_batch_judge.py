"""Smoke test for the batched (concurrent-dispatch) LLM judge.

Verifies the shared-loop judge runtime in
`envs/agent_world_model_env/server/verifier.py`:

  1. submit_judge works from a sync context and returns the parsed verdict.
  2. Concurrent submit_judge calls actually OVERLAP (run on one shared loop)
     instead of running one-by-one.
  3. The asyncio.Semaphore caps in-flight requests at JUDGE_MAX_INFLIGHT.
  4. _get_shared_loop returns a single reused loop.

No real LLM is contacted: _get_client is patched to return a fake async client
whose chat.completions.create sleeps and records observed concurrency.

Run:
    PYTHONPATH=/home/jake/OpenEnv:/home/jake/OpenEnv/src \
        uv run python open-env/smoke_test_batch_judge.py
"""

import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

from envs.agent_world_model_env.server import verifier

# Cap must be set BEFORE the first submit_judge call, because the semaphore is
# created lazily on first use and then cached for the process lifetime.
CAP = 4
NUM_CALLS = 20
CALL_SLEEP_S = 0.2

verifier.JUDGE_MAX_INFLIGHT = CAP


# --- fake async OpenAI client -------------------------------------------------

_concurrency_lock = threading.Lock()
_in_flight = 0
_max_in_flight = 0
_client_calls = 0


def _make_response():
    msg = types.SimpleNamespace(
        content='{"classification": "complete", "reasoning": "ok", '
        '"confidence_score": [100, 0, 0, 0], "evidence": {}}'
    )
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class _FakeCompletions:
    async def create(self, **kwargs):
        global _in_flight, _max_in_flight, _client_calls
        with _concurrency_lock:
            _in_flight += 1
            _max_in_flight = max(_max_in_flight, _in_flight)
            _client_calls += 1
        try:
            # asyncio.sleep so the event loop can interleave other requests;
            # a blocking sleep here would (correctly) still serialize.
            import asyncio

            await asyncio.sleep(CALL_SLEEP_S)
        finally:
            with _concurrency_lock:
                _in_flight -= 1
        return _make_response()


class _FakeClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_FakeCompletions())


_fake_client = _FakeClient()


def _fake_get_client(base_url, api_key):
    return _fake_client


verifier._get_client = _fake_get_client


# --- the test -----------------------------------------------------------------


def _one_call(i):
    return verifier.submit_judge(
        task=f"task-{i}",
        verifier_result={"rows": []},
        llm_base_url="http://fake.local/v1",
        llm_api_key="EMPTY",
        llm_model="fake-judge",
        trajectory=[{"action": "noop"}],
    )


def main() -> int:
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=NUM_CALLS) as pool:
        results = list(pool.map(_one_call, range(NUM_CALLS)))
    elapsed = time.monotonic() - start

    failures = []

    # 1. every call returns the parsed "complete" verdict
    bad = [r for r in results if r[0] != "complete"]
    if bad:
        failures.append(f"{len(bad)}/{NUM_CALLS} calls did not return 'complete': {bad[:3]}")

    # 2. the client was actually exercised once per call
    if _client_calls != NUM_CALLS:
        failures.append(f"expected {NUM_CALLS} client calls, saw {_client_calls}")

    # 3. requests overlapped (proves concurrent dispatch, not one-by-one)
    if _max_in_flight < 2:
        failures.append(
            f"no overlap observed (max in-flight={_max_in_flight}); calls ran serially"
        )

    # 4. semaphore capped concurrency at JUDGE_MAX_INFLIGHT
    if _max_in_flight > CAP:
        failures.append(f"semaphore breached: max in-flight={_max_in_flight} > cap={CAP}")

    # 5. wall-clock consistent with batched dispatch, not serial
    #    serial would be ~NUM_CALLS * CALL_SLEEP_S; batched ~ceil(N/CAP)*sleep.
    serial_floor = NUM_CALLS * CALL_SLEEP_S
    if elapsed >= serial_floor:
        failures.append(
            f"elapsed {elapsed:.2f}s >= serial floor {serial_floor:.2f}s; not batching"
        )

    # 6. one shared loop, reused
    if verifier._get_shared_loop() is not verifier._get_shared_loop():
        failures.append("shared loop is not reused")

    print(f"calls={NUM_CALLS} cap={CAP} max_in_flight={_max_in_flight} "
          f"client_calls={_client_calls} elapsed={elapsed:.2f}s "
          f"(serial floor {serial_floor:.2f}s)")

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: judge calls dispatched concurrently, capped at the semaphore.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
