# Agent Tool Harness Example

Run the harness:

```bash
uv run python -m harness.run
```

Run one case:

```bash
uv run python -m harness.run --case lookup-capital
```

This directory is a small, deterministic example of an agent tool-use harness. It loads cases from `cases.json`, runs an example agent through a bounded tool loop, records a transcript, and checks the final answer plus required tool calls.

To use this with a real agent, replace `ExampleAgent.next_action` in `run.py` with a call to the agent or model under test. Keep the harness loop and scoring code unchanged until the cases require more behavior.
