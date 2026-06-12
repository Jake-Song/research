---
name: inspect-rollouts
description: Use this skill when asked to inspect, summarize, or debug a rollouts.jsonl trajectory file from an async-GRPO training run (e.g. "look at the rollouts", "why are rewards low", "show me a failing trajectory").
---

Inspect a `rollouts.jsonl` file written by `open-env/openenv_awm_async_grpo.py` (`_save_trajectory` appends one JSON line per finished rollout, in completion order).

### Part 1: Locate the file

If the user didn't give a path, look for (in order): a path mentioned in the conversation, `<output_dir>/rollouts.jsonl` from the run being discussed, or `rollouts.jsonl` in the repo root (typically scp'd back from the GPU node). Files are tens of MB — never load with the Read tool; stream with python via Bash.

### Part 2: Know the schema

One JSON object per line:

- `scenario` (str), `task_idx` (int) — which AWM task.
- `reward` (float) — verifier reward: 1.0 = task done, 0.1 = incomplete/partial, 0.0 = verifier judged failed. −1.0 (format violation) is reserved for `invalid_action` only and may not appear at all in a run.
- `status` (str) — `complete` (reward 1.0), `incomplete` (covers both reward 0.1 and 0.0), `server_error` (scoring failed; reward forced to 0.1), `discarded` (caught in worker teardown; reward unused).
- `prompt` — list of `{role, content}` (system + user task).
- `completion` — multi-turn message list: assistant messages (may carry `tool_calls`) interleaved with `tool` messages (`name` + `content` = tool result).

Line order ≈ training order, so reward over line index shows training progress (groups of `--num-generations`, default 8, share a prompt).

### Part 3: Always start with an overview

```bash
python3 - <<'EOF'
import json, collections
path = "rollouts.jsonl"
n, rew, status, scen = 0, [], collections.Counter(), collections.defaultdict(list)
for line in open(path):
    r = json.loads(line); n += 1
    rew.append(r["reward"]); status[r["status"]] += 1
    scen[r["scenario"]].append(r["reward"])
print(f"{n} rollouts, mean reward {sum(rew)/n:.3f}, statuses {dict(status)}")
k = max(1, n // 10)
print("reward by tenth:", [round(sum(rew[i:i+k])/len(rew[i:i+k]), 2) for i in range(0, n, k)])
for s, rs in sorted(scen.items(), key=lambda kv: sum(kv[1])/len(kv[1])):
    print(f"{s:24s} n={len(rs):4d} mean={sum(rs)/len(rs):.3f}")
EOF
```

Also report the exact reward-value histogram (`collections.Counter` over rewards) — the 0.0 vs 0.1 split distinguishes "failed" from "partial" — and the **zero-variance group count**: group rewards by `(scenario, task_idx)` and count groups where all rewards are identical. Those groups have zero GRPO advantage and contribute no gradient; if a large share (e.g. >50%) is zero-variance, say so — it explains a flat reward curve better than any per-trajectory pathology, and points to difficulty filtering as the fix.

Then drill into whatever the user actually asked about.

### Part 4: Read individual trajectories

To answer "why" questions (low reward, weird behavior), print a few matching trajectories. Filter on scenario/status/reward, and truncate tool outputs so the transcript stays readable:

```bash
python3 - <<'EOF'
import json, itertools
def show(r):
    print(f"=== {r['scenario']}#{r['task_idx']} reward={r['reward']} status={r['status']}")
    print("TASK:", r["prompt"][-1]["content"][:300])
    for m in r["completion"]:
        if m["role"] == "assistant":
            for tc in m.get("tool_calls") or []:
                f = tc["function"]; print(f"  -> {f['name']}({json.dumps(f['arguments'])[:200]})")
            if m["content"]: print("  ASSISTANT:", m["content"][:300])
        else:
            print(f"  <- {m.get('name')}:", str(m["content"])[:200])
rows = (json.loads(l) for l in open("rollouts.jsonl"))
match = (r for r in rows if r["status"] == "incomplete")  # adjust filter
for r in itertools.islice(match, 3): show(r)
EOF
```

Useful variants: compare a `complete` vs an `incomplete` rollout of the same `(scenario, task_idx)`; count tool calls per rollout (`sum(len(m.get("tool_calls") or []) for m in r["completion"] if m["role"]=="assistant")`); grep tool results for error strings.

### Part 4b: Triage failures into known categories

Classify each failing rollout by its tool-call names — this splits the failure mass into actionable buckets:

- **`no_tool_calls`** — zero tool calls: the model refused from chat habit ("I don't have access to..."), common on finance/payment tasks. Fix lives in the system prompt.
- **`direct_mcp_names`** — any call whose name is not `list_tools`/`call_tool`: the model called an MCP tool by name instead of through the `call_tool` wrapper. Signature: tool results like `{'error': "'list_collections'"}` (a bare `KeyError` repr from `tool_dict[name]`). Count these wasted turns; models often conclude "the system is broken" and give up.
- **`proper_wrapper`** — mechanics correct, content wrong. Look for grounding errors: the model inventing absolute dates for relative ones ("next 14 days" → queries from `2025-01-01`), then trusting an empty result. Check whether the system prompt includes today's date.

Also check the truncation signature: rollouts whose last assistant message has no text (empty content, possibly dangling `tool_calls`) were cut off by `max_tokens` — with a thinking model this is the runaway-`<think>` / empty-response failure that `--thinking-token-budget` addresses. If nearly every rollout ends with proper text, truncation is not the problem.

### Part 5: Report

Summarize findings in prose: overall reward + trend, which scenarios drag the mean down, and what the failing trajectories actually do (e.g. "model claims it has no tools instead of calling list_tools"). Quote short transcript excerpts as evidence.
