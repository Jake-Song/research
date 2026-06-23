# Lightweight Benchmarks for Agent World Model Environments

Research snapshot: 2026-06-23

## Recommendation

Use two complementary evaluations:

1. **AWM-mini-50** for the real end-to-end result: run a fixed set of 50 tasks from distinct Agent World Model scenarios and score them with the deterministic code verifier.
2. **MCPMark Verified filesystem/easy-10** as an external tool-use control: it is small, requires no external service account, and uses automated verification.

This combination is lightweight while covering both performance inside AWM and general MCP tool-use reliability.

## Shortlist

### 1. Native AWM mini benchmark

The Agent World Model environment already provides the correct benchmark boundary: stateful MCP tools, task-specific database state, and verification logic. The full suite contains 1,000 environments and 10,000 tasks, so a fixed subset is sufficient for quick iteration.

Suggested initial protocol:

- Select 50 tasks from 50 distinct scenarios using a fixed seed.
- Use `verifier_mode="code"` for deterministic, judge-free scoring.
- Use one generation per task with a fixed temperature and maximum-turn limit.
- Record success rate, reward-type counts, average tool calls, latency, and token usage.
- Preserve the task IDs and model configuration so results remain comparable.

Source: [Agent World Model documentation](https://github.com/huggingface/OpenEnv/blob/main/envs/agent_world_model_env/README.md)

### 2. MCPMark Verified Easy

MCPMark evaluates agents in real MCP tool environments with isolated tasks and automated verifiers. MCPMark Verified became the default task set on 2026-06-12. Its easy suite contains 50 tasks across five services; the ten filesystem tasks are the smallest zero-account option.

Recommended use:

- Start with all ten `filesystem/easy` tasks at `k=1`.
- Avoid Notion, GitHub, Postgres, and Playwright until broader coverage is needed.
- Treat it as an external sanity check, not a replacement for native AWM evaluation.

Sources: [MCPMark](https://github.com/eval-sys/mcpmark), [filesystem/easy tasks](https://github.com/eval-sys/mcpmark/tree/main/tasks/filesystem/easy), [MCPMark Verified release](https://github.com/eval-sys/mcpmark/pull/264)

### 3. tau3-bench mini-run

The current tau-bench v1.0 release adds updated tasks, additional domains, knowledge retrieval, and voice evaluation. A text-only run can be limited to five tasks.

Use it only when evaluating conversational policy compliance and interaction with a simulated user. It is heavier and more expensive than AWM or MCPMark because both an agent model and a user model participate.

Source: [tau-bench](https://github.com/sierra-research/tau2-bench), [v1.0.0 release](https://github.com/sierra-research/tau2-bench/releases/tag/v1.0.0)

### 4. BFCL V4 partial evaluation

BFCL is useful as a fast preflight test for function selection, argument construction, parallel calls, and multi-turn function calling. It supports selected categories and partial evaluation.

BFCL does not exercise AWM's stateful environment or task completion verifier. Use it to catch tool-calling regressions before an AWM run, not as the primary benchmark.

Source: [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

## Relevant but not currently lightweight

### MCP-Persona

MCP-Persona was released on 2026-06-01 and is closely related to AWM: it contains 173 stateful tool-chain tasks spanning 139 tools and 18 simulated MCP servers, without requiring live credentials. However, its published runner and evaluation workflow are not yet as turnkey as MCPMark, and parts of its evaluation rely on an LLM judge.

Source: [MCP-Persona](https://github.com/wwh0411/MCP-Persona), [paper](https://arxiv.org/abs/2606.02470)

### PlanBench-XL

PlanBench-XL was released on 2026-06-21 and evaluates long-horizon planning over 327 retail tasks and 1,665 tools. It is recent and relevant but too large for a lightweight regression benchmark.

Source: [PlanBench-XL paper](https://arxiv.org/abs/2606.22388)

## Minimal evaluation stack

| Layer | Suite | Size | Purpose |
|---|---|---:|---|
| Primary | AWM-mini-50 | 50 tasks | End-to-end AWM task completion |
| External control | MCPMark filesystem/easy | 10 tasks | General MCP tool-use reliability |
| Optional preflight | BFCL V4 partial | Small selected subset | Function-call formatting and selection |
| Optional conversational test | tau3-bench | 5 tasks | User interaction and policy compliance |

