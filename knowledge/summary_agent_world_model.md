# Agent World Model (AWM): Infinity Synthetic Environments for Agentic RL

**arXiv:** 2602.10090 (ICML 2026, accepted)
**Authors:** Zhaoyang Wang (UNC), Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, Yuxiong He (Snowflake)
**Code/Data:** https://github.com/Snowflake-Labs/agent-world-model

## TL;DR

AWM is an open-source pipeline that uses an LLM (GPT-5) to synthesize **1,000 fully-executable, code-driven, SQLite-backed agentic environments** with **35,062 tools** and **10,000 verified tasks**, exposed via MCP. Agents trained with GRPO on a 526-env subset show strong out-of-distribution generalization on τ²-bench, BFCLv3, and MCP-Universe — beating both LLM-simulated environments and the concurrent EnvScaler (191 envs).

Two ideas worth stealing:
1. **Synthesize the environment, not just the trajectories.** Tasks → DB schema → MCP toolset → verification code, each stage with execution-based self-correction.
2. **History-aware GRPO.** Train under the same sliding-window truncation the agent will see at inference, instead of training on full history.

## Why this paper exists

Scaling agentic RL is bottlenecked by environments, not tasks. Three failure modes the paper calls out:

- **Real-world APIs** are slow, rate-limited, and brittle for RL (needs thousands of rollouts/step).
- **Human-built envs** (τ²-bench, TheMCPCompany) cap at 3–5 environments.
- **LLM-simulated envs** (one LLM call per step) hallucinate state transitions and are expensive at RL scale.

Closed-source pipelines (DeepSeek-V3.2, Qwen Tongyi) exist but haven't released code. AWM open-sources both the pipeline and the resulting environment set.

## Method: the 5-stage pipeline

Each environment is a POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, T, R)$ instantiated by code + SQLite, mirroring how real software is built.

| Stage | Input | Output | Maps to POMDP |
|---|---|---|---|
| **1. Scenario** | seed of 100 domain names | 1,000 unique stateful-app descriptions (CRUD-heavy, dedup'd) | — |
| **2. Tasks** | scenario | k=10 API-solvable, post-auth user tasks per scenario | drives downstream synthesis |
| **3. Database** | scenario + tasks | SQLite schema + populated initial state | $\mathcal{S}, s_0$ |
| **4. Interface** | tasks + DB schema | Python MCP server (tool schema first, then code) | $\mathcal{A}, \mathcal{O}, T$ |
| **5. Verification** | task + DB state | code that diffs DB before/after + LLM-as-Judge over the diff | $R_\tau$ |

**Execution-based self-correction.** Every stage runs the generated code; runtime errors are fed back to the LLM, up to 5 retries. Avg 1.13 retries; 85%+ first-attempt success. They tolerate up to 10% loss per stage for cost reasons.

**Why SQLite, not key-value?** Concurrent works (AutoEnv, AutoForge, EnvScaler) use simpler stores. AWM argues relational schemas with FK constraints are what give state transitions their consistency guarantees.

**Why code-augmented LLM-as-Judge, not pure code verification?** Synthetic envs have bugs (74% contain ≥1 bug; ~4% rollout-time errors). Pure code verification produces false negatives on partial executions / transient failures. The judge sees structured state-diff signals + the agent trajectory and returns one of `{Completed, Partially Completed, Agent Error, Environment Error}`. Costs ~$1.80 per training step (1024 samples).

## RL recipe

- **Algorithm:** GRPO (DeepSeek-Math style), online, multi-turn.
- **Models:** Qwen3 thinking models at 4B / 8B / 14B.
- **Scale:** 64 batch × 16 rollouts = **1,024 isolated env instances per step**, 96 steps, lr 7e-7.
- **Reward shaping:**
  - Step-level: any tool-call format violation → early terminate, $r_t = -1.0$.
  - Outcome: 1.0 / 0.1 / 0.0 for Completed / Partial / Failed.
  - Outcome reward broadcast to all action steps in the rollout.

### History-Aware Training (the under-appreciated bit)

Standard RL frameworks (verl, OpenRLHF) optimize all actions in one forward pass over the **full** rollout history. But at inference, the agent's context manager truncates to a sliding window $h_t^{\text{trunc}}$. This is a train-inference distribution mismatch.

Fix: re-segment the trajectory into per-step sub-trajectories, each conditioned on its own truncated history $h_t^{\text{trunc}}$, and optimize the GRPO objective over those. Costs more forward passes but eliminates the mismatch. Ablation (Table 6) shows ~2–4pt gains when inference is also truncated, and matches full-history baseline when both use full history.

## Results

OOD generalization on three benchmarks (none used during training):

| Benchmark | What it tests | Base (Qwen3-8B) | AWM | Notes |
|---|---|---:|---:|---|
| BFCLv3 | function-calling | 53.83 | **65.94** | beats Simulator + EnvScaler; slight regression on hallucination (format reward always encourages tool use) |
| τ²-bench | conversational multi-turn | — | competitive w/ EnvScaler | beats Simulator |
| MCP-Universe | real MCP servers | — | **best overall** | large gains on Financial + Location |

**Concurrent baselines:**
- *Simulator* (GPT-5 as transition model): consistently underperforms AWM — programming-backed transitions > LLM-simulated.
- *EnvScaler* (191 envs from existing task sets): competitive on τ² but **regresses** on BFCLv3 (-8.93) and MCP-Universe (-1.39). Probable cause: task overlap with τ² and overfitting.

**Scaling curve (Fig 5).** 10 envs → degradation (overfit). 100 → big gains. 526 → continues to improve monotonically. At matched 191 envs, AWM > EnvScaler.

**Pipeline universality.** Swapping GPT-5 for Claude-4.5-Sonnet → 99% code success, same quality. Open-source Qwen3.5-122B-A10B → 77% success with more bugs but still usable. Diversity is pipeline-driven, not generator-driven.

## Things to remember

- The synthesis "trick" is decomposing into **scenario → task → schema → interface → verification**, so each LLM call has a narrow, schema-grounded job.
- Tool schema is generated **before** tool code — pilot showed naive direct codegen hits 3000+ LoC envs and fails.
- Bug taxonomy: 44% unhandled edge inputs, 14% DB constraint violations. Rest are scattered.
- Cross-env code dedup: AST function duplicate rate **0.0%**, endpoint-name Jaccard **0.004**. Diversity holds at the code level.
- Judge reliability: GPT-5.1 hits 95.5% pairwise agreement, Fleiss' κ = 0.891, 9.2% reward-flip rate across 5-vote ensembles.

## Connection to this repo

This is directly relevant — `open-env/agent_env.py` is already wired against `AWMEnv` (the same Agent World Model package), so this paper *is* the upstream design doc for the environment you're actually running. Concretely:

- **`open-env/agent_env.py`** uses `AWMEnv(base_url=...)` with `reset(scenario="e_commerce_33", task_idx=0)` — that's one of the 1,000 synthesized scenarios from this paper. The `verifier_mode={"code","sql"}` and the `verify`/`done` tool calls match the verification design in §3.3.3.
- **What's *not* in the OpenEnv wrapper:** the **code-augmented LLM-as-Judge** that the paper argues is the more robust reward signal. Right now your wrapper exposes `verifier_mode=code` and `sql`. If you want to reproduce paper-quality reward signals for training, you'd need to add an LLM-judge mode that consumes (trajectory, state-diff) and returns one of the 4 reward labels. Worth checking the upstream repo for whether they expose it as a third verifier mode.
- **History-Aware Training (§4.2) is the most portable idea.** If `harness/run.py` or the GRPO setup in `open-env/openenv_2048_grpo.py` / `openenv_2048_async_grpo.py` is training on full histories while inference uses a sliding window, you're paying the distribution-mismatch tax the paper documents. Concrete check: look at how the loss mask is constructed in the GRPO rollout loop and whether each action token sees its full prefix or a truncated one. The paper's recipe is to re-segment and run multiple forward passes with truncated prefixes.
- **Reward shaping detail worth porting:** the early-terminate-on-bad-format step penalty ($r_t = -1.0$). For 2048/wordle/openspiel envs, the analog is malformed action strings — penalizing + terminating saves rollout compute and the paper shows it's a meaningful regularizer.
- **For your own env synthesis (if it ever comes up):** the 5-stage decomposition + execution-based self-correction (max 5 retries) is the lesson. Don't try to one-shot a 3kLoC environment.

### Likely follow-up questions when you actually use this

- Does `AWMEnv` already support the LLM-judge verifier, or only `code`/`sql`? (Check the OpenEnv server, not just the client.)
- Of the 1,000 scenarios, which subset overlaps with the 526 the paper actually trained on? Training on the held-out 474 would give a clean ID-vs-OOD split inside AWM itself.
- The format-correctness reward penalizes refusals — if you care about hallucination resistance (BFCLv3-style), you may want to whitelist a "no valid tool" escape hatch instead of always penalizing.
