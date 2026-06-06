# Experiment Plan: Does AWM Environment Training Transfer to General Agentic Skill?

> Status: draft for discussion. Scope locked with user — measure **OOD agentic transfer**
> (τ²-bench, BFCLv3, MCP-Universe), compare **base vs AWM-trained**, **Qwen3-4B on the local
> 8-GPU rig**. Async GRPO is the training vehicle, not the object of study.

## Context

The repo has a working asynchronous-GRPO trainer for the Agent World Model
(`open-env/openenv_awm_async_grpo.py`) on an 8-GPU topology (1 vLLM inference GPU + 7
FSDP2 trainer GPUs, NCCL weight transfer). The AWM paper (`knowledge/summary_agent_world_model.md`)
*claims* that GRPO training on its synthetic, SQLite-backed MCP environments produces
**out-of-distribution generalization** — gains on τ²-bench, BFCLv3, and MCP-Universe, none
seen during training. That paper used **synchronous** GRPO at large batch (64 × 16 rollouts,
96 steps); this repo uses **asynchronous, off-policy** GRPO at much smaller scale on a single
4B model.

The question is **not** about the GRPO algorithm. It is: **does training on AWM environments
actually make the model better at agentic tasks it was never trained on?** The deliverable is a
clean, executable **base-vs-AWM-trained** transfer study, measured on held-out agentic
benchmarks. If the transfer claim holds in-house, AWM becomes a trusted training source; if it
does not, we learn that before investing further. A null result is a valid, reportable outcome.

## Research Question

Does asynchronous-GRPO training of Qwen3-4B on AWM environments improve agentic capability on
**held-out agentic benchmarks** (τ²-bench, BFCLv3, MCP-Universe) relative to the untrained base
model — and is the improvement a genuine skill transfer rather than an artifact of tool-calling
format compliance?

## Hypotheses

- **H1 (primary transfer).** AWM-trained Qwen3-4B outperforms base on the aggregate OOD agentic
suite (mean over τ²-bench, BFCLv3-executable, MCP-Universe), significant across seeds.
- **H2 (monotone learning).** OOD score rises with AWM training step — checkpoints at
0/¼/½/full show a rising curve, not a one-off jump.
- **H3 (format vs. skill confound).** The format-correctness reward (malformed tool call →
−1.0, early terminate) inflates *tool-use frequency*. Expect gains on execution/success
subsets but a **possible regression on the BFCLv3 hallucination / relevance subset** (the
paper reports exactly this). Separating "calls tools more" from "solves more tasks" is a
first-class result.
- **H4 (no general-capability collapse — guardrail).** AWM training does not catastrophically
degrade non-agentic ability; an IFEval/GSM8K spot-check stays within a small band of base.

## Experimental Setting

Reused verbatim from the existing AWM async-GRPO stack:

- **Trainer:** `open-env/openenv_awm_async_grpo.py` (TRL `AsyncGRPOTrainer` + `AWMRolloutWorker`
out-of-band scoring; reward never enters the model context).
- **Env server:** `agent_world_model_env` (`AWMEnv`) via
`uvicorn envs.agent_world_model_env.server.app:app` on CPU/port 8899.
- **Inference:** vLLM on GPU 0 — `open-env/scripts/run_vllm_awm.sh`
(`Qwen/Qwen3-4B-Instruct-2507`, `--max-model-len 45000`, NCCL weight transfer).
- **Trainers:** GPUs 1–7 under FSDP2 — `open-env/configs/fsdp2.yaml` (`Qwen3DecoderLayer` wrap,
bf16, full reshard). Single-GPU debug path: `open-env/scripts/run_trainer_awm.sh`.
- **Reward / verifier:** `verifier_mode="sql"` + `code` (`AWMEnvironment._score_rollout`); the
SQL judge configured via `OPENENV_AWM_LLM_`* env vars.
- **Model:** `Qwen/Qwen3-4B-Instruct-2507` only.

The **new** infrastructure required is the **OOD evaluation harness** — there is currently no
τ²-bench / BFCLv3 / MCP-Universe runner in the repo.

## Conditions (independent variable)


| Condition            | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| **C0 — Base**        | `Qwen/Qwen3-4B-Instruct-2507`, no training. The control.           |
| **C1 — AWM-trained** | C0 after async-GRPO on AWM (full schedule), best/final checkpoint. |
| **C0.25 / C0.5**     | Intermediate AWM checkpoints (¼, ½ steps) — for H2 only.           |


- **Seeds:** train C1 with **≥3 seeds**; all reported deltas are over the seed distribution.
- **Held-out integrity:** before training, audit string/schema overlap between AWM scenario +
tool names and the three eval suites; document it.
- **Inference parity:** C0 and C1 evaluated through the **identical** harness, prompt,
tool-call format, decoding params, and max-turn budget. Eval-time advantage must come from
weights, not prompting.

## Training Design

Start from the existing trainer config; align cheap knobs to the paper, keep repo defaults
otherwise:

- `model_id = Qwen/Qwen3-4B-Instruct-2507`; `num_generations = 8`;
`max_completion_length = 1024`; `max_tool_calling_iterations = None`.
- `learning_rate = 1e-6` (repo default; paper used 7e-7 — note divergence, don't re-tune unless
H2 shows instability).
- `gradient_accumulation_steps = 16`, `per_device_train_batch_size = 1`.
- `weight_sync_steps = 1`, `max_staleness = 4` — async defaults **held fixed**; this study does
not vary the algorithm.
- `bf16 = True`, `gradient_checkpointing = True`; `dataset_size = 1000` via `build_dataset()`.
- Reward: `sql` + `code` verifier sum, scored out-of-band in `AWMRolloutWorker._generate_one`.
- **Length:** ~96 steps (paper order of magnitude) or train to reward plateau on a held-out AWM
dev split, whichever first. Checkpoint at 0/¼/½/full (≥4 evaluable).
- **Logging:** wandb (`openenv-awm`); track mean reward, reward-by-label fractions
(Completed/Partial/Failed/format-error), mean turns per rollout — feeds the H3 analysis.

No new training code is required; the trainer already produces checkpoints. The work is: run 3
seeds, build the eval harness, analyze.

## Evaluation Suite (the main new infrastructure)

Three external held-out agentic benchmarks, each run on **C0 and every C1 checkpoint/seed**
through one shared adapter so decoding and tool-call parsing are identical:

1. **BFCLv3** — Berkeley Function-Calling Leaderboard v3 (Gorilla). Report executable /
  multi-turn categories **and**, separately, the **relevance/hallucination** subset (H3
   signal). Cheapest — **do first** as pipeline shakedown.
2. **τ²-bench** — `sierra-research/tau2-bench`. Conversational multi-turn tool use with user
  simulator + DB verifier. Report pass^1 (pass^k if budget allows).
3. **MCP-Universe** — real MCP servers across domains (Financial, Location, …). Per-domain +
  overall success. Heaviest (needs live MCP servers); schedule last.

Each adapter wraps the model behind the harness's expected tool-call interface, points it at a
vLLM serve of the checkpoint, records full transcripts. **Sanity gate:** before trusting any
delta, confirm the harness reproduces the *published base-model* number for Qwen3-4B within a
reasonable margin.

### Metrics & statistics

- **Primary:** per-benchmark success/pass rate + **aggregate OOD mean** (C1 − C0 Δ).
- **Secondary (H3):** BFCLv3 relevance/hallucination accuracy; tool-call frequency; mean turns;
format-error rate. Decompose "more tool calls" vs. "more tasks solved."
- **Guardrail (H4):** IFEval + GSM8K (small fixed sample), C0 vs C1.
- **Uncertainty:** bootstrap CIs over **task instances** within each run, mean ± spread over 3
seeds. Treat seed spread as the honest error bar; avoid over-claiming p-values at n=3.
- **Learning curve (H2):** OOD aggregate vs. training step, one line per seed.

## Threats to Validity

- **Format-reward confound (H3).** Always-reward-tool-use can lift agentic benches while hurting
"should-not-call" cases. Mitigation: report hallucination subset separately; optional ablation
whitelisting a "no valid tool" escape.
- **Contamination.** Synthetic CRUD apps could echo benchmark domains. Mitigation: overlap audit
before training.
- **Harness fidelity.** Tool-call grammar mismatch can mimic/mask transfer. Mitigation: the
published-base-number sanity gate; C0/C1 parity through the *same* adapter.
- **Async/off-policy, small scale.** With `max_staleness=4` and a 4B model, in-house results may
legitimately differ from the paper. A null/smaller effect is valid and reportable.
- **τ²-bench simulator seed.** Fix and report it; average over 2–3 if budget allows.

## Optional Ablations (only if the primary result warrants)

- **Env-diversity lever:** retrain on 10 / 100 / 500 AWM scenarios (paper Fig 5) — strongest
follow-up if H1 is positive.
- **No-multi-turn control:** single-turn prompts + final reward, to test if interaction is
necessary.
- **Reward-source swap:** `code`-only vs `sql`+`code`, to see if the LLM-judge signal matters
for transfer.

Out of scope for the first pass; listed so a positive result can route here.

## Compute Budget & Sequencing

- **Runs:** 3 seeds × ~96 steps on the 1+7 GPU rig; each ≈ one node-day, env-server-bottlenecked
(rollouts hit AWM synchronously over multiple turns — GPU is not the bottleneck).
- **Eval:** C0 + 3×(≥3 checkpoints) × 3 benchmarks via vLLM serves.
- **Order:** (1) BFCLv3 adapter + reproduce base number → (2) train seed-1, eval end-to-end
(vertical slice) → (3) add τ²-bench, then MCP-Universe → (4) seeds 2–3 → (5) analysis.

## Critical Files

- Reuse as-is: `open-env/openenv_awm_async_grpo.py`, `open-env/configs/fsdp2.yaml`,
`open-env/scripts/run_vllm_awm.sh`, `open-env/scripts/run_trainer_awm.sh`.
- New (eval): a small `eval/` harness — one adapter per benchmark (`eval/bfcl.py`,
`eval/tau2.py`, `eval/mcp_universe.py`) + shared vLLM client + results aggregator. No trainer
changes for the core study.
- Reference: `knowledge/summary_agent_world_model.md`, `open-env/recovery_experiment_plan.md`.

## Verification (end-to-end)

1. **Stack up.** Launch AWM env server (8899), `run_vllm_awm.sh` (GPU 0), confirm
  `OPENENV_AWM_LLM_`*; 1-step trainer smoke test on a single GPU; confirm non-trivial reward in
   wandb.
2. **Harness sanity gate.** Run base through BFCLv3 adapter; confirm it reproduces a known
  Qwen3-4B reference within margin. Do not proceed until this passes.
3. **Vertical slice.** Train seed-1 full; eval C0 vs C1 on BFCLv3; confirm a measurable,
  correctly-signed delta and sane transcripts.
4. **Full matrix.** Add τ²-bench + MCP-Universe; all checkpoints × 3 seeds; aggregate OOD Δ with
  bootstrap CIs + H3 hallucination decomposition.
5. **Report.** Learning curve (H2), per-benchmark + aggregate table with CIs (H1),
  format-vs-skill decomposition (H3), IFEval/GSM8K guardrail (H4), contamination audit.

## Open Questions for Discussion

- **Training length / stopping:** fixed ~96 steps, or train-to-plateau on an AWM dev split?
- **Eval harness build vs. reuse:** stand up the official upstream harnesses (heavier setup) vs.
a thin in-repo adapter per benchmark — which fidelity/effort tradeoff do we want?
- **τ²-bench / MCP-Universe cost:** both can be API/compute heavy; cap task counts or run full
splits?
- **Seeds vs. depth:** 3 seeds × 1 model, or fewer seeds and add the env-diversity ablation now?

