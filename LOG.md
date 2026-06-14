# Analysis of `rollouts.jsonl` (2026-06-11)

1,446 rollouts, 163 scenarios, 180 GRPO groups. Branch: `relax-format-violation`.

**Headline finding: a large share of failures trace to one mechanical bug — the model calls env tools directly by name instead of through the `call_tool` wrapper, and the env replies with a useless raw `KeyError` string.**

## Reward overview

- Pass rate **38.8%** (561 of 1,446 at reward 1.0). The rest: 610 at 0.1, 275 at 0.0 — both marked `incomplete`; the 0.0-vs-0.1 split comes from the AWM env verifier itself, not the trainer.
- No upward trend across the file: per-decile mean reward bounces between 0.32 and 0.64 with no slope, so the run isn't visibly learning yet.

## The `call_tool` bypass bug

The system prompt tells the model to use `list_tools` then `call_tool`, but in **521 rollouts (38% of tool-using rollouts)** the model emits the discovered tool name directly (e.g. `list_forms(project_id=1)`). There were 1,684 such direct calls and **99.7% of them fail**, returning `{'error': "'list_forms'"}` — a stringified Python `KeyError` that gives the model no hint it should use `call_tool` instead. Models that did this pass at 33.6% vs 43.8% for wrapper-only rollouts. A typical trajectory: direct call fails → retries the same call → concludes "the environment has a limitation" → gives up or hallucinates success.

Two cheap fixes, either of which should lift reward: have the dispatcher return `Unknown tool 'X' — invoke it via call_tool(name='X', args={...})`, or simply register the discovered tools as directly callable.

## `call_tool` error breakdown (500 errors out of 2,608 calls, 19.2%)

- **354 HTTP 500 Internal Server Error** — env-side, not model fault (consistent with the recent 0.1-for-server-error commits, but note these surface inside tool responses mid-rollout, so the rollout still proceeds and typically ends 0.0/0.1).
- **126 argument validation errors** (e.g. `None is not of type 'integer'`, `int_parsing`) — genuine model mistakes, recoverable.
- 20 not-found/404.

## Failure modes among the 885 failed rollouts

- **478 gave up citing errors/inability** — heavily driven by the KeyError bug and the 500s above; 288 failed rollouts contain 3+ error tool responses (flailing on repeats).
- **244 claimed success but the verifier disagreed** — e.g. "I have successfully created a new Calendar view..." at reward 0.1; either genuine near-misses or false completion claims.
- **52 never called a single tool** — hallucinated "I don't have access to Zelle/banking tools" and refused (all of `banking_4` failed this way, 16/16 at 0.0).
- 111 other.

## GRPO signal efficiency

- **55% of groups (99/180) have zero reward variance** — 41 all-pass, 58 all-fail — contributing zero gradient. Only 81 groups carry signal.
- Of the mixed groups, 28 are (0.0, 0.1)-only mixtures: the gradient there only pushes between two failure modes, which is weak signal at best. Truly informative pass/fail contrast exists in ~53 groups (~29%).

## Bottom line

Before tuning reward shaping further, fixing the `call_tool` KeyError response is the single highest-leverage change — it's implicated in roughly a third of all rollouts and directly suppresses pass rate, and the 354 mid-rollout 500s are worth an env-side look too.

---

# Status analysis of `rollouts.jsonl` (2026-06-11, 3,263-rollout snapshot)

Newer, larger snapshot (3,263 rollouts). Branch: `relax-format-violation`.

## Status distribution

| Status | Count | % |
|---|---|---|
| incomplete | 1,652 | 50.6% |
| complete | 1,265 | 38.8% |
| format_violation | 278 | 8.5% |
| server_error | 68 | 2.1% |

## format_violation is scenario-concentrated

Top offenders: `hr_payroll_management_1` 24/44 (54.5%), `workforce_management_1` 36/80 (45.0%), `payments_billing_1` 20/80 (25.0%), `booking_and_scheduling_1` 19/80 (23.8%), `tournament_management_1` 17/80 (21.2%). The top 5 scenarios hold 116 of 278 violations (~42%); 8 scenarios have zero. `hr_payroll_management_1` is also short on rollouts (44 vs the usual 80), as are `survey_and_forms_1` (68) and `b2b_marketplace_2` (78) — truncation may correlate with whatever triggers the violations.

## server_error status undercounts real server errors

- The `server_error` *status* hits only 3 scenarios, all booking: `booking_and_appointments_1` 33/113 (29.2%), `booking_marketplace_1` 28/160 (17.5%), `booking_1` 7/80 (8.8%). The first two have inflated totals (113 and 160 vs the usual 80), i.e. they were resampled/retried.
- 7 of the 68 server_error rollouts still carry reward 0.0 despite the 0.1-floor commit — either pre-fix data or a bypassing path.
- Meanwhile, HTTP 500s appear *inside* rollouts classified as everything else: 220/1,652 incomplete (13.3%) contain a `Status code: 500` tool response, and **112 of those ended on the 500** — almost certainly killed by the server but labeled `incomplete` and denied the 0.1 floor. 125/1,265 complete (9.9%) and 68/278 format_violation (24.5%) also contain 500s.
- format_violation containing 500s at 2x the base rate suggests some "violations" are the model reacting badly after the env broke (workforce_management_1 / tournament_management_1 appear in both top lists).

## Bottom line

The status classifier only flags a server error when the episode-level call fails; mid-rollout 500s — including ones that terminate the trajectory — fall through to `incomplete`/`format_violation` and get punished as model failures. Classifying "last tool response was a 500" as server_error would roughly triple the bucket (68 → ~180) and remove that much noise from the reward signal.

---

# Next experiment: small-pool learning-signal probe (2026-06-11)

Hypothesis to test: shrinking the task pool makes any learning signal directly measurable — the RL equivalent of "overfit a small batch". If the policy can't improve on a handful of tasks it revisits many times, there is no usable learning signal at all, isolated from the measurement noise above.

## Why it works mechanically

At 2 groups/step, a 24-step run processes ~48 groups. The current ~1000-row dataset visits each task at most 1–2 times, which is why per-task curves were impossible. With a pool of ~15–20 tasks, every task gets revisited 2–3 times *within one run* — per-task reward becomes a real curve, and the paired visit-over-visit comparison (which already hinted at +0.11) gets statistical teeth.

## Setup requirements

1. **Loop the dataset.** With `num_train_epochs=1`, a 20-task dataset ends the run after ~10 steps. Raise epochs (or confirm the worker's repeat-iterator governs run length) so step count, not epoch count, decides when it stops.
2. **Pick tasks where signal can exist.** A group of 8 only produces gradient when outcomes differ (reward_std > 0). Known-learnable from the data: `booking_and_appointments_1` (0.1 → 1.0 on second visit), `marketplace_1` and `hr_payroll_management_1` task 2 (mixed outcomes within groups). Including 2–3 of the former format-trap tasks (payroll tasks 5/9) directly tests the `invalid_args` relaxation: deterministic -1.0 before, should retry past schema errors now.
3. **Select scenarios explicitly, not via `dataset_size`.** Truncating the shuffled dataset gives a random 20 tasks; a controlled probe needs a fixed hand-picked set — add a small scenario-name filter in `build_dataset`.

## Expected readout

- Per-task curves trending up on the learnable set → pipeline confirmed, scale back to the full dataset.
- Flat per-task curves despite repeated visits, healthy reward_std, and no format traps → genuine learning-signal problem; suspects become LR=7e-7 too small, advantage quality, or credit assignment over multi-turn tool masks.

## 2026-06-14 — rollouts.jsonl (repo root, 422 rollouts)

**Overview.** 422 rollouts, mean reward **0.354**. Reward histogram: `1.0`×124 (complete), `0.1`×255, `0.0`×43 (agent_error). No `-1.0` format violations. Status mix: complete 124, incomplete 123, agent_error 43, plus a large **non-model / scoring-failure band of 132 (31%)**: rollout_error 42, episode_already_done 35, server_error 24, no_verifier 16, judge_error 15 — all forced to 0.1.

**The end-of-run reward "collapse" is an infra artifact, not regression.** Reward by tenth: `0.40 0.43 0.43 0.32 0.41 0.26 0.46 0.42 0.32 0.12 0.10`. The last two tenths crater because the final 44 rollouts are **42× `rollout_error`** (all reward 0.1) — the env/trainer stack died at the tail. Drop that band and the curve is flat-ish around 0.40, no learning trend.

**Zero-variance groups: 11/54 (20%).** Not the bottleneck — most groups still produce GRPO advantage, so difficulty filtering isn't the priority here.

**Root cause of the model failures: the `call_tool` wrapper is not understood.** Real MCP tools must be invoked via `call_tool(name=..., arguments=...)`; only `call_tool`/`close_session`/`list_tools` are directly callable. The model repeatedly fails this and gives up. Triage of the 166 agent_error+incomplete rollouts: `proper_wrapper` 69, `direct_mcp_names` 61, `no_tool_calls` 36.

- **direct_mcp_names (61).** Model calls the MCP tool by name. Signature, `notes_knowledge_management_1#5`: after `list_tools` it calls `create_database` directly → `{'error': "Unknown tool 'create_database'. The only tools you can call directly are ['call_tool','close_session','list_tools']"}`. Wasted turns, then surrender.
- **no_tool_calls / giving up (36).** Model reads the tool list, sees only the 3 wrapper tools as directly callable, and concludes the task is impossible. `it_asset_management_1#8`: *"The tools available (call_tool, close_session, list_tools) do not support direct manipulation of device assignments... No function exists..."* — then stops or calls `close_session` (→ the 35 `episode_already_done`).

**Scope of the give-up pathology: 92/422 (22%)** of rollouts end on an explicit "not available / cannot be completed / does not exist" message. This single misunderstanding bleeds across buckets — `episode_already_done` (premature close_session), `direct_mcp_names`, `no_tool_calls`, and many `incomplete`.

**Truncation (`max_tokens`) is minor: 35/422** end with an empty final assistant message. Not the dominant problem.

**Worst scenarios** (mean ~0.05–0.10, near-total failure): `membership_management_4` 0.05, `it_asset_management_1` 0.075, `hr_system_1` 0.075, `healthcare_patient_portal_5` 0.083, plus a long tail of finance/booking/CRM tasks at 0.088–0.10. **Best:** `clinic_management_2` 1.00, `team_collaboration_1`/`gaming_2` 0.89, `content_bookmark_management_1` 0.875.

**Recommended fixes, in order.**
1. Fix the system prompt / tool docs so the model uses `call_tool(name, arguments)` from `list_tools` output instead of calling tool names directly or declaring the task impossible. This is the single highest-leverage change (~22% of rollouts surrender on it).
2. Investigate the tail `rollout_error` burst (last 44 rollouts) — env server / trainer stability, not the model.
3. Reduce the scoring-failure band (server_error/judge_error/no_verifier = 55 rollouts forced to 0.1) so reward reflects the policy.
