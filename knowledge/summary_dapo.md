# DAPO: An Open-Source LLM RL System at Scale

**arXiv:** 2503.14476 · ByteDance Seed / Tsinghua AIR · March 2025
**Source:** `~/.cache/research/knowledge/2503.14476/`
**Project page:** https://dapo-sia.github.io/ · built on the `verl` framework

---

## One-line takeaway

DAPO = GRPO with the KL term removed plus **four targeted fixes** that, stacked,
take Qwen2.5-32B (base) from **30 → 50** on AIME 2024 (beating DeepSeek-R1-Zero-Qwen-32B's
47 in **half the steps**). The paper's value is that each fix is an isolated, ablatable
recipe change, not a new algorithm.

## Setup / baseline

- Verifiable math (DAPO-Math-17K, answers transformed to **integers** so a rule parser
  can score them without a reward model — sidesteps reward hacking).
- Rule reward: +1 if answer equivalent, −1 otherwise.
- Naive GRPO baseline (group-normalized advantage, value-function-free) gets only 30.
- **KL penalty removed entirely**: for long-CoT RL the policy is *supposed* to move far
  from the base model, so anchoring to a reference model just fights training.

## The four techniques (each is one ablation row)

Incremental ablation (AIME24 avg@32):

| Step | Score |
|------|-------|
| Naive GRPO | 30 |
| + Overlong Filtering | 36 |
| + Clip-Higher | 38 |
| + Soft Overlong Punishment | 41 |
| + Token-level Loss | 42 |
| + Dynamic Sampling (**DAPO**) | **50** |

### 1. Clip-Higher — decouple the PPO clip range
Single clip ε causes **entropy collapse**: with ε=0.2, a 0.9-prob "exploitation" token
can rise toward 1.0, but a 0.01-prob "exploration" token is capped at 0.012 — exploration
is structurally suppressed. Fix: split into `ε_low=0.2`, `ε_high=0.28`. Raising only the
upper bound gives low-prob tokens room to grow, keeps entropy up, diversifies samples.
Keep ε_low low so token probs aren't driven to 0 (which collapses the sample space).

### 2. Dynamic Sampling — filter zero-advantage groups, oversample to refill
If all G samples in a group are correct (or all wrong), reward is uniform → group advantage
= 0 → **zero gradient**. As training proceeds, the fraction of all-correct prompts grows, so
effective batch size silently shrinks and gradient variance rises. Fix: **oversample and
discard any prompt whose group accuracy is exactly 0 or 1**, keep sampling until the batch
is full of "informative" groups. Constraint: `0 < |{correct o_i}| < G`. Doesn't hurt
wall-clock (long-tail generation dominates anyway) and converges in fewer steps.

### 3. Token-level Policy Gradient Loss
GRPO averages loss per-sample then per-group → every sequence gets equal weight regardless
of length. In long-CoT this means tokens in long responses are under-weighted: good long
reasoning is under-learned, and **gibberish/repetition in long junk responses is
under-penalized**, driving unhealthy length+entropy blowup. Fix: normalize by **total tokens
across the group** (`1/Σ|o_i| · ΣΣ`), so each token contributes equally. Small accuracy gain
but big stability/length-health gain.

### 4. Overlong Reward Shaping (two parts)
Truncated-but-sound reasoning getting a hard −1 is **reward noise**.
- **Overlong Filtering:** mask the loss of truncated samples entirely → big stability win
  (this alone is the 30→36 jump).
- **Soft Overlong Punishment:** length-aware penalty. Within a soft "cache" window before the
  hard max, penalty ramps linearly from 0 to −1; beyond max, −1. Signals "too long" without a
  cliff. (L_max=16384, L_cache=4096, so generation cap = 20480 tokens.)

## Practical training notes worth stealing

- Const LR 1e-6, AdamW, 20-step linear warmup.
- Rollout batch 512 prompts × 16 samples; mini-batch 512.
- **Monitoring discipline** (their stated debugging methodology): watch (a) **response length**
  — stagnation/decline signals deterioration; (b) **reward** — rises stably but *correlates
  poorly with val accuracy* → train-set overfitting; (c) **entropy / mean token prob** — want
  entropy on a *slow upward* trend; collapse = no exploration, spike = gibberish.
- Reflection/backtracking behaviors **emerge during** RL, absent at the start.

---

## Connection to this repo (open-env async-GRPO / AWM)

Our `open-env/openenv_awm_async_grpo.py` runs TRL's `AsyncGRPOTrainer`, and the config block
(`openenv_awm_async_grpo.py:859`) **already adopts two DAPO ideas**:

- `epsilon_high=0.28` — literally commented `# DAPO-style high clip for more exploration`
  (Clip-Higher, technique #1). ✅
- **No KL penalty** — `beta=0`, no reference model (DAPO's KL removal). ✅ (Our reason is
  partly mechanical — old_log_probs are vLLM sampling logprobs — but the effect matches.)

It also uses GSPO-style sequence-level importance sampling (`importance_sampling_level=
"sequence_token"`, `loss_type="grpo"`), which is a *different* axis from DAPO's token-level
**loss normalization** — don't conflate them.

### DAPO ideas we are NOT yet using — candidate experiments

These map cleanly onto our 10-step-cap, one-branch-per-experiment workflow
(see [[feedback_experiment_branch_and_steps]]):

1. **Dynamic Sampling (highest-value, +8 pts in the paper).** Our AWM reward is
   `R ∈ {1.0, 0.1, 0.0}` (`_score_rollout`), so groups where every rollout lands on the same
   value contribute ~zero advantage — exactly the wasted-batch problem DAPO targets. We
   already log per-group `reward_std` in `_save_calibration` (`:574-591`), so we have the
   signal to **filter groups with std≈0 and oversample to refill**. Worth checking whether
   TRL's AsyncGRPOConfig exposes a dynamic-sampling / `generation_batch_size` overshoot knob
   before hand-rolling it. *This is the first thing I'd try.*

2. **Soft Overlong Punishment / Overlong Filtering.** We have `max_completion_length` +
   `thinking_token_budget` and currently force `reward=-1.0` on truncation/format-error
   (`:542`). That hard −1 on a *truncated-but-reasonable* trajectory is precisely the reward
   noise DAPO's overlong filtering removes. Option: **mask loss on length-truncated rollouts**
   (distinct from genuine format errors, which deserve the −1), or ramp a soft length penalty.

3. **Token-level loss normalization.** Less relevant for us — our multi-turn samples are
   windowed and loss-masked per assistant turn (`_windowed_messages`, `completion_mask`), and
   we're on sequence-level GSPO. Probably skip unless we see length/entropy blowup in W&B.

4. **Entropy as a first-class monitor.** DAPO's core diagnostic is entropy trend. If not
   already on our W&B dashboard alongside `reward_ema`, add it — entropy collapse would be the
   first symptom that `epsilon_high=0.28` isn't enough for the AWM setting.

Related notes: [[summary_agent_world_model]] (the AWM env + R∈{1.0,0.1,0.0} reward this
trainer scores against), [[feedback_experiment_branch_and_steps]].
