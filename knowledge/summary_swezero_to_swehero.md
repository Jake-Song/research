# From SWE-Zero to SWE-Hero: Execution-free to Execution-based Fine-tuning for SWE Agents

**arXiv:** 2604.01496 (NVIDIA, COLM 2026 preprint)
**Authors:** Nikolai Ludwig, Wasi Uddin Ahmad, Somshubra Majumdar, Boris Ginsburg

## TL;DR

Two-stage SFT recipe for software-engineering agents that distills frontier open-weight LLMs into smaller agents (Qwen2.5-Coder 7B/14B/32B):

1. **SWE-Zero** — large-scale **execution-free** SFT on ~300k trajectories generated without any task-specific Docker environment. Cheap, broad, teaches "code semantics."
2. **SWE-Hero** — much smaller (~13k) **execution-backed** SFT on top, run inside real containers. Adds the test/feedback loop ("grounded refinement").

Result: SWE-Hero-32B hits **62.2%** on SWE-bench Verified (vs 55.7% if you skip the Zero stage and go directly to Hero), and **44.1%** on SWE-bench Multilingual despite training only on Python.

## Why this paper exists (the bottleneck)

Standard SWE-agent training (SWE-Gym, R2E-Gym, OpenSWE, etc.) requires per-task Docker environments. The authors argue this creates three scaling walls:

- **Data**: most real-world repos/PRs can't be containerized cleanly → discarded.
- **Training**: orchestrating thousands of task-specific images is expensive infra.
- **Inference**: TTS / parallel rollouts pay the cost of resetting environments every time.

They observe that frontier teachers like **Qwen3-Coder-480B** can already resolve **69.5% of SWE-bench Verified without any execution** — i.e., the teacher's "world model" is good enough to skip the runtime entirely. So why force students to learn from execution-grounded trajectories only?

## Method

### Setup
- **Scaffold:** OpenHands; tools = `str_replace_editor`, `execute_bash`, `think`, `finish`.
- **Teacher:** Qwen3-Coder-480B-A35B-Instruct.
- **Student:** Qwen2.5-Coder-Instruct (7B / 14B / 32B), trained with multi-turn SFT, tool outputs masked from loss, YaRN to extend context 32k → 128k.
- **Data sources:** ~180k Python tasks aggregated from SWE-Gym, R2E-Gym, SWE-rebench, SWE-Lego across 3.5k+ repos. 13.5k of these have working containerized envs.

### SWE-Zero trajectories (execution-free)
- Teacher gets problem statement + repo at base commit + a **generic sandbox** (no per-task setup, no ability to run repo code).
- Trajectories follow 5 phases: requirements analysis → repo exploration → fix localization → patch implementation → review.
- ~40% fewer tokens than execution-backed counterparts.
- Generate N=3–5 rollouts per task.
- **Filtering pipeline:**
  1. Rule-based bash parser to discard trajectories where the LLM cheated and tried to actually run code (~35% pruned).
  2. Quality filter: discard if exceeds step limit, null patch, modifies a file in the held-out test patch (shortcut prevention), wrong tool-call cardinality, or ≥3 `str_replace_editor` errors.
- Final: **300k trajectories across 150k tasks** (2 trajectories each).

### SWE-Hero trajectories (execution-backed)
- Teacher gets the full Docker environment.
- Single rollout per task. Skip stage-1 filter (no need to police execution).
- Don't drop based on whether the task was actually resolved (corpus too small to be picky).
- Final: **13.2k trajectories**.

### Training
- Stage 1: SFT on SWE-Zero → "SWE-Zero-Agent."
- Stage 2: continue SFT on SWE-Hero, initialized from the SWE-Zero-Agent → "SWE-Hero-Agent."
- 3 epochs, batch 32, cosine LR 1e-5 → 1e-8, 0.1 warmup.
- Inference: temp 0.7, top-p 0.8, top-k 20, 100 turns max, 128k ctx.

### Anti-leakage
- Strip all git commits/tags/branches created after the base commit ("git hacking" prevention).
- SWE-Zero is structurally immune to git-history leakage because it has no env to query.

## Key empirical results

| Model | SWE-bench Verified | SWE-bench Multilingual |
|---|---|---|
| SWE-Zero-Agent-32B | 57.5% | — |
| SWE-Hero-Agent-32B | **62.2%** | **44.1%** |
| Direct-to-Hero (skip Zero) 32B | 55.7% | 30.8% |
| SWE-Hero-Agent-7B | 52.7% | — |
| SWE-Hero-Agent-7B + TTS@32 | 57.9% (+5.2) | — |

Headline takeaways:
- **The Zero stage is load-bearing.** Skipping it loses ~6.5 pts on Verified and ~13 pts on Multilingual. The big multilingual gap shows the Zero stage is what gives the agent generalizable reasoning patterns.
- **Data scaling on Zero is real but diminishing.** 4k → 150k PRs takes 14B from 42.1% → 49.1%; most of the gain is in the 4k → 32k regime.
- **TTS works but is verifier-limited.** Best@K vs Pass@K gap is ~15% at K=16 and growing — the bottleneck is the open-source verifier's discrimination, not the policy.
- **Efficiency:** Zero trajectories are ~40% shorter (10–30 turns vs 20–60 for Hero). Hero spends more tokens but resolves harder bugs.

## Mental model

Think of it as a curriculum:

1. **Cheap & broad first** — flood the model with "what does a reasonable patch look like for this repo?" across the entire long tail of GitHub, including repos you could never containerize.
2. **Expensive & narrow second** — on the small subset where you *can* run tests, teach the model the iterative test-fix-test loop.

The structural insight: **semantic intuition before runtime grounding** outperforms either alone, and the cheap-stage data is what unlocks cross-language transfer.

## Connection to this repo (sparse autoencoders / Delphi explainer-scorer)

This repo isn't an SWE-agent project — it's SAE feature interpretation work. But there are some transferable patterns worth noting:

### 1. The "free signal vs. expensive signal" curriculum
Our `sae/explain.py` + `sae/score.py` pipeline (Delphi-based) is itself a two-stage thing: an LLM produces a feature explanation cheaply from activations, then a scorer validates it. The SWE-Zero idea — *generate a huge volume of cheap, unverified outputs first, then refine on a small verified set* — is directly analogous. If we ever fine-tune a small SAE-explanation model, this recipe suggests:

- **Stage 1 (zero):** distill a huge teacher (e.g. a frontier LLM) on lots of (activation context → explanation) pairs without checking the explanation against the scorer. Filter only on shallow heuristics (well-formed, non-degenerate, didn't peek at held-out activations).
- **Stage 2 (hero):** continue SFT on the much smaller subset where the explanation actually scored well via `score.py`.

### 2. "Filter out trajectories that touched held-out data"
SWE-Zero filters trajectories where the agent modified a file present in the test patch. This is the same problem we have in SAE explanation: the explainer must not see the held-out activations the scorer will use. Worth being deliberate about that split in our pipeline.

### 3. Verifier ceiling is the bottleneck for TTS
For SAE explanations, the analogue is: even if we generate K explanations and pick the best, we're capped by how good the **scorer** is at distinguishing good from bad explanations. The paper's ~15% Best@K vs Pass@K gap is a useful warning — investing more in the scorer (cleaner held-out activations, better prompting, ensemble verifiers) probably matters more than scaling K.

### 4. Cross-domain transfer from a broad cheap stage
SWE-Zero (Python only) transfers well to other languages purely because the cheap stage is large and diverse. If we ever train a feature-interpreter, broad coverage at the cheap stage may matter more than depth — train across many SAEs / many layers / many models cheaply, then refine on the one we care about.

### Things NOT to copy
- This is a 7B–32B SFT paper. Doing SFT on a teacher's traces is overkill for our explainer scoring loop unless we explicitly want a small specialized model to replace the explainer LLM.
- The agent scaffolding (OpenHands, tool calls, 100-turn rollouts) has no analogue here.

## Open questions the paper leaves

- How well does the "zero → hero" recipe transfer to **reasoning-style** models (R1, o-series style internal CoT)? The authors flag this as future work.
- Can RL replace the Hero stage and squeeze more out of the same 13k verified envs? Other works (DeepSWE, SkyRL) suggest yes; this paper deliberately doesn't try.
- The verifier ceiling for TTS — building a discriminative reward model is the named open problem.
