# Qwen-Scope: Turning Sparse Features into Development Tools for LLMs

**Authors:** Qwen Team (Alibaba) — core: Boyi Deng, Xu Wang, Yaoning Wang, Yu Wan, Yubo Ma, Baosong Yang
**Date:** 2026-04-30
**Release:** [huggingface.co/collections/Qwen/qwen-scope](https://huggingface.co/collections/Qwen/qwen-scope) · [modelscope.cn/collections/Qwen/Qwen-Scope](https://modelscope.cn/collections/Qwen/Qwen-Scope)
**Full text:** [qwen_scope.txt](qwen_scope.txt)

## TL;DR

Qwen-Scope is an open-source suite of **Sparse Autoencoders (SAEs)** trained on **every layer** of 7 Qwen3 / Qwen3.5 backbones (dense + MoE), released as 14 SAE groups. The contribution is not the SAEs themselves — it's the argument and evidence that SAE features should be treated as a **reusable representation-level interface** for model development, not just a post-hoc analysis tool. The paper demonstrates this across four directions: inference-time steering, evaluation-set analysis, data-centric workflows (classification + synthesis), and post-training (SFT + RL).

## The SAE release

| Backbone | Type | Layers | Hidden | SAE width | Expansion | Top-k (L0) |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | Dense | 1–28 | 2048 | 32K | 16 | 50, 100 |
| Qwen3-8B-Base | Dense | 1–36 | 4096 | 64K | 16 | 50, 100 |
| Qwen3.5-2B-Base | Dense | 1–24 | 2048 | 32K | 16 | 50, 100 |
| Qwen3.5-9B-Base | Dense | 1–32 | 4096 | 64K | 16 | 50, 100 |
| Qwen3.5-27B-Instruct | Dense | 1–64 | 5120 | 80K | 16 | 50, 100 |
| Qwen3-30B-A3B-Base | MoE | 1–48 | 2048 | 32K / 128K | 16 / 64 | 50 / 100 |
| Qwen3.5-35B-A3B-Base | MoE | 1–40 | 2048 | 32K / 128K | 16 / 64 | 50 / 100 |

**Training notes:**
- Top-k SAE on residual-stream activations, trained on in-house pretraining data.
- Auxiliary loss (weight 1/32, following Gao et al. 2024) to keep dead features near zero.
- Filter activations with extremely large L2 norms (esp. first-token activations on Qwen3-1.7B / 8B) to stabilize reconstruction (Marks et al. 2024).
- Qwen3.5-27B is the only backbone whose SAEs are trained on the **instruct** variant; all others on base.
- For MoE models, an additional wider 128K / 64× SAE is released to capture finer-grained features.

## Four applications

### 1. Inference-time steering (§3)

Standard recipe: `h' = h + α·d`, where `d` is an SAE feature direction.

**Two ways to find features:**
- **Contrastive** — pick a target behavior, build positive + negative example sets, rank features by mean-activation difference (He et al. 2025, Bayat et al. 2025).
- **Auto-interpretation** — describe features from their top-activating contexts using a stronger LLM (Paulo et al. — the Delphi pipeline, [[summary_delphi_autointerp]]).

**Two case studies on Qwen3:**
- Diagnose unintended Chinese mixing in English output → rank features on the bad response → find feature 6159 ("Chinese") → suppress → English restored.
- Style transfer: activate feature 36398 ("classical Chinese") to rewrite a modern Chinese story in classical literary style.

### 2. Evaluation analysis (§4)

Treats a benchmark's "feature footprint" `F(D) = ⋃ F(xᵢ)` as a fingerprint of what it probes — letting you analyze benchmarks **without running models**.

**Benchmark redundancy.** Defines a feature-coverage curve `cₙ = E[|F(S)| / |F(D)|]` analogous to the Kendall-τ ranking curve `τₙ`. The scalar metric is
```
R̂(D) = AUC(cₙ) · N / |F(D)|     (Equation 9)
```
The `N/|F(D)|` correction prevents small-feature-set benchmarks from looking artificially diverse.

**Headline result:** Across 17 benchmarks (MMLU, MMLU-Redux/Pro, SuperGPQA, C-Eval, CMMLU, GSM8K, MATH, GPQA-D, TheoremQA, MBPP, EvalPlus, MultiPL-E, MMMLU, INCLUDE, KOR-Bench, ICLEval) and 26 Qwen pretraining checkpoints, `R̂(D)` correlates with the true performance-redundancy `R(D)` at **Spearman ρ ≈ 0.85** (Pearson r ≈ 0.78 in log-y). GSM8K is high-redundancy despite small size; SuperGPQA is low-redundancy despite 26K samples.

**Inter-benchmark similarity.** Asymmetric and min-normalized feature-overlap matrices reveal e.g. GSM8K↔MATH ≈ 0.63, MBPP↔MultiPL-E ≈ 0.53, with code and math forming distinct clusters.

### 3. Data-centric workflows (§5–6)

**Toxicity classification (§5).** A simple rule-based classifier on top of contrastively-discovered "toxic" features:
- Cross-lingual transfer: features discovered in English generalize to other languages.
- Multi-layer composition helps mostly when single-layer signal is weak; rank layers by `top1-diff` and only add layers when needed.
- **Data efficiency:** ~10% of the discovery set recovers ~99% of full-data classifier performance. The most stable toxic-biased features are found early.

**Safety data synthesis (§6).** Move data construction from the prompt level to the **representation level**:
1. Define a seed corpus `D_seed`. For each feature `(ℓ, f)`, compute a binary coverage `c_f^(ℓ)(D_seed)`.
2. Pair each feature with a natural-language explanation; a judge model assigns relevance score `s_f^(ℓ)` and gives the candidate inventory `T = {(ℓ,f) : s_f^(ℓ) ≥ τ}`.
3. Highest-priority targets are the **uncovered** ones: `T_miss = {(ℓ,f) ∈ T : c_f^(ℓ)(D_seed) = 0}`.
4. For each target, generate a vanilla prompt + adversarial rewrites, plus a refusal-style or benign response (depending on safety label). **Verify in feature space:** retain only examples where `h_{i,f}^(ℓ) = 1`.

**Headline numbers (Qwen3-8B, layer 30, ~65K-latent SAE, WildJailbreak seed):**
- Feature-driven synthesis reaches **99.74% target-feature coverage**; random safety-related synthesis hits ~90% at the same budget; natural sampling plateaus much lower.
- With only **4k real + 4k feature-driven synth** safety data: ASR ↓ to 24.0, RR ↓ to 20.5, Acc 77.75 — approaches the **120k safety-only** baseline (Acc 78.75) while topping IFEval (53.23) and TruthfulQA (57.32).
- Holds when prompt+response generation is swapped from GPT-4/3.5 to Gemini-3-Flash → the gain comes from feature targeting, not the generator.

### 4. Post-training: SASFT (§7)

**Problem:** Unexpected code-switching — multilingual LLMs occasionally drop tokens of an unintended language into a response. Standard SFT only pushes toward the target response and gives no negative signal against the wrong language.

**Method (Deng et al. 2026, ICLR):**
1. Identify a target-language SAE feature contrastively (positive = target-language texts, negative = non-target).
2. Add an auxiliary **feature-suppression loss** during SFT on non-target-language data:
```
L_training = L_cross-entropy + λ · L_suppress
```
This explicitly trains the model to *not* activate the unwanted-language feature.

### 4. Post-training: SAE-guided RL (§8)

**Problem:** Endless repetition is a low-frequency RL failure mode — vanilla DAPO rollouts almost never produce repetitive samples, so RL has no signal to suppress them.

**Method:** In each DAPO rollout group of size G, replace one sample with an **SAE-steered repetitive rollout** (amplify a repetition feature during generation). This guarantees the group contains a rare-negative example that gets a low reward, providing an explicit gradient against repetition.

**Results across Qwen3-1.7B / 8B / 30B-A3B:** Repeat ratio drops sharply and stays much lower than vanilla RL throughout training. General benchmarks (MMLU, Flores, HellaSwag, LogiQA, IFEval, MGSM) stay competitive — MGSM improves notably (+5.56 on 1.7B, +5.84 on 30B-A3B). Effect on other capabilities is mixed and task-dependent.

## Why this matters

The throughline: **the same SAE feature dictionary supports all four directions.** Once you have it, you can:
- *steer* features at inference,
- *measure* benchmark coverage in feature space,
- *prioritize* training data by which features it does or doesn't activate,
- *inject* feature-targeted losses or rollouts during SFT / RL.

Prior SAE releases (Gemma Scope, Llama Scope) emphasized the artifact; Qwen-Scope's framing emphasizes the **workflow**. The data-synthesis and RL-rare-negative results are the most novel contributions.

## Future directions (§9.2)

- **Reasoning-model interpretability** across CoT branches and resampled trajectories (Macar et al. 2026, Bogdan et al. 2025).
- **Internals-based monitoring** for deception, hidden objectives, jailbreak susceptibility, hallucination.
- **Model diffing** before/after fine-tuning or RL — which features change?
- **Interpretability-driven training** generalizing the SASFT / RL recipes.
- **Data-centric interpretability** — feature-coverage as a data-curation signal.
