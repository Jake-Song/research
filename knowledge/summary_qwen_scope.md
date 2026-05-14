# Qwen-Scope: Sparse Features as Development Tools for LLMs

**arxiv:** [2605.11887](https://arxiv.org/abs/2605.11887) — Qwen Team (Alibaba)
**Venue framing:** COLM 2024 template, technical report style
**Core claim:** SAE features should be treated as a **reusable representation-level interface for model development** (steering, evaluation, data curation, post-training), not merely a post-hoc analysis tool.

## 1. The SAE Release

14 SAE groups across 7 Qwen3 / Qwen3.5 backbones, residual stream, **all layers**, Top-k activation.

| Backbone | Type | Layers | Hidden | SAE width | Expansion | Top-k |
|---|---|---|---|---|---|---|
| Qwen3-1.7B-Base | Dense | 1–28 | 2048 | 32K | 16 | {50, 100} |
| Qwen3-8B-Base | Dense | 1–36 | 4096 | 64K | 16 | {50, 100} |
| Qwen3.5-2B-Base | Dense | 1–24 | 2048 | 32K | 16 | {50, 100} |
| Qwen3.5-9B-Base | Dense | 1–32 | 4096 | 64K | 16 | {50, 100} |
| Qwen3.5-27B-Instruct | Dense | 1–64 | 5120 | 80K | 16 | {50, 100} |
| Qwen3-30B-A3B-Base | MoE | 1–48 | 2048 | 32K + 128K | 16 / 64 | 50 / 100 |
| Qwen3.5-35B-A3B-Base | MoE | 1–40 | 2048 | 32K + 128K | 16 / 64 | 50 / 100 |

**Training stability tricks:**
- Auxiliary loss weight 1/32 (Gao et al. 2024) → almost zero dead features at convergence.
- L2-norm outlier filter on activations (Marks et al. 2024) — outliers cluster on first-token positions in Qwen3-1.7B/8B.
- Qwen3.5-27B trained on the **instruct** model; all others on **base**.
- MoE models get a second, much wider (128K, 64×) SAE for finer-grained features.

## 2. Steering (§3)

Standard form: `h' ← h + α · d` where `d` is an SAE feature direction. Two ways to find features:

1. **Contrastive** — define positive/negative prompt sets, rank features by activation difference.
2. **Auto-interpretation** — collect top-activating contexts, ask a strong LLM for a natural-language description (Delphi-style pipeline, Paulo et al.).

**Two demos on Qwen3:**
- Diagnosing a Chinese leak in an English response → rank by activation → feature 6159 ("Chinese") → suppress → English restored.
- Style transfer: amplify feature 36398 ("classical Chinese") to convert a modern Chinese continuation into classical literary style.

The steering equation `h' = h + α·d` is the same operator used in both directions — only the sign of α changes.

## 3. Evaluation (§4) — Benchmark Analysis Without Running Models

Per-sample feature footprint:
```
F(x_i) = { j : z_j(x_i) > 0 }  (last-token SAE encoding)
F(D)   = ⋃ F(x_i)
```

### Redundancy (intra-benchmark)
Performance-based redundancy `R(D) = (1/N) Σ τ_n` measured via Kendall's τ between full-benchmark and subset rankings across M models. Expensive — requires `O(M·N)` forward passes.

**Feature-based proxy** (no model evaluation):
```
ĉ_n  = E[|F(S)| / |F(D)|]        over random subsets S of size n
R̂(D) = AUC(c_n) · N / |F(D)|     # Eq. 9
```
The `N/|F(D)|` correction prevents a small absolute feature set from looking artificially "diverse" purely because its normalized coverage curve climbs fast.

**Headline:** Across 17 benchmarks × 26 Qwen pretraining checkpoints, Spearman ρ(R, R̂) ≈ **0.85**. GSM8K is high-redundancy despite small size (1.3K); SuperGPQA is low-redundancy despite 26K samples — feature-redundancy captures this where raw size cannot.

### Inter-benchmark similarity
Asymmetric overlap:
```
overlap(D1, D2) = |F(D1) ∩ F(D2)| / |F(D1)|
```
Symmetric (min-normalized) version correlates with cross-model Pearson similarity at **75.5%** after partialling out MMLU as a general-ability proxy (up from 68.4% raw). Practical use: low-overlap benchmarks probe distinct capabilities (keep both); high-overlap pairs are consolidation candidates. Example: `overlap(GSM8K, MATH) = 0.63`, `overlap(MATH, GSM8K) = 0.10` — MATH subsumes GSM8K's feature footprint, not vice versa.

## 4. Data Classification (§5) — Toxicity

Pipeline reduces to two stages, no trained head:

1. **Discovery** on a fixed split (2K toxic + 2K clean per language). Compute per-feature firing-rate difference
   ```
   Δ_f^(ℓ) = P(h=1 | y=1) − P(h=1 | y=0)
   ```
   and take the top K per layer.
2. **Inference**: OR-rule over the selected feature set
   ```
   ŷ = 1[ max_{f ∈ S_ℓ, t} a_{f,t}^(ℓ) > ε ]
   ```

**Findings:**
- F1 > 0.90 on English with very small K (1–10) for both Qwen3-1.7B and 8B; best layer is middle-to-late.
- Cross-lingual transfer: English-discovered features generalize well to European languages, weaker for Amharic/Arabic/Chinese — feature overlap peaks in middle layers and is more stable in the larger model.
- **`top1-diff` layer-selection proxy** — pick the layer where the single best feature has the largest Δ, no eval data needed — recovers near-best layer almost everywhere.
- **Data efficiency:** ~10% of discovery data → ~99% of full-data F1. Top toxic-biased features are the most stable; they emerge early.

## 5. Data Synthesis (§6) — Safety-Oriented

The data-construction unit moves from prompts to **internal directions**:

1. Build seed corpus `D_seed`; compute coverage `c_f^(ℓ)(D_seed)` (any firing?).
2. Pair each feature with a natural-language explanation; judge model scores semantic relevance `s_f^(ℓ)` → eligible set `T = { (ℓ,f) : s ≥ τ }`.
3. Priority synthesis targets: `T_miss = { (ℓ,f) ∈ T : coverage = 0 }`.
4. For each target, generate **vanilla prompt + adversarial rewrites**, assign safety label, generate refusal-style or benign response.
5. **Representation-level verification** — keep only examples where `h_{i,f}^(ℓ) = 1`.

**Setup:** Qwen3-8B, layer-30 SAE (~65K latents), WildJailbreak seed corpus.

**Coverage:** Feature-driven synthesis reaches **99.74%** of target features at a fixed budget vs. random safety-related synthesis ≈ 90% and natural sampling much lower.

**Downstream SFT (Alpaca + safety + synth):**
| Setting | ASR↓ | RR↓ | Acc↑ | IFEval | TQA | MMLU |
|---|---|---|---|---|---|---|
| + Safety 8k | 22.0 | 34.5 | 71.75 | 53.05 | 57.11 | 76.25 |
| + Safety 120k | 21.0 | 21.5 | **78.75** | 48.06 | 54.80 | 76.34 |
| + Safety 4k + Random synth 4k | 20.0 | 36.0 | 72.00 | 48.98 | 56.94 | 76.08 |
| + Safety 4k + **Feature synth 4k** | 24.0 | 20.5 | 77.75 | **53.23** | **57.32** | **76.58** |

4k+4k with feature-driven synth approaches the 120k safety-only result while topping IFEval/TQA/MMLU. Holds when swapping GPT-4/3.5 for Gemini-3-Flash — the gain is from feature targeting, not the generator.

## 6. SASFT (§7) — Auxiliary Loss for Code-Switching

**Problem:** Multilingual LLMs occasionally leak tokens of an unintended language; standard SFT supplies no negative signal.

**Two observations that motivate the method:**
- Pre-activation of the unintended language's SAE feature **rises in the tokens leading up to** the first code-switched token, peaks at the switch.
- Directional ablation `x' = x − λd` at the immediately-preceding token monotonically cuts the code-switching ratio (irrelevant features have no effect → causal).

**Method:** Identify language-specific features by **monolinguality score** `ν_s^L = μ_s^L − γ_s^L`. Add a suppression loss on non-target-language data:
```
L_reduce = E_{j ≠ L} E_{x ∈ D_j} Σ_{s ∈ S_L} ReLU(f_s(x) − α_j)
L_train  = L_CE + λ · L_reduce
```
α_j is a per-language pre-estimated baseline (using 0 would be too aggressive when average pre-activation is negative).

**Results (Qwen3-1.7B / 8B, 110k & 210k mixes; targets zh, ru, ko):** SASFT cuts code-switching ratio by ≥50% in most cells; some cells go to 0% (Qwen3-1.7B → ko). General benchmarks (MMLU, HumanEval, Flores, HellaSwag, LogiQA, IFEval, MGSM) stay flat or improve, except for a small dip on a couple of benchmarks.

## 7. SAE-Guided DAPO (§8) — Rare-Negative Augmentation in RL

**Problem:** Endless repetition is a low-frequency RL failure mode — vanilla DAPO rollouts almost never produce it, so RL gets no gradient against it.

**Failed first attempt:** Use SAE steering to generate *positive* rollouts. Bad: steering doesn't fix multi-step reasoning, degrades fluency, and the model learns bad patterns.

**Revised:** Use steering to generate *negative* rollouts. Fluency doesn't matter when the model is meant to avoid them. Identification:
- For each repeated token, compare SAE activation at first vs. last occurrence in context. Features with the largest jump are "repetition features" (causal: bidirectional steering both induces and suppresses repetition).
- Repetition features also fire in **benign** repetition (e.g., reproducing answer choices) — so we do **not** suppress them at training time (would hurt normal behavior). Instead, augment.

**Algorithm:** In each DAPO rollout group of size G, sample G−1 normal outputs and **one** SAE-steered output (`h' = h + α·d`) biased toward repetition. The steered sample reliably gets a low reward → explicit negative gradient against repetition.

**Results on Qwen3-1.7B / 8B / 30B-A3B:** Repeat-ratio drops sharply within the first training steps and stays much lower than vanilla DAPO throughout. General-capability benchmarks remain competitive; MGSM gains notably (+5.56 on 1.7B, +5.84 on 30B-A3B). Other benchmarks are mixed but never significantly worse.

## 8. Why the SAEs are the Same SAEs Everywhere

The four applications share one feature dictionary per (model, layer). That's the practical point of the report: once you've trained an SAE, the marginal cost of each new use case is small.

| Direction | What changes per use case |
|---|---|
| Steering | Sign and magnitude of α |
| Evaluation | What set of inputs you encode |
| Classification | Which features you OR over |
| Data synthesis | Which features you target + a feature-space accept filter |
| SASFT | An auxiliary loss penalizing one feature set |
| SAE-DAPO | One steered rollout per group |

## 9. Connections to this Repo (`/home/jake/research/sae/`)

Your local `sae/` looks like a **reimplementation of pieces of Qwen-Scope** against an externally-trained Qwen3.5-2B/9B SAE checkpoint (`layer{L}.sae.pt`, W_enc 32K×2048, Top-k=50). Direct correspondences:

- **`sae.py`** — Minimal SAE encoder (`pre_acts = residual @ W_enc.T + b_enc; topk(50)`). Equivalent to the Top-k ReLU operator referenced in §2.
- **`identify_features.py`** — Almost certainly the §3 / §5 contrastive feature-discovery step (POS/NEG → mean-activation Δ). `pipeline.py:84–110` implements exactly that with a sentiment toy set.
- **`classify_features.py`** — §5 toxicity-classifier OR-rule. Your modified status (in `git status`) suggests you're iterating on this — paper recipes worth checking: top1-diff layer selection (§5.3.1), multi-layer composition (only when single-layer weak), 10% discovery-data sufficiency.
- **`benchmark_redundancy.py`** — §4 redundancy / overlap. The exact formula to mirror is `R̂(D) = AUC(c_n) · N / |F(D)|` (Eq. 9); compute `c_n` by random-subset sampling rather than enumerating.
- **`explain.py` + `score.py` + `pipeline.py`** — Delphi auto-interpretation (DefaultExplainer + DetectionScorer + FuzzingScorer) — this is §3's "automatic interpretation methods" branch. Already wired through OpenRouter / Claude Sonnet 4.5.
- **`steer.py`** — §3 / §8 steering. `h' = h + α·d`. The §8 contribution is using *negative* steering for RL rare-negatives — only relevant if you wire this into a DAPO/GRPO loop (you have a `grpo/` folder).

**If you're reproducing:** the order in the paper (training → steering → eval → classification → synthesis → SFT → RL) is also a sensible build order, since each downstream step reuses the same SAE + the same `(ℓ, f) → meaning` mapping you built earlier. The §5 toxicity classifier is the smallest end-to-end win and a good first integration test for a fresh SAE checkpoint.

**If you want a low-effort novel angle:** the paper leaves "uncovered but eligible" features as the priority synthesis targets (§6), but stops short of using **feature-coverage curves** of a candidate training set as a *data-selection* signal (rather than synthesis) — i.e., greedy subset selection on real data to maximize `|F(S)|`. Mentioned as a future direction in §9.2 ("Data-centric interpretability") but not done. Your `benchmark_redundancy.py` plus a sampler would be 80% of that machinery.

## 10. Future Directions (paper §9.2)

- **Reasoning-model interpretability** — features across CoT branches / resampled trajectories.
- **Internals-based monitoring** — deception, hidden objectives, jailbreak susceptibility, hallucination.
- **Model diffing** — which features shift under SFT/RL?
- **Interpretability-driven training** — generalize SASFT / DAPO rare-negative to other low-frequency failure modes.
- **Data-centric interpretability** — feature coverage as a data-curation signal.
