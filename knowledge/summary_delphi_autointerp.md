# Automatically Interpreting Millions of Features in Large Language Models

**Authors:** Gonçalo Paulo, Alex Mallen, Caden Juang, Nora Belrose (EleutherAI + Northwestern)
**Arxiv:** [2410.13928](https://arxiv.org/abs/2410.13928)
**Code:** [github.com/EleutherAI/delphi](https://github.com/EleutherAI/delphi) — this is the `delphi` library already used by `sae/pipeline.py` in this repo.

## TL;DR

SAEs blow neuron-level interpretability up into millions of features per model, so manual labeling is hopeless. This paper builds an end-to-end pipeline that (1) uses an LLM "explainer" to produce a natural-language description for each SAE feature from its activating contexts, and (2) introduces **five cheap scoring methods** for those interpretations that beat the previous standard (simulation scoring, from Bills et al. 2023) on cost and surface different failure modes. They release Delphi and publish explanations for Gemma-2-9B and Llama-3.1-8B SAEs on Neuronpedia.

The headline contributions:

1. **A binary-classifier framing** of "what counts as a good interpretation": the interpretation should let a scorer LLM separate activating from non-activating contexts. This is a departure from simulation scoring, which only looks at activating examples (~0.01% of the relevant distribution).
2. **Four correlational scoring methods** — detection, fuzzing, surprisal, embedding — all 5x to 30x cheaper than simulation.
3. **Intervention scoring** — a *causal* scoring method that catches "output features" (features whose meaning lives in their downstream effect, not in their input patterns) that correlational methods miss.
4. Practical findings about how to sample contexts and prompt the explainer.

## The pipeline

Three sequential phases:

1. **Collect activations.** Run the target model on a corpus (10M tokens of RedPajama-v2), record which SAE features fire where. Important practical wrinkle: on Gemma-2-9B-131k with 256-token contexts on RPJv2, **15% of features never activate** and **30% activate <200 times**. Numbers shrink to ~1% / ~15% on Pile with the same context size, and to ~5% no-fire even on RPJv2 with 1024-token contexts. So sampling distribution and context length matter for "alive" feature coverage.
2. **Generate interpretation.** Show the explainer (Llama-3.1-70B-Instruct) ~40 activating examples (32 tokens each) with the activating tokens delimited by `<< … >>` and their activation strengths printed in parens after the example. Ask for a concise NL description. Cost ≈ \$200 per 1M features. No COT (didn't help with Llama 70B).
3. **Score the interpretation.** Apply one or more of the five scoring methods below.

### Sampling examples for the explainer

This is the most actionable design lesson:

- **Top-only sampling** → concise, specific interpretations that **overfit to the top decile**. Higher precision, lower recall — they fail on the lower-activation tail.
- **Random / quantile-stratified sampling** → broader interpretations that generalize across the activation distribution.
- Recommendation: **stratified sampling across the deciles** of the activation distribution.

The current `sae/pipeline.py` in this repo only ever shows the explainer the (≤8) hand-written `POS` sentences, so this consideration doesn't bite yet — but as soon as the pipeline starts pulling activating contexts from a real corpus, decile-stratified sampling is the default to reach for.

## The five scoring methods

Ranked roughly by cost, cheapest first:

### 1. Embedding (cheapest, ~\$50 / 100k features)
Embed activating and non-activating contexts as "documents" with a small encoder (400M params worked; 7B didn't help much), embed the interpretation as a "query", and use cosine similarity AUROC as the score. Great for fast filtering; not great for quality (Spearman 0.32 vs humans).

### 2. Detection (\$588 / 100k features, Llama-70B)
Show the scorer 5 candidate contexts and the interpretation. Ask which sentences activate the feature. AUROC over many such examples. **Forgiving** — does not require localizing to a specific token. Spearman 0.59 vs humans.

### 3. Fuzzing (\$676 / 100k features)
Same idea as detection, but tokens are delimited with `<<…>>` markers (some real, some fake) and the scorer judges whether the *highlighted* tokens are the ones that actually activate the feature. Closer in spirit to simulation. **Highest correlation with human ratings (Spearman 0.69).** Combine with detection: high-fuzz / low-detection means the explainer found the right tokens but wrong context.

### 4. Surprisal
Compare `log p(context | interpretation)` to `log p(context | dummy interpretation)`. AUROC over activating vs non-activating contexts. Requires logprob access (rules out most closed models). Authors think this one is underdeveloped — lowest correlation with the others.

### 5. Simulation (the prior SOTA — included as baseline only)
Bills et al. 2023. \$3.6k–\$18.7k / 100k features depending on AAO vs token-by-token. Only evaluates on activating examples (bias). Highly correlated with fuzzing because both reduce to "predict which tokens have nonzero activation".

### 6. Intervention scoring (the new causal method)

Defined as:

$$S = \mathbb{E}_{\mathbf x}\Big[\mathbb{E}_{\mathbf i \sim \mathcal G_I(\mathbf x)}[\log p_{\mathcal M}(\mathbf z | \mathbf i)] - \mathbb{E}_{\mathbf g \sim \mathcal G(\mathbf x)}[\log p_{\mathcal M}(\mathbf z | \mathbf g)]\Big]$$

where $\mathbf z$ is the interpretation, $\mathcal G(\mathbf x)$ is clean generation from prompt $\mathbf x$, and $\mathcal G_I(\mathbf x)$ is generation while clamping the feature ("intervention $I$"). Big positive $S$ means: text generated under the intervention is more describable by $\mathbf z$ than clean text is. Must compare interventions at **fixed strength** $\sigma(I) = E_x[D_{KL}(\text{clean} \| \text{intervened})]$, otherwise extreme clamps trivially win.

Key empirical finding: fuzzing and intervention scores are **slightly anti-correlated**. There exist "output features" with no clean input description but a clean output description (their concrete example: a Gemma-2-9B-L32 feature that, when activated, makes the model say things related to "reputation", but doesn't itself fire on reputation-related inputs). Pure correlational interpretability misses these.

This is the reason `sae/steer.py` exists in this repo, and it's the part of the paper most worth chasing here.

## Other findings (skim, but worth knowing)

- **SAE size matters.** Larger SAEs → higher per-feature scores. Way better than neurons even when neurons are sparsified by top-k.
- **Residual-stream SAEs slightly beat MLP-output SAEs.** (Current `sae/pipeline.py` hooks the residual stream — good.)
- **Layer depth.** Early layers score worse; flat after the first few.
- **Explainer model size.** More examples → marginally better. Claude-3.5-Sonnet ≈ Llama-3.1-70B for interpretation quality. (Scorer model size matters more than explainer model size.)
- **Multi-method evaluation is recommended.** Use ≥2 of fuzzing/detection/embedding because each has a different failure mode.

## Connection to this repo

`sae/pipeline.py` is already a thin client over `delphi`: it builds `LatentRecord`s, uses `DefaultExplainer`, `DetectionScorer`, `FuzzingScorer` via an `OpenRouter` client. So most of the paper's machinery is one-import away. Practical things to try as the SAE work in this repo grows up:

1. **Replace the hand-curated `POS`/`NEG` lists with real-corpus stratified sampling.** Right now `sae/pipeline.py:32-52` uses 8 positive and 8 negative sentences chosen by the researcher. That's fine for "does this work end-to-end" but per the paper it will (a) hide the precision/recall tradeoff entirely, and (b) bias every explanation toward the top decile. The natural next step is: run the model over a token stream, collect activating contexts per feature, stratified-sample across deciles, and pass that to `DefaultExplainer`. Delphi has a `LatentDataset` abstraction that does exactly this.

2. **Use detection + fuzzing together, not just both in isolation.** The detection-low / fuzzing-high split flags "right tokens, wrong context" features; detection-high / fuzzing-low flags "right context, wrong tokens". The current `pipeline.py:190` already runs both — just worth surfacing the disagreement explicitly in the summary printout (`pipeline.py:197-205`) as a "disagreement" column. That's a 3-line diff.

3. **Cheap pre-filter with embedding scoring.** If interpretation+score budgets grow, the paper's recommendation is: filter with embedding (≈\$50/100k), then spend dollars on detection+fuzzing for the survivors.

4. **Add intervention scoring for the `steer.py` use case.** The most novel idea here, and the one that fits this repo best because `sae/steer.py` already exists. Concrete plan if you want to chase "output features":
   - Pick prompts $\mathbf x$ from the corpus, sample one clean and one intervened generation per prompt (intervention = clamp the SAE feature direction in the residual stream at a chosen strength).
   - Score `log p(z | intervened) − log p(z | clean)` under the same scorer LLM you use for detection/fuzzing.
   - Calibrate strength by KL: run several clamp magnitudes, pick the one whose `E[D_KL(clean ‖ intervened)]` matches a target — the paper uses fixed-σ comparisons (Fig 4 left panel sweeps three strengths). Without this you cannot meaningfully compare features.
   - This pairs naturally with the steering work: the same intervention parameters serve both "how interpretable is this feature's effect on output" and "is this feature a useful steering knob".

5. **Coverage check.** The paper's "30% of features barely fire on RPJv2-256" result is worth replicating early — it'll tell you what fraction of your SAE's features are even reachable with whatever corpus you choose, and that bounds everything downstream.

## Caveats the paper raises about itself

- Interpretation length is not penalized; longer interpretations probably cheat. They suggest accounting for length in future metrics.
- Non-activating example selection is unprincipled (sampled randomly). Hard negatives might dramatically change all the AUROCs.
- Surprisal scoring underperforms relative to its potential and probably needs prompt-engineering work.
- "Long-range" features (those needing >32 tokens of context to make sense) are not handled.
