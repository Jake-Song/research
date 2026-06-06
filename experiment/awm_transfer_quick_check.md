# Quick Check: Is the Full AWM Transfer Study Worth Implementing?

> Status: proposed pilot. This is a low-cost go/no-go check for the full study in
> `experiment/awm_transfer_experiment_plan.md`, not evidence of general OOD transfer by itself.

## Decision

Should we invest in the full three-seed evaluation across BFCLv3, tau2-bench, and
MCP-Universe?

The quick check looks for two things:

1. A directional transfer signal after short AWM training.
2. Evidence that the trainer and evaluation path are reliable enough to support the full study.

## Minimal Experiment

Use the existing async-GRPO stack without trainer changes:

- **Model:** `Qwen/Qwen3-4B-Instruct-2507`.
- **Training:** one seed, approximately 24 AWM GRPO steps.
- **Checkpoints:** base, step 12, and step 24.
- **Evaluation:** a fixed BFCLv3 pilot set evaluated identically at every checkpoint.
- **Pilot set:** 100 examples, split evenly between state-based `multi_turn_base` cases and
  `irrelevance` cases where calling a tool is inappropriate.
- **Inference parity:** identical prompt, tool schema, parser, decoding parameters, and turn
  budget for all checkpoints.

BFCLv3 is used first because it directly tests function calling, exposes the expected
format-versus-hallucination tradeoff, and is cheaper to integrate than tau2-bench or
MCP-Universe.

## BFCLv3 Baselines

Use two distinct baselines:

1. **Published reference:** the official
   [`Qwen/Qwen3-4B-Instruct-2507` model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
   reports a **61.9 BFCL-v3 aggregate score**. This is a full-benchmark result and is only a
   harness sanity reference; it is not directly comparable to the stratified 100-example pilot.
2. **Experimental control:** evaluate the untrained `Qwen/Qwen3-4B-Instruct-2507` checkpoint on
   the frozen pilot set using the same harness and inference settings as every trained
   checkpoint. This local result is the authoritative baseline for all step-12 and step-24
   deltas.

Do not use the AWM paper's 53.83 base score as this experiment's baseline. That number is for
Qwen3-8B, while this pilot uses Qwen3-4B.

## BFCL Quick-Check Script

`experiment/run_bfcl_quick_check.py` runs the frozen pilot through the official BFCL evaluator.
It pins `bfcl-eval==2026.3.23`, selects 50 `multi_turn_base` and 50 `irrelevance` IDs with seed
`20260606`, and stores independent generation logs, score files, metadata, IDs, and a compact
`summary.json` for each checkpoint.

The maintained evaluator packages these categories under its current BFCL data version. They
originate from the BFCLv3 function-calling and multi-turn evaluation design, but this partial
100-case run is not an official full BFCLv3 leaderboard evaluation and must not be compared
directly with the published 61.9 aggregate.

Serve each checkpoint through vLLM with the stable model alias expected by BFCL:

```bash
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --served-model-name Qwen/Qwen3-4B-Instruct-2507
```

For a trained checkpoint, replace the first argument with its path while keeping
`--served-model-name` unchanged. Evaluate each served checkpoint with a unique run name:

```bash
uv run experiment/run_bfcl_quick_check.py --run-name base
uv run experiment/run_bfcl_quick_check.py --run-name step-12
uv run experiment/run_bfcl_quick_check.py --run-name step-24
```

The default endpoint is `http://127.0.0.1:8000/v1`. Override it with `--base-url`; use
`--api-key` when authentication is enabled and `--tokenizer-path` when the tokenizer is
available at a local checkpoint path.

For Colab, use the standalone script that starts and stops vLLM itself:

```bash
uv run experiment/run_bfcl_quick_check_colab.py --run-name base
```

To evaluate a trained checkpoint, pass its Hugging Face ID or local path with `--model`. Results
are written under `/content/bfcl_quick_check/<run-name>/`.

## Prerequisite Gates

Do not interpret checkpoint differences unless all gates pass.

### 1. Training Stack

- Complete a one-step training smoke test without crashes.
- Confirm that model weights update and a checkpoint can be served.
- Confirm that at least 90% of rollouts receive a verifier result rather than a scoring
  exception or timeout fallback.
- Record wall-clock time per training step and the fraction of failed environment sessions.

### 2. Evaluation Harness

- Run the base model on the complete pilot set before training.
- Manually inspect at least 10 transcripts across both subsets.
- Confirm that valid model tool calls are accepted and malformed calls are scored consistently.
- Confirm that relevance cases permit a correct no-tool response.
- Compare the local base result with the published 61.9 full-benchmark reference. A large
  difference does not automatically fail the gate because the pilot uses a different category
  mix, but it must trigger checks of the prompt, tool-call parser, decoding settings, and sampled
  categories.

If either prerequisite gate fails, classify the result as **Revise**, fix the infrastructure,
and repeat the quick check without drawing a transfer conclusion.

## Metrics

Report the following for the base, midpoint, and final checkpoints:

| Metric | Purpose |
| --- | --- |
| `multi_turn_base` accuracy | Primary stateful transfer signal |
| Relevance/hallucination accuracy | Detects indiscriminate tool use |
| Tool-call format-error rate | Separates formatting gains from task skill |
| Tool-call frequency | Detects a learned bias toward calling tools |
| Mean AWM training reward | Confirms learning on the training environment |
| Verifier failure rate | Measures reward reliability |
| Training time per step | Estimates full-study cost |

For each BFCLv3 metric, report the absolute score and change from base. Because this pilot has
one seed and a small fixed sample, use the results as directional evidence only. Do not claim
statistical significance.

## Decision Rules

Apply these rules after step 24. Do not select the rule retrospectively.

### Go

Proceed with the full transfer study when all of the following hold:

- `multi_turn_base` accuracy improves by at least 5 percentage points over base.
- The midpoint and final checkpoints do not show a clear reversal of the improvement.
- Relevance/hallucination accuracy declines by no more than 3 percentage points.
- Tool-call format errors do not account for the entire `multi_turn_base` gain.
- Training and evaluation prerequisite gates pass.
- Observed runtime fits the available budget for three approximately 96-step seeds.

### Revise

Repeat a targeted pilot before deciding when any of the following holds:

- `multi_turn_base` accuracy improves, but relevance/hallucination accuracy declines by more than
  3 percentage points.
- Results are dominated by tool-call parsing or prompt-format differences.
- Verifier failures exceed 10% of training rollouts.
- Training reward rises while BFCLv3 performance is flat.
- Runtime is unexpectedly high but has an identifiable infrastructure bottleneck.

The revision should address only the observed confound. It should not expand into tau2-bench,
MCP-Universe, multiple seeds, or reward ablations.

### No-Go

Do not implement the full study in its current form when any of the following holds after the
prerequisite gates pass:

- `multi_turn_base` accuracy is flat or lower at both trained checkpoints.
- Any `multi_turn_base` gain is paired with a severe relevance/hallucination regression of at least
  10 percentage points.
- Training reward fails to improve and rollout behavior shows no qualitative learning signal.
- Estimated full-study runtime or verifier cost is prohibitive on the available stack.

A no-go result rejects the current training-and-evaluation proposal, not the general claim that
AWM training can transfer at a different scale or with the paper's synchronous recipe.

## Execution Sequence

1. Select and freeze the 100-example BFCLv3 pilot set.
2. Run and inspect the base-model evaluation.
3. Run a one-step AWM training smoke test and serve its checkpoint.
4. Train one seed to approximately 24 steps, saving steps 12 and 24.
5. Evaluate both checkpoints on the frozen pilot set.
6. Compare metrics and inspect all changed outcomes plus at least 10 unchanged failures.
7. Estimate full-study compute from measured training and evaluation runtimes.
8. Record **Go**, **Revise**, or **No-Go** using the predefined rules.

## Results Template

**Published full BFCL-v3 reference:** 61.9 for `Qwen/Qwen3-4B-Instruct-2507`.

| Checkpoint | AWM reward | Multi-turn accuracy | Irrelevance accuracy | Format errors | Tool-call rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | N/A |  |  |  |  |
| Step 12 |  |  |  |  |  |
| Step 24 |  |  |  |  |  |

- **Training time per step:**
- **Verifier failure rate:**
- **Estimated cost of three 96-step seeds:**
- **Observed confounds:**
- **Decision:** Go / Revise / No-Go
- **Reason:**

## Interpretation Limits

This quick check cannot establish:

- statistically reliable transfer across training seeds;
- transfer beyond function calling;
- monotonic improvement over a full training schedule;
- performance on conversational tasks or real MCP servers; or
- equivalence to the AWM paper's training scale and synchronous GRPO setup.

Those claims require the full experiment. The pilot exists only to determine whether building
and running that experiment is justified.
