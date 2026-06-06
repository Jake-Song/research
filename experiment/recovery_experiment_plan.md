# Experiment Plan: Multi-Turn Agent Recovery in Interactive Environments

## Research Question

Can training language-model agents in multi-turn tool environments improve their ability to recover from errors, misleading observations, failed tool calls, and suboptimal intermediate decisions?

The core claim is not just that environment training improves final task success. The stronger claim is that it improves recovery: after the model enters a bad state, it can diagnose the state, choose corrective actions, and still complete the task.

## Motivation

Most agent evaluations report final success rates, but real agent failures often happen before the final answer: the model calls the wrong tool, omits required arguments, misreads tool output, commits to a mistaken plan, or fails to repair inconsistent state. A model that can recover from these states is more useful than one that succeeds only when its first plan is correct.

Multi-turn environments provide a natural training signal for this behavior because the model receives observations after each action. If recovery can be learned, we should see improvement not only in final reward but also in conditional success after perturbation, error-localization quality, and the number of corrective actions needed after a failure.

## Hypotheses

H1: Multi-turn environment RL improves conditional recovery success compared with supervised or prompt-only baselines.

H2: Recovery improvement is strongest when training includes naturally occurring errors plus controlled perturbations, rather than only clean successful trajectories.

H3: Training on delayed outcome reward alone improves some recovery behavior, but adding recovery-shaped diagnostics improves sample efficiency and reduces repeated error loops.

H4: Recovery behavior transfers across task instances within the same environment family, but cross-environment transfer requires exposure to abstract failure modes such as invalid action, missing precondition, inconsistent memory, and misleading observation.

## Operational Definition of Recovery

A recovery event starts when the agent enters a degraded state. Examples:

- Invalid tool call: wrong tool name, malformed arguments, or missing required fields.
- Failed tool call: server error, timeout, permission error, or empty result.
- Wrong intermediate state: the agent modifies the environment incorrectly or chooses a harmful action.
- Misinterpretation: the agent receives enough evidence to correct itself but initially draws the wrong conclusion.
- Stale-plan state: the current plan is no longer valid because an observation contradicts an assumption.

A successful recovery occurs when the agent later completes the task or reaches a verified acceptable state without external intervention.

## Experimental Setting

Use an agent environment with real multi-turn interaction, tool discovery, tool calls, observations, and delayed verification. The local reference setting is OpenEnv/AWM-style interaction: the model lists tools, calls environment tools over several turns, receives textual observations, and is scored by a verifier after the episode.

The plan should support at least two environment families:

- Structured state environments: games or state-transition tasks where invalid actions and poor intermediate states are easy to identify.
- Tool/database environments: MCP or AWM-style tasks where the agent must inspect tools, query or mutate state, and satisfy a final verifier.

This separation is important because recovery in deterministic games may overfit to state search, while recovery in tool environments tests instruction following, observation grounding, and repair after tool misuse.

## Model Conditions

Compare the following conditions with the same base model:

1. Base model, no environment training.
2. Base model with recovery-oriented system prompt only.
3. SFT on successful clean trajectories.
4. SFT on mixed successful and recovered trajectories.
5. GRPO or equivalent RL with final task reward only.
6. GRPO with final task reward plus recovery-aware auxiliary metrics used for selection or reward shaping.
7. Curriculum GRPO: clean tasks first, then injected perturbations, then adversarial recovery cases.

Use the same inference budget and maximum turn count for all conditions unless the experiment explicitly studies budget sensitivity.

## Recovery Data Construction

Build three datasets.

Clean tasks: ordinary environment tasks with no artificial perturbation.

Natural failure tasks: tasks where the model's own rollout creates the failure state. These are collected by sampling rollouts from weaker checkpoints and labeling whether later actions recover.

Injected failure tasks: tasks where the environment or prompt introduces a controlled degraded state. Examples:

- Hide a required tool until the second `list_tools` call.
- Return an ambiguous or partially truncated observation.
- Insert an irrelevant tool result between useful observations.
- Make the first plausible tool call fail with a recoverable error.
- Start the environment after a prior incorrect action and ask the agent to finish.
- Provide a misleading natural-language hint that conflicts with tool evidence.

The injected failures should be parameterized by failure type, severity, and recoverability. Unrecoverable cases should be excluded from the primary recovery metric but retained for calibration analysis, because a strong agent should also recognize when recovery is impossible.

## Training Design

Use multi-turn rollouts where the model can call tools until it stops or reaches the turn limit. Score the final state with the environment verifier.

For RL, use group-based policy optimization such as GRPO over multiple sampled rollouts for the same task. This is appropriate because recovery is stochastic: the same prompt can produce clean success, failed non-recovery, or successful recovery depending on sampled intermediate actions.

Recommended reward variants:

- Final reward: verifier success only.
- Recovery bonus: positive bonus if success occurs after a detected failure event.
- Loop penalty: penalty for repeated identical failed actions or repeated calls that do not change state.
- Correction bonus: small bonus when the model explicitly changes strategy after contradictory evidence.
- Turn cost: small penalty per tool call to avoid rewarding aimless exploration.

Keep the first training run simple: final reward plus turn cost. Add recovery-specific shaping only after the baseline establishes whether delayed reward alone is enough.

## Evaluation Protocol

Evaluate on held-out tasks and held-out failure injections. Do not evaluate only on final success. Report:

- Overall task success.
- Conditional recovery success: success rate given that a failure event occurred.
- Perturbed-task success: success rate on injected failure starts.
- Clean-task success: ensure recovery training does not harm normal performance.
- Mean turns to recovery: number of actions between failure event and first corrective action or final success.
- Repeated-error rate: fraction of episodes with the same invalid or failed action repeated.
- Premature-stop rate: fraction of episodes where the model stops after a recoverable failure.
- Tool-grounding accuracy: whether tool arguments match observed schema and state.
- Recovery transfer: performance on failure types not seen during training.

For each metric, report confidence intervals across task seeds and bootstrap over task instances, not only over rollouts.

## Key Ablations

Remove multi-turn observations: train or evaluate with only the initial prompt and final reward. This tests whether improvement requires interaction.

Remove failure injection: train only on clean tasks. This tests whether recovery emerges naturally from final reward.

Remove final reward: train only with local recovery-shaped signals. This tests whether local repair behavior is enough for real task completion.

Vary turn budget: evaluate at strict, normal, and generous maximum-turn limits. Good recovery should improve under normal budgets without relying on unlimited retries.

Vary tool-output noise: test whether the model distinguishes useful evidence from misleading or irrelevant observations.

Swap environment family: train on one environment family and evaluate on another to estimate abstraction rather than memorization.

## Analysis Plan

First, estimate whether each trained condition improves final success and conditional recovery success over the base model. The main statistical test should compare paired task-level outcomes across models, because the same task can be evaluated under each condition.

Second, model recovery as a time-to-event problem. The event is successful correction or task completion after a failure. Use survival curves or hazard models to ask whether trained models recover earlier, not just more often.

Third, categorize failures from trajectory logs. Useful categories include invalid action, wrong tool, wrong argument, ignored observation, repeated failed call, premature stop, and harmful state mutation. Report which categories improve and which remain brittle.

Fourth, inspect trajectories qualitatively. A small number of annotated examples should show whether the model is actually diagnosing the problem or merely trying more actions until something works.

## Expected Outcomes

The strongest evidence for the research claim would be:

- Higher conditional recovery success on held-out perturbations.
- No meaningful drop in clean-task success.
- Lower repeated-error and premature-stop rates.
- Faster correction after contradictory observations.
- Some transfer to unseen failure types within the same environment family.

A weaker but still useful result would be final success improvement without clear recovery improvement. That would suggest the model learned better initial policies rather than better repair behavior.

A negative result would be improved training reward but unchanged perturbed-task recovery. That would indicate overfitting to clean trajectories or verifier-specific shortcuts.

## Risks and Controls

Reward hacking: the model may learn to trigger easy verifier states without genuine recovery. Control this with hidden verifier cases and trajectory audits.

Prompt dependence: recovery may come from the system prompt rather than training. Control this with prompt-only baselines.

Turn-budget confounding: models with more retries may look better. Keep budgets fixed and report turn-normalized metrics.

Environment leakage: injected perturbations may appear in training and test with similar wording. Hold out perturbation templates and parameter values.

Judge noise: if an LLM judge scores final states, use repeated judging or deterministic verifiers where possible, and separate judge disagreement from model failure.

## Minimal First Experiment

Start with one environment family and one base model.

1. Collect clean rollouts and natural failure rollouts from the base model.
2. Define four injected failure types: invalid tool argument, failed tool result, misleading observation, and stale-plan state.
3. Train three conditions: prompt-only, SFT on recovered trajectories, and GRPO with final reward plus turn cost.
4. Evaluate on clean held-out tasks and injected held-out failures.
5. Report final success, conditional recovery success, repeated-error rate, premature-stop rate, and turns to recovery.

This first experiment is enough to answer whether multi-turn environment training improves recovery beyond prompting and imitation. Only after that result is clear should the study expand to reward shaping, curricula, and cross-environment transfer.

## Success Criterion

The experiment supports the thesis if the multi-turn trained model improves conditional recovery success by a practically meaningful margin over both base and prompt-only baselines, while preserving clean-task success and reducing repeated-error behavior. The result is strongest if the improvement holds on held-out failure types that were not directly injected during training.
