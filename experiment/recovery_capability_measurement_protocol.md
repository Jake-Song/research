# Measuring Recovery Capability in Tool-Using Language-Model Agents

## Study Status

Draft research protocol for preregistration and pilot validation.

## Abstract

This study measures whether language-model agents can recover after entering a
verified degraded state during a multi-turn tool-use task. Recovery is distinct
from initial competence: an agent may avoid errors because it has a strong
initial policy, or it may recover well after an error despite making errors
frequently. Conventional task-success metrics conflate these behaviors.

We propose a paired evaluation in which open-weight instruction models complete
matched clean and perturbed versions of held-out tasks. Perturbations create
controlled, recoverable failures such as invalid tool arguments, transient tool
errors, missing preconditions, reversible state mutations, misleading
observations, and stale plans. Every perturbed state must have an executable
oracle repair and a deterministic final-state verifier. The primary endpoint is
verified completion from the degraded state under a fixed interaction budget.
Secondary endpoints measure recovery cost, correction latency, repeated errors,
premature abandonment, diagnostic accuracy, and calibration on unrecoverable
states.

The design combines controlled perturbations, which permit matched causal
comparisons, with a separate bank of naturally occurring failures, which
supports ecological-validity analysis. Hierarchical models and task-clustered
bootstrap intervals quantify differences across model scale, environment,
failure class, and severity.

## 1. Research Question

How reliably and efficiently can a tool-using language-model agent restore
progress and complete a task after it encounters a recoverable degraded state?

The study addresses four subordinate questions:

1. Does recovery capability increase with model scale?
2. Which failure classes are most difficult to recover from?
3. Does measured recovery transfer across environment families?
4. Do controlled perturbation results predict recovery from naturally occurring
   failures?

This is a capability-measurement study. It does not test a training
intervention, recovery-specific reward shaping, or fine-tuning method.

## 2. Construct Definition

### 2.1 Recovery opportunity

A recovery opportunity begins when the agent receives an observation from, or
is placed in, a degraded state that:

- prevents the current plan from succeeding unchanged;
- remains solvable under the available tools and interaction budget;
- contains enough accessible evidence for a competent agent to determine a
  corrective action; and
- has at least one executable repair trajectory that reaches a verifier-approved
  final state.

### 2.2 Successful recovery

A successful recovery occurs when the agent reaches the original
verifier-approved task goal from the degraded state without external
intervention and within the predefined action and token budgets.

### 2.3 Related but distinct constructs

- **Initial competence:** probability of completing a clean task without a
  perturbation.
- **Robustness:** probability that a perturbation does not disrupt the agent's
  current policy.
- **Recovery:** probability of success after the perturbation has created a
  consequential degraded state.
- **Exploration:** trying alternative actions without necessarily diagnosing or
  repairing the failure.
- **Calibration:** recognizing when recovery is impossible rather than
  repeatedly acting or falsely claiming success.

The benchmark must not count an inconsequential perturbation as recovery. A
perturbation is consequential only if replaying the pre-perturbation plan no
longer succeeds or an explicit corrective action is required.

## 3. Hypotheses

### Confirmatory hypotheses

**H1: Scale effect.** Larger open-weight instruction models have higher
conditional recovery success than smaller models under matched degraded states.

**H2: Recovery cost.** Models with higher recovery success require fewer actions
and tokens after the perturbation, conditional on task and failure class.

**H3: Failure-class heterogeneity.** Recovery performance differs across failure
classes, with state-changing failures producing a larger recovery penalty than
transient execution failures.

**H4: Cross-environment consistency.** Model rankings are positively associated
across the two environment families, after adjusting for clean-task competence.

### Exploratory hypotheses

- Recovery on controlled perturbations predicts recovery on replayed natural
  failures.
- Stronger models are better calibrated on unrecoverable states.
- Explicitly correct diagnosis predicts later task recovery, but the strength
  of this relationship varies by failure class.

## 4. Experimental Setting

### 4.1 Model panel

Evaluate three open-weight instruction-tuned models at approximately 2B, 4B,
and 8B parameters. Prefer checkpoints from one model family when suitable
checkpoints exist, because this reduces tokenizer, architecture, and
post-training confounds. The exact immutable model revision, tokenizer revision,
chat template, and serving version must be recorded before evaluation.

All models use:

- the same system prompt and tool descriptions;
- the same decoding policy;
- the same maximum tool-action and token budgets;
- the same model-visible transcript format; and
- the same model-serving interface.

If context limits differ, use the largest common limit that accommodates all
included tasks.

### 4.2 Environment families

Use two structurally different tool environments:

1. **Structured database/tool environment:** AWM-style tasks requiring schema
   discovery, queries, state mutations, and deterministic final-state
   verification.
2. **Conversational service environment:** multi-turn tasks such as
   tau2-bench, in which tools modify service state while an external user or
   simulator provides additional observations.

The environments must differ in interaction structure and domain semantics.
This prevents the study from equating recovery with memorization of one tool
grammar or database pattern.

### 4.3 Task sample

Construct a minimum of 300 held-out base tasks, balanced as closely as practical
across environment family and predefined difficulty strata. No task, template,
tool schema, or perturbation realization used during benchmark development may
appear in the final evaluation set.

Each task must have:

- a stable task identifier;
- a deterministic initial-state snapshot;
- an executable clean reference trajectory;
- a deterministic or programmatically checkable final-state verifier;
- a declared action and token budget; and
- one or more validated perturbation points.

Tasks that cannot be restored deterministically or verified reliably are
excluded before model evaluation.

## 5. Failure Taxonomy and Perturbations

The primary benchmark contains six failure classes.

| Failure class | Controlled perturbation | Required corrective behavior |
| --- | --- | --- |
| Invalid tool arguments | Reject a plausible call because an argument is malformed, missing, or inconsistent with the schema | Inspect the error or schema and issue a corrected call |
| Transient tool failure | Return a temporary timeout or server failure for an otherwise valid call | Retry appropriately or use an equivalent route |
| Missing precondition | Place the task in a state where a required prerequisite has not been satisfied | Identify and satisfy the prerequisite before continuing |
| Reversible harmful mutation | Begin after an incorrect but reversible state-changing action | Inspect state, undo or compensate for the mutation, and continue |
| Misleading observation | Supply an incomplete or misleading observation that conflicts with recoverable tool evidence | Seek disambiguating evidence and revise the inferred state |
| Stale plan | Change relevant state after the plan was formed so that the planned next action is no longer valid | Detect the contradiction and replan |

### 5.1 Severity

Each perturbation is assigned one of three preregistered severity levels:

- **Low:** one direct corrective action restores the clean trajectory.
- **Medium:** repair requires two or three causally related actions.
- **High:** repair requires replanning or a compensating state mutation, but
  remains feasible within the common budget.

Severity is determined by the shortest validated oracle repair, not by a
researcher's subjective judgment.

### 5.2 Perturbation validation

Before inclusion, each perturbation must pass four checks:

1. Replaying the pre-perturbation plan without correction fails.
2. An executable oracle repair reaches the original verified goal.
3. The perturbation changes only its declared state or observation component.
4. Independent reviewers agree that the model has access to enough information
   to discover the repair.

Perturbations that make the task irreversible, remove necessary information, or
change the original goal are excluded.

## 6. Experimental Procedure

### 6.1 Paired clean and degraded episodes

For each model-task pair, run:

- a clean episode from the original initial state; and
- one or more degraded episodes from validated perturbation states.

Clean and degraded episodes use the same task goal and final-state verifier.
Execution order is randomized within blocks defined by model, environment, and
failure class. The model is not told the perturbation label or oracle repair.

Use five fixed stochastic seeds per model-condition pair. Seeds are shared
across models where the serving stack permits deterministic seed control.
Repeated samples characterize model stochasticity but do not increase the
number of independent task units.

### 6.2 Interaction budget

Set a common maximum number of tool actions and generated tokens using the
pilot-task distribution. The budget must permit every validated oracle repair
with a preregistered margin while preventing unlimited brute-force retries.

The primary analysis uses the common fixed budget. Budget-sensitivity analyses
at 0.75x and 1.5x the primary action budget are secondary.

### 6.3 Natural failure bank

Natural failures are analyzed separately from the controlled benchmark:

1. Run each model on clean development tasks.
2. Detect candidate failure points using environment errors, verifier state,
   and blinded trajectory review.
3. Save replayable snapshots immediately after each consequential error.
4. Deduplicate snapshots by task state and failure mechanism.
5. Pool failures across source models.
6. Evaluate every model from the same snapshot bank without revealing which
   model produced the failure.

This pooled design avoids evaluating each model only on its own selected
failures. Results from natural failures are evidence of ecological validity, not
the primary causal comparison.

### 6.4 Unrecoverable controls

Create a secondary set of states in which an essential resource is irreversibly
lost or a necessary tool is unavailable. The model should identify that the
original goal cannot be completed and stop without falsely claiming success.
These states are never included in the primary recovery-success denominator.

## 7. Outcome Measures

### 7.1 Primary endpoint

**Conditional recovery success (CRS):**

\[
\mathrm{CRS}_{m,f} =
\Pr(\text{verified task success} \mid
\text{validated degraded state}, m, f)
\]

where \(m\) identifies the model and \(f\) the failure class.

An episode counts as successful only when the environment verifier confirms the
original goal. Self-reported success is insufficient.

### 7.2 Principal secondary endpoints

**Recovery penalty:** paired difference in completion probability between clean
and degraded versions of the same task.

\[
\Delta_{\mathrm{recovery}} =
\Pr(\text{success}_{clean}) -
\Pr(\text{success}_{degraded})
\]

**Recovery efficiency:** number of valid tool actions and generated tokens from
the perturbation until verified repair or task completion.

**Correction latency:** number of turns from perturbation exposure to the first
action that moves the environment onto an oracle-valid repair path.

**Repeated-error rate:** fraction of degraded episodes in which the agent
repeats the same failed action without receiving relevant new evidence.

**Premature-abandonment rate:** fraction of recoverable episodes in which the
agent stops while a validated repair remains feasible.

**Diagnostic accuracy:** whether the first post-failure strategy correctly
identifies the violated precondition, invalid action, or contradictory evidence.
Diagnosis may be inferred from the selected action; private chain-of-thought is
not required or scored.

**Calibration on unrecoverable states:** balanced accuracy and false-abandonment
rate when distinguishing recoverable from unrecoverable cases.

### 7.3 Natural-failure reporting

For natural failures, report both:

- failure incidence on clean-start episodes; and
- probability of eventual verified success after a consequential failure.

Conditional recovery must not be reported alone, because a model with few but
severe failures can otherwise appear worse than a model that fails often on
easy-to-repair states.

## 8. Statistical Analysis

### 8.1 Primary model

Fit a hierarchical logistic regression for episode-level verified recovery:

\[
\operatorname{logit} \Pr(Y_{ijr}=1) =
\beta_0 + \beta_{\mathrm{model}} +
\beta_{\mathrm{failure}} + \beta_{\mathrm{severity}} +
\beta_{\mathrm{environment}} +
\beta_{\mathrm{model}\times\mathrm{failure}} +
u_{\mathrm{task}_i} + u_{\mathrm{template}_j}
\]

where \(Y_{ijr}\) is recovery success for task \(i\), perturbation template
\(j\), and rollout seed \(r\). Include random intercepts for task and
perturbation template. Add a model-by-environment interaction for the
cross-environment hypothesis.

Report posterior or model-based marginal recovery probabilities, pairwise
contrasts, and 95% uncertainty intervals. The inferential framework must be
fixed during preregistration; Bayesian and frequentist results must not be
selected post hoc based on significance.

### 8.2 Paired contrasts

Estimate clean-to-degraded recovery penalties within task. Use a
task-clustered bootstrap with at least 10,000 resamples for confidence intervals.
Bootstrap tasks, retaining all associated perturbations and rollout seeds
within each sampled cluster.

### 8.3 Time-to-recovery

Analyze correction latency using a discrete-time survival model. Verified
correction is the target event. Premature abandonment and budget exhaustion are
competing outcomes rather than successful censoring.

### 8.4 Multiplicity

The three model-scale pairwise comparisons for the primary endpoint are
confirmatory and use Holm correction. Failure-class and environment interaction
analyses are reported with full uncertainty intervals and clearly labeled as
confirmatory or exploratory according to the preregistration.

### 8.5 Missingness and exclusions

Infrastructure failures that occur before a model receives the task are rerun
and logged. Model-caused malformed calls, timeouts, and context exhaustion are
outcomes, not missing data. Post-execution exclusions are permitted only for
verifier corruption or unrecoverable environment faults and must be reported by
model and condition.

## 9. Pilot and Power Analysis

Run a blinded pilot on 50 tasks not used in the final evaluation. The pilot
estimates:

- baseline recovery probabilities;
- between-task and perturbation-template variance;
- within-task correlation across models;
- rollout-level stochastic variance;
- oracle-repair length; and
- annotation prevalence and agreement.

Use simulation from the fitted pilot variance structure to choose the final
number of tasks and perturbations. The design must achieve at least 90% power
for a five-percentage-point absolute difference in recovery success between
adjacent model sizes at a family-wise alpha of 0.05.

The final benchmark must contain at least 300 independent tasks. Additional
rollout seeds cannot substitute for insufficient task diversity.

## 10. Human Annotation

Human annotation is required for correction latency, diagnostic accuracy, and
qualitative failure analysis.

- Two annotators independently label at least 20% of trajectories.
- Annotators are blinded to model identity and checkpoint size.
- The coding manual includes positive and negative examples for each failure
  class and corrective-action label.
- Require Krippendorff's alpha of at least 0.80 for confirmatory labels.
- If agreement is lower, revise the manual, retrain annotators, and relabel the
  affected sample before analysis.
- Resolve remaining disagreements through blinded adjudication.

Automated labels derived from state transitions may replace human labels only
after validation against the double-annotated subset.

## 11. Validity Threats and Controls

### Construct validity

**Threat:** the model succeeds by blind exploration rather than recovery.

**Control:** report correction latency, repeated actions, path efficiency, and
diagnostic accuracy in addition to final success.

**Threat:** the perturbation is too weak to require correction.

**Control:** require unchanged-plan replay to fail before including the case.

### Internal validity

**Threat:** models receive different effective prompts, schemas, or budgets.

**Control:** use one adapter, one transcript format, and common budgets.

**Threat:** perturbation templates leak through recognizable wording.

**Control:** hold out templates and parameterizations, and hide perturbation
labels.

### External validity

**Threat:** controlled failures do not represent real agent failures.

**Control:** compare controlled results with a pooled natural-failure snapshot
bank and use two environment families.

### Statistical validity

**Threat:** repeated rollouts are treated as independent evidence.

**Control:** define the task as the independent sampling unit and cluster all
resampling and uncertainty estimates accordingly.

### Evaluation validity

**Threat:** verifier exploitation produces apparent recovery.

**Control:** use hidden verifier checks, state-difference audits, and manual
inspection of a stratified sample of successes.

## 12. Reproducibility and Data Schema

Each episode must be exported as a JSONL record containing:

- protocol and benchmark version;
- model, checkpoint, tokenizer, and serving revisions;
- environment, task, perturbation-template, failure-class, and severity IDs;
- clean, degraded, natural-failure, or unrecoverable condition;
- initial-state and degraded-state snapshot hashes;
- recoverability-validation result and oracle-repair length;
- random seed and decoding parameters;
- full timestamped action-observation trajectory;
- action count, generated-token count, and termination reason;
- final verifier result and verifier version;
- first corrective-action turn and recovery turn;
- repeated-error and premature-abandonment labels;
- human annotation status and adjudication result; and
- infrastructure-error and exclusion fields.

Release the perturbation generators, task identifiers, seeds, prompts,
transcripts, exclusion log, metric code, and analysis scripts. Any data that
cannot be released must be described with a deterministic reconstruction
procedure.

## 13. Implementation Requirements

Extend the existing deterministic tool harness with the minimum interfaces
needed for this protocol:

- deterministic state snapshot and restoration;
- perturbation injection by task, failure class, severity, and seed;
- oracle-repair execution and recoverability validation;
- a common vLLM-compatible model adapter;
- bounded clean and degraded episode execution;
- JSONL trajectory export; and
- deterministic metric extraction.

The benchmark must remain independent of a specific trainer. No training code,
reward shaping, or fine-tuning infrastructure is part of this study.

## 14. Verification Tests

Before collecting final results, verify that:

1. Restoring a snapshot reproduces the exact state and verifier output.
2. Repeating a perturbation with the same seed produces the same degraded state.
3. Each perturbation changes only its declared component.
4. Every included oracle repair succeeds within the primary budget.
5. Replaying the unchanged plan fails for every included degraded state.
6. Clean and degraded episodes share the same goal and final verifier.
7. Fixed inference seeds reproduce outputs within the serving stack's documented
   determinism limits.
8. Metric extraction handles immediate recovery, repeated errors, abandonment,
   timeout, and unrecoverable states correctly.
9. A scripted repair policy outperforms scripted repetition and immediate-stop
   baselines.
10. Hidden verifier cases detect deliberately constructed shortcut solutions.

## 15. Decision Criteria

The study supports the claim that a model has stronger recovery capability when
it demonstrates:

- higher conditional recovery success under matched degraded states;
- a smaller clean-to-degraded recovery penalty;
- equal or lower correction latency and recovery cost;
- fewer repeated errors and premature terminations; and
- consistent improvements across failure classes and both environment families.

Higher degraded-task success alone is insufficient if it results from a larger
interaction budget, lower clean-task competence, verifier exploitation, or
unstructured repeated trial-and-error.

## 16. Planned Outputs

The final report will contain:

- a preregistered primary-results table with recovery probabilities and paired
  contrasts;
- clean-versus-degraded performance by model;
- recovery success and efficiency by failure class and severity;
- cross-environment model rankings;
- time-to-recovery curves;
- natural-failure incidence and rescue rates;
- unrecoverable-state calibration results;
- annotation agreement and exclusion accounting; and
- a stratified set of blinded qualitative trajectory analyses.

