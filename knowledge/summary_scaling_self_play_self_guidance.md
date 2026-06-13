# Scaling Self-Play with Self-Guidance

Paper: [arXiv:2604.20209](https://arxiv.org/abs/2604.20209)  
Authors: Luke Bailey, Kaiyue Wen, Kefan Dong, Tatsunori Hashimoto, Tengyu Ma  
Method: Self-Guided Self-Play (SGS)

## Core Claim

Asymmetric LLM self-play often stops improving during long training runs because
the task generator learns to exploit its difficulty reward. It produces
artificially difficult, messy, or irrelevant tasks rather than useful stepping
stones toward the target problems.

SGS adds a frozen LLM reviewer, called the Guide, that scores generated problems
for relevance and formulation quality. The generator is rewarded only when a
problem is both:

1. At an informative difficulty level for the current solver.
2. Judged to be a clean and relevant stepping stone toward a specific unsolved
   target.

The paper's second major claim is that solver entropy is part of the self-play
system, not merely a solver-side diagnostic. If the solver becomes
near-deterministic, generated problems have solve rates concentrated near zero
or one. This removes the intermediate-difficulty signal needed to train the
generator.

## Problem Setting

The method starts with a fixed target set:

$$
\mathcal{D} = \{x_1, \ldots, x_N\}.
$$

The goal is to solve as many targets as possible under a large compute budget.
In the experiments, targets are Lean 4 theorem statements and solutions are
formally verified by the Lean compiler.

SGS initializes three roles from the same base model:

- **Solver** $\pi_\theta$: attempts target and synthetic problems.
- **Conjecturer** $g_\phi$: generates a synthetic problem conditioned on an
  unsolved target.
- **Guide** $\rho$: scores whether the synthetic problem is relevant, clean,
  and useful.

The Solver and Conjecturer have separate trainable weights. The Guide is frozen
after a small supervised formatting stage.

## Algorithm

For each self-play round:

1. Sample target problems and separate them into solved and unsolved sets.
2. For every unsolved target $x$, ask the Conjecturer for one simpler, related
   synthetic problem $\tilde{x}$.
3. Generate $k$ Solver attempts for every target and synthetic problem.
4. Verify every attempt with Lean.
5. Update the Solver on verified correct solutions from sufficiently difficult
   problems.
6. Update the Conjecturer using the product of a solve-rate reward and the Guide
   score.

No synthetic problem is generated for a target once that target has been
solved.

### Solver objective

Each proof receives binary verifier reward $v(y) \in \{0,1\}$. The selected
objective, called $\text{REINFORCE}^{1/2}$, applies a length-normalized
REINFORCE update only to problems whose empirical solve rate is at most 0.5.

With binary reward, this is effectively maximum likelihood on verified correct
rollouts, restricted to hard problems. It avoids spending updates on already
easy targets.

### Conjecturer objective

For a synthetic problem, estimate:

$$
s(\tilde{x}) = \frac{1}{k}\sum_{i=1}^{k}v(y_{\tilde{x}}^i).
$$

The solve-rate component is zero when:

- $s(\tilde{x}) = 0$, because the task supplies no positive rollout;
- the task is in the top 30% of solve rates in the batch, because it is too
  easy.

For the remaining bottom 70% of nonzero-solve-rate tasks:

$$
R_{\text{solve}} = 1 - s(\tilde{x}).
$$

The complete generator reward is:

$$
R_{\text{synth}} =
R_{\text{solve}} \cdot R_{\text{guide}}.
$$

Rewards are linearly normalized to $[0,1]$ within the generator batch before a
REINFORCE update.

### Guide rubric

The Guide evaluates:

- relevance to the target theorem, scored 0 to 5;
- redundant premises, scored 0 or 1;
- conclusion complexity, scored 0 to 4.

High-complexity conclusions receive zero reward. Otherwise:

$$
R_{\text{guide}} =
\max(0,\ \text{relevance} + (2-\text{complexity})
+ (1-\text{redundancy})).
$$

This rubric was developed by inspecting generator failures and modifying the
prompt to penalize observed reward-hacking patterns. The Guide model needed SFT
on 2,048 examples to increase correctly formatted outputs from 54.7% to above
99%.

## Experimental Setup

- Base model: DeepSeek-Prover-V2-7B.
- Initial source: 5,000 sampled Goedel-Pset-V1 problems.
- Final target set: 3,323 filtered problems, called $D_{\text{3k}}$.
- Rollouts: 8 proof attempts per problem per round.
- Target batch: all 3,323 problems in every round.
- Synthetic tasks: one per unsolved target.
- Generation temperature: 1.0.
- Maximum sequence length: 8,192.
- Solver and Conjecturer learning rate: $3 \times 10^{-6}$.
- Training used more than 6 billion generated tokens and over 230 passes over
  the target set.

The authors also penalize responses that consume more than 80% of the context
window and reject proofs using a Lean tactic that frequently caused loops.

## Main Results

- SGS has a fitted asymptotic cumulative solve rate seven percentage points
  above the strongest standalone RL baseline.
- It exceeds the fitted asymptotic performance of that baseline in fewer than
  80 self-play rounds.
- At 6.3 million generations, the trained 7B model solves more target problems
  than the reported pass@4 result of DeepSeek-Prover-V2-671B.
- On the 1,346 targets never solved by the RL baseline, SGS solves close to 10%
  after about 8 million generations.
- SGS scales better than the prior Lean-specific STP method, crossing it at
  roughly one million generations.

The paper fits a sigmoid in log compute to cumulative solve rate. The fitted
asymptote changes by a standard deviation of 1.1 percentage points under a
random 50% subsampling sensitivity test. The authors therefore treat
differences below 1.1 points cautiously and do not present fitted asymptotes as
substitutes for observed long-run results.

## Important Ablations

### No target conditioning

Generating arbitrary difficult theorems does not outperform standalone RL. A
task can be well calibrated to the Solver's ability while being irrelevant to
the targets that training is intended to solve.

This shows that difficulty-based curricula alone are insufficient.

### No Guide

Removing the Guide reduces the fitted asymptotic solve rate from 67.1% to 65.5%.
The unguided generator produces more solvable synthetic tasks, but those tasks
transfer less effectively to target problems.

Late in training, the generator exploits the solve-rate reward by creating very
long conclusions with many disjunctions. Nearly all generated conclusions
eventually contain disjunctions, compared with less than 10% in the target
distribution. The Guide prevents this collapse.

### Frozen Conjecturer

A frozen generator initially helps but eventually saturates. As the Solver
learns its fixed synthetic-task distribution, fewer generated tasks remain
useful and the procedure approaches standalone RL.

The generator therefore must adapt, but adaptation requires quality control.

### Group-relative solver RL

CISPO, a grouped RL objective related to GRPO, undergoes rapid entropy collapse.
Solver success rates concentrate at zero and one, so the Conjecturer rarely
observes tasks with an informative intermediate solve rate. SGS with CISPO then
performs approximately like standalone CISPO.

The authors suggest entropy bonuses or KL regularization if grouped objectives
are used in self-play.

## What the Paper Establishes

The strongest contribution is not simply adding an LLM judge. It identifies
three necessary conditions for sustained asymmetric self-play:

1. **Target grounding:** generate curricula in relation to concrete unsolved
   targets.
2. **Generator quality control:** reward usefulness and naturalness separately
   from empirical difficulty.
3. **Solver diversity:** preserve enough policy entropy to produce informative
   difficulty estimates for generator learning.

The Generator and Solver form a coupled learning system. Optimizing either role
in isolation can destroy the other's training signal.

## Limitations

- The main evidence comes from one 7B model family and one formal theorem
  proving domain.
- Lean provides an unusually reliable verifier. In agent environments, a
  generator may need to create the goal, initial state, dynamics, and verifier.
- The Guide rubric was manually developed after inspecting generator failures,
  introducing substantial researcher design effort.
- The Guide is frozen, so its concept of a useful stepping stone cannot evolve
  with the Solver.
- The headline 7B versus 671B comparison is cumulative training-set solve rate
  versus the larger model's pass@4 yardstick, not a general capability or
  compute-matched comparison.
- Training and evaluation focus on a fixed target set. Generalization to
  held-out target distributions is not the paper's central result.
- Fitted asymptotic performance remains an extrapolation, even though the
  sensitivity tests are useful.

## Translation to AWM Agent RL

The closest AWM analogue is not unrestricted task generation. A generated agent
task requires a valid environment state and reliable outcome verifier, which is
much harder than emitting a Lean statement whose verifier already exists.

A conservative adaptation is **target-conditioned trajectory transformation**:

1. Maintain a fixed set of unsolved AWM target tasks.
2. For each target, let the generator create one simpler curriculum instance by
   changing only a validated component:
   - reduce the number of required state changes;
   - expose one otherwise hidden intermediate fact;
   - start from a later valid state snapshot;
   - remove one constraint;
   - inject a single recoverable failure into a successful trajectory.
3. Require the transformed instance to reuse or mechanically derive its final
   verifier from the original target.
4. Run multiple Solver trajectories and estimate empirical completion rate.
5. Score the transformation with a Guide on:
   - relevance to the original target;
   - validity and recoverability;
   - minimality of the modification;
   - absence of verifier leakage;
   - absence of unnecessary tool or state complexity.
6. Train the generator only on tasks with nonzero but non-saturated success and
   a high Guide score.

For recovery training, the generated object should preferably be a validated
degraded-state snapshot rather than an entirely new task. This constrains the
generator's action space and permits deterministic checks:

- continuing the old plan must fail;
- an oracle repair must still reach the original verifier-approved goal;
- the failure must change only its declared state or observation component.

### Implications for async GRPO

The existing AWM setup uses grouped trajectories and discrete verifier rewards,
so the entropy-collapse result is directly relevant. Before training a
generator, monitor:

- mean per-token Solver entropy;
- fraction of task groups with all-identical rewards;
- distribution of per-task completion rates;
- number of generated tasks accepted for Solver training;
- Guide score and transfer to the corresponding original target;
- structural generator drift, such as task length, tool count, and state-edit
  count.

Do not reward an AWM generator solely for producing a roughly 50% completion
rate. That objective can be hacked through ambiguous instructions, unreliable
verifiers, excessive tool complexity, or irrelevant tasks. Difficulty,
validity, target relevance, and formulation quality need separate measurements.

## Recommended First Experiment

Run an offline generator study before jointly training Solver and Generator:

1. Select 100 currently unsolved AWM tasks.
2. Generate four constrained curriculum variants per task with a frozen model.
3. Validate environment reset and verifier behavior mechanically.
4. Collect eight Solver rollouts per variant.
5. Compare three selectors:
   - solve-rate only;
   - target-conditioned generation without a Guide;
   - target-conditioned generation with a Guide.
6. Train identical short Solver runs on the selected tasks.
7. Evaluate improvement on the original 100 targets, not just generated-task
   reward.

This tests the paper's central causal claim in the AWM domain: Guide-selected
synthetic tasks should transfer better to their target tasks even when an
unguided generator produces more solvable training examples.

## Bottom Line

The paper supports target-conditioned challenger-solver self-play, but rejects
the simple objective of generating tasks at the Solver's competence boundary.
Long-running self-play needs an explicit quality signal for generated tasks and
an RL objective that preserves enough Solver diversity to keep curriculum
learning informative.
