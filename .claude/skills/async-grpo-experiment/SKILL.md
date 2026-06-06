---
name: async-grpo-experiment
description: Use this skill to auto-run an async-GRPO training experiment on the provisioned GPU cloud node — e.g. "run an AWM async GRPO experiment", "launch a training run on the 8-GPU rig", "train Qwen3-4B on AWM and report". It makes a fresh git branch for the run, brings up the 3-process stack (AWM env server + vLLM + FSDP2 trainers), smoke-tests it, runs a fixed 10-step training run, monitors reward/logs, and writes a results note. Runs ON the GPU node, not the laptop.
---

You autonomously run one async-GRPO training experiment **on the provisioned GPU cloud node** and report on it. The stack is three cooperating processes with NCCL weight transfer:

1. **AWM env server** (CPU, port 8899) — serves the multi-turn MCP environment + SQL/code verifier.
2. **vLLM server** (GPU 0, port 8000) — generates rollouts, receives trainer weights over NCCL.
3. **FSDP2 trainers** (GPUs 1–N) — `AsyncGRPOTrainer`; rank 0 drives the rollout worker.

The single object of study is the training run. **Do not** rewrite the trainer or build an eval harness — reuse the existing files. Your job is: branch, bring the stack up in order, verify it, run **exactly 10 steps**, babysit, and write up what happened.

Two standing rules for this rig:
- **Every experiment runs on its own git branch** (never on `main`). One branch per run.
- **Every experiment trains for exactly 10 optimizer steps** (`max_steps=10`). Fast-iteration default; do not run longer unless the user explicitly overrides.

Reused as-is (don't rewrite the training logic):
- Trainer: `open-env/openenv_awm_async_grpo.py`
- Multi-GPU accelerate config: `open-env/configs/fsdp2.yaml` (`num_processes: 7`)
- Single-GPU debug config: `open-env/configs/accelerate_fp8_single_gpu.yaml`
- vLLM launcher: `open-env/scripts/run_vllm_awm.sh` (GPU 0, NCCL weight transfer)
- Single-GPU trainer launcher: `open-env/scripts/run_trainer_awm.sh`

### Part 0: Preconditions (check once, fast — stop if any fail)

- **You are on the GPU node.** Run `nvidia-smi --query-gpu=index,name --format=csv,noheader`. If `nvidia-smi` is missing or shows 0 GPUs, STOP — this skill must run on the provisioned cloud box (where `startup.sh` installed claude), not the laptop. Tell the user to invoke it there.
- **GPU count → topology.** GPU 0 is always vLLM. The rest are trainers. 8 GPUs → 1 vLLM + 7 trainers (`fsdp2.yaml`). If there are exactly 2 GPUs, use the single-GPU trainer path. Set the trainer `num_processes` to `GPU count − 1`; edit `fsdp2.yaml`'s `num_processes` only if the count differs from 7.
- **Repo provisioned.** Confirm sibling clones exist: `../trl` (on the `AsyncGRPO` branch) and `../OpenEnv`. If missing, run `bash startup.sh` then `bash open-env/scripts/install_packages.sh` (from `open-env/`). Skip reinstall if `uv run python -c "import trl.experimental.async_grpo"` already succeeds.
- **P2P / NCCL.** Run `bash check_gpu_p2p.sh`. Weight transfer from trainers → vLLM rides NCCL P2P; if the functional P2P test fails, the run hangs at the first weight sync. Surface this before launching rather than letting it hang.
- **Verifier LLM judge env vars.** The SQL verifier needs an external judge: `OPENENV_AWM_LLM_BASE_URL`, `OPENENV_AWM_LLM_API_KEY`, `OPENENV_AWM_LLM_MODEL`. Confirm all three are set. If unset, ask the user — don't launch without them or every reward is broken.
- **Logging/hub auth.** Push-to-hub defaults **on**: ensure `hf auth whoami` works, or pass `--no-push-to-hub`. For wandb, ensure it's logged in (or `WANDB_API_KEY` is set); otherwise export `WANDB_MODE=offline` so the run doesn't block on a prompt.

### Part 1: Create the experiment branch

From a clean tree on `main`, create and check out a fresh branch for this run — `git checkout -b exp/<run-name>` (e.g. `exp/awm-lr2e6-20260607`). Pick a short, unique, descriptive name from the config you're about to run. Everything below (any config edits, the results note) is committed on this branch, never on `main`. If the tree is dirty, ask before branching.

### Part 2: Resolve the experiment

- Default target is the **AWM** async-GRPO run (`openenv_awm_async_grpo.py`). Only deviate if the user names a different `openenv_*_async_grpo.py` script.
- **Step count is fixed at 10.** Launch with `max_steps=10`. The trainer has no `--max-steps` flag yet — add one: an argparse line (`--max-steps`, type int, default 10) and pass `max_steps=args.max_steps` into the `AsyncGRPOConfig(...)`. This is a small, necessary wiring change and it lives on the experiment branch. Do not instead try to back into 10 steps via `dataset_size`/epochs — `max_steps` is exact.
- Other knobs are CLI flags (`--dataset-size`, `--num-generations`, `--learning-rate`, `--gradient-accumulation-steps`, `--max-completion-length`, `--save-steps`, `--wandb-name`, `--output-dir`, `--no-push-to-hub`, …). Use the user's specified knobs; otherwise the defaults from `experiment/awm_transfer_experiment_plan.md` (Qwen3-4B, `num_generations=8`, `max_completion_length=1024`, `grad_accum=16`, `lr=1e-6`, `dataset_size=1000`). Set `--save-steps` so a checkpoint lands within 10 steps (e.g. `--save-steps 5`). State the final config back to the user.
- Use a unique `--wandb-name` / `--output-dir` (match the branch name) so you don't clobber a prior run.

### Part 3: Bring up the stack (tmux, ordered)

Use a tmux session (`tmux new-session -d -s grpo`) with one window per process so they survive turn boundaries and you can tail each independently. Each step must be healthy before the next:

1. **Env server** (from `OpenEnv/`):
   `PYTHONPATH=src:envs uv run uvicorn envs.agent_world_model_env.server.app:app --host 0.0.0.0 --port 8899`
   Health gate: `curl -fs localhost:8899` returns without connection-refused.
2. **vLLM** (GPU 0): `bash open-env/scripts/run_vllm_awm.sh`.
   Health gate: poll `curl -fs http://localhost:8000/health` until 200 (weight load takes minutes). Do **not** start the trainer before vLLM is serving.
3. **Trainers** (GPUs 1–N), from repo root:
   - Multi-GPU: `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 uv run accelerate launch --config_file open-env/configs/fsdp2.yaml open-env/openenv_awm_async_grpo.py --env-url http://localhost:8899 --max-steps 10 [flags]`
   - 2-GPU debug: `bash open-env/scripts/run_trainer_awm.sh --env-url http://localhost:8899 --max-steps 10 [flags]`
   (Adjust the `CUDA_VISIBLE_DEVICES` list and `fsdp2.yaml num_processes` to the actual trainer count.)

### Part 4: Smoke gate (first step or two)

A 10-step run is already smoke-scale, so don't do a separate throwaway run — just watch the first 1–2 steps of the real run and confirm: env reachable, vLLM generates, the **first NCCL weight sync completes** (doesn't hang), and a **non-degenerate reward** appears (not all `format_error`/−1.0). If it hangs at weight sync, kill it and revisit P2P (Part 0). Once the first steps look sane, let it run to 10.

### Part 5: Run + monitor the 10 steps

- Tail the trainer tmux window and watch wandb (`openenv-awm` project): **mean reward**, the reward-by-label fractions (Completed / Partial / Failed / format-error), and **mean turns per rollout**.
- Throughput is **env-server bottlenecked**, not GPU — rollouts hit the AWM env synchronously over multiple turns, so steps are slow and low GPU util is expected, not a bug.
- Known failure modes and the fix:
  - **Scoring timeouts** killing the rollout worker (the AWM SQL judge can be slow; `message_timeout_s` is already bumped to 300 — if workers crash on `TimeoutError`, that's the cause).
  - **NCCL hang** at a weight-sync step → P2P/topology issue.
  - **OOM** on a trainer → lower `max_completion_length` or `per_device_batch_size`.
  - **Degenerate reward** (all format-error / stuck) → check env-server logs and the judge env vars.
  If a process dies, capture the tail of its tmux pane, diagnose, and either fix-and-relaunch or report the blocker. Don't silently restart in a loop.
- Stop at step 10 (the `max_steps` cap handles this).

### Part 6: Report

Write a markdown results note to `experiment/<run-name>_results.md` (style of the existing `experiment/*.md`), then commit it on the experiment branch:
- **Config**: exact flags (incl. `max_steps=10`), GPU topology (vLLM + N trainers), model, dataset size.
- **Smoke gate**: pass/fail + what the first steps confirmed.
- **Training summary**: final & best mean reward over the 10 steps, the reward-by-label trend, mean turns, and the curve shape (rising / flat / collapsed). With only 10 steps, frame this as a sanity/early-signal read, not a converged result.
- **Artifacts**: wandb run URL, checkpoint `output_dir` (and hub repo if pushed), the branch name.
- **Anomalies**: any crashes, restarts, timeouts, P2P issues — with the cause.
- **Read** (1–3 sentences): did the stack train cleanly for 10 steps, and is it worth a longer run / eval per `experiment/awm_transfer_experiment_plan.md`?

Finish with `git add experiment/<run-name>_results.md && git commit` on the branch, then tell the user the branch name, where the note and checkpoints are, and the one-line verdict.
