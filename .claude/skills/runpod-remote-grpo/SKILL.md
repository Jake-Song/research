---
name: runpod-remote-grpo
description: Use this skill to run the AWM async-GRPO experiment on a remote RunPod GPU pod, driven from the local laptop — e.g. "run the async GRPO experiment on RunPod", "spin up a remote 8-GPU box and launch the training run", "provision a runpod pod and train Qwen3-4B on AWM from my laptop". It provisions the pod with `runpodctl`, bootstraps the 3-process stack over SSH, then hands off to the on-node `async-grpo-experiment` skill to actually run + monitor training. Runs ON the laptop (orchestration); the training runs on the pod.
---

You orchestrate, **from the local laptop**, a remote async-GRPO run on a freshly provisioned RunPod GPU pod. Your job is the *plumbing*: provision the box with `runpodctl`, bring the repo + deps up over SSH, verify topology/P2P, then **hand off to the `async-grpo-experiment` skill running on the pod** — that skill owns the actual training (branch, 3-process stack, two-phase 12→24 run, monitoring, results note). Do not reimplement the training logic here.

Division of labor — be strict about it:
- **You (laptop):** `runpodctl` pod lifecycle, SSH bootstrap, file transfer, teardown.
- **The user:** every secret and the spend decision. Never put the user's secrets in commands you run yourself — have them set the API key, and write run secrets to a `.env` on the pod (Part 4). Provisioning spends money: **confirm before the `pod create` call.**
- **Claude on the pod:** runs `/async-grpo-experiment`.

### Part 0: Preconditions (check fast — stop if any fail)

- **You are on the laptop, not a GPU node.** This skill provisions a *remote* box; it is the counterpart to `async-grpo-experiment` (which runs ON the node). If `nvidia-smi` shows a multi-GPU rig locally, you're probably already on the node — use `async-grpo-experiment` directly instead.
- **`runpodctl` installed + authed.** `runpodctl doctor` must report `api_key: pass`, `api_connectivity: pass`, and `ssh_key` synced. If the API key is unset, STOP and ask the user to run `runpodctl config --apiKey=<KEY>` (or `runpodctl doctor`) themselves — it's their secret, don't handle it.
- **SSH key present.** `~/.ssh/runpod_desktop` (+ `.pub`) exists and `doctor` shows it synced to cloud. If missing, have the user add one (`runpodctl ssh add-key`).
- **Run secrets available.** The run needs the SQL-judge trio (`OPENENV_AWM_LLM_BASE_URL`, `OPENENV_AWM_LLM_API_KEY`, `OPENENV_AWM_LLM_MODEL`), plus `HF_TOKEN` and `WANDB_API_KEY`. You don't need their values now, but confirm the user has them — without them every reward is broken and the run blocks on login prompts.

### Part 1: Resolve the pod config (ask at runtime)

Use `AskUserQuestion` to get two decisions before provisioning:

1. **GPU config** — type + count. Default/recommended **8× A100 80GB** (maps to 1 vLLM + 7 trainers, matches `fsdp2.yaml num_processes: 7`, no edits). Offer 8× H100 (faster but the run is env-server bottlenecked, so limited gain), 4× A100 (cheaper; needs `fsdp2.yaml num_processes=3` + a trimmed `CUDA_VISIBLE_DEVICES`), and 2× (the skill's debug path via `run_trainer_awm.sh`).
2. **Teardown policy** — auto-stop+keep-disk (recommended), auto-delete, or leave running. Apply this in Part 7.

Confirm the GPU `--gpu-id` string matches RunPod's exact naming for the chosen type. State the resolved config back to the user.

### Part 2: Provision the pod ⚠️ spends money — confirm first

Show the exact command and get an explicit go-ahead before running it:

```bash
runpodctl pod create \
  --name awm-grpo-$(date +%Y%m%d) \
  --image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
  --gpu-id "<GPU_ID>" \
  --gpu-count <N> \
  --container-disk-in-gb 60 \
  --volume-in-gb 120 \
  --ports "22/tcp"
```
- Use a CUDA **`devel`** image — `bitsandbytes`/`kernels` build against `nvcc`.
- Optional persistence: if the user has a network volume (created in the RunPod web console — the CLI can't create volumes), add `--network-volume-id <vol_id>` so `/workspace` survives stop/delete.
- Capture the printed pod id as `<POD_ID>`.

### Part 3: Wait for ready + bootstrap over SSH

```bash
runpodctl pod get <POD_ID>     # poll until desiredStatus RUNNING
runpodctl ssh info <POD_ID>    # → host, port, user, keyPath
```
Set `SSH="ssh -i ~/.ssh/runpod_desktop -p <PORT> root@<HOST>"`. Then bootstrap (the `research` repo is **public**, clone directly):

```bash
# clone + provision: startup.sh installs uv + claude and clones trl@AsyncGRPO + OpenEnv as siblings of open-env/
$SSH 'cd /workspace && git clone https://github.com/Jake-Song/research.git && cd research && bash startup.sh'

# python deps — install_packages.sh's `uvx hf auth login` is interactive; pass a token non-interactively instead
$SSH 'cd /workspace/research/open-env && HF_TOKEN=<HF_TOKEN> uvx hf auth login --token "$HF_TOKEN" \
      && uv venv && uv pip install -e ../trl[vllm] && uv pip install -U transformers \
      && uv pip install wandb bitsandbytes kernels==0.14.0 \
      && uv pip install -e ../OpenEnv/envs/agent_world_model_env python-dotenv'
```

### Part 4: Secrets + sanity checks

Have the user supply secret values; write them to the pod's `.env` (project uses `python-dotenv`):
```bash
$SSH 'cat > /workspace/research/open-env/.env' <<EOF
OPENENV_AWM_LLM_BASE_URL=<JUDGE_BASE_URL>
OPENENV_AWM_LLM_API_KEY=<JUDGE_API_KEY>
OPENENV_AWM_LLM_MODEL=<JUDGE_MODEL>
WANDB_API_KEY=<WANDB_KEY>
EOF
```
Verify topology + P2P before handing off (a P2P failure makes the run hang at the first weight sync):
```bash
$SSH 'cd /workspace/research && nvidia-smi --query-gpu=index,name --format=csv,noheader && bash check_gpu_p2p.sh'
```
If you chose **4×**, edit `open-env/configs/fsdp2.yaml` `num_processes` to 3 on the pod; if **2×**, the inner skill uses its 2-GPU debug path. For 8× no edit is needed.

### Part 5: Hand off to Claude on the pod

The intended path is to run the on-node skill, which owns the whole training run:
```bash
$SSH -t 'cd /workspace/research && claude'
# in that session:  /async-grpo-experiment
```
Tell the user this command so they can drive the on-pod Claude (it branches, brings up env-server + vLLM + trainers, runs the two-phase 12→24 steps, monitors, writes the results note). *Alternative:* if the user would rather not run Claude on the pod, you can execute the inner skill's Parts 3–5 directly over `$SSH` yourself — slower to babysit across turns, but valid.

### Part 6: Results back

The inner skill commits a results note on its experiment branch. Pull it to the laptop:
```bash
$SSH 'cd /workspace/research && git push -u origin <exp-branch>'   # public repo, push works
# or croc a single file/checkpoint:
$SSH 'cd /workspace/research && runpodctl send experiment/<run>_results.md'   # prints a code
runpodctl receive <code>                                                      # on the laptop
```

### Part 7: Teardown (apply the Part 1 choice) ⚠️ controls billing

```bash
runpodctl pod stop <POD_ID>     # auto-stop: stops GPU billing, keeps /workspace
runpodctl pod delete <POD_ID>   # auto-delete: terminates; only a network volume survives
# leave-running: do nothing, but remind the user it bills GPU time until they stop it
```

Finish by telling the user: the pod id, the GPU config + cost-relevant state (running/stopped/deleted), the SSH command for Part 5, where the results note lives, and a one-line "what's next."
