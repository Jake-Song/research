---
name: hf-jobs
description: Use this skill to quickly run or iterate on a one-off compute experiment on Hugging Face Jobs (the `hf jobs` CLI) — e.g. "run this on an A10G", "try this on HF jobs", "launch a GPU job to test X". Centers on the fast `hf jobs uv run` loop with a self-contained PEP 723 script.
---

You help the user run a quick, throwaway compute experiment on **Hugging Face Jobs** — managed containers (CPU/GPU) billed per-minute. The fast loop is: write ONE self-contained Python script with inline deps, launch it with `hf jobs uv run`, stream logs, iterate. Do not build Docker images or set up multi-file projects unless the user asks.

The `hf` CLI is already installed (`hf jobs --help` to confirm). Jobs run on HF's infra, NOT locally and NOT in Colab.

### Part 0: Preconditions (check once, fast)

- Auth: `hf auth whoami`. If it errors, tell the user to run `hf auth login` (or `! hf auth login`) — a token with **write/inference** scope is needed to launch jobs. Don't proceed until authed.
- Cost awareness: GPU flavors bill per minute (see table in Part 2). Always pass `--timeout` so a hung job can't bill indefinitely. Default to the cheapest flavor that can plausibly run the experiment; only scale up when the user asks or the small one OOMs.

### Part 1: Write the experiment script

Default to a single PEP 723 uv script (inline `# /// script` dependency block) so `uv` resolves deps on the worker — no image building, fastest iteration.

- If the user already has a target `.py`, use it; add a `# /// script` block if missing.
- Otherwise copy `templates/job.py` to a sensible path (e.g. `./<name>/job.py` or `./hf_job.py`) and adapt it to the task. The template is a minimal transformers generation that first prints `torch.cuda` device info so a GPU run is verifiable.
- Keep the script self-contained: all `import`s covered by the inline deps, all inputs via `argparse` or env vars, all results printed to stdout or saved to a mounted volume (Part 4). The worker filesystem is ephemeral — anything not printed or uploaded is lost when the job ends.

### Part 2: Pick a flavor

`hf jobs hardware` lists live options + price. Common picks (cost/hr):

| Flavor        | Accelerator     | $/hr  | Use for |
| ------------- | --------------- | ----- | ------- |
| `cpu-basic`   | — (2 vCPU/16GB)  | $0.01 | smoke-test the script logic, no model |
| `t4-small`    | 1× T4 16GB       | $0.40 | small models, cheapest GPU |
| `l4x1`        | 1× L4 24GB       | $0.80 | ~7B inference, good default GPU |
| `a10g-small`  | 1× A10G 24GB     | $1.00 | ~7B inference/light finetune |
| `a100-large`  | 1× A100 80GB     | $2.50 | large models / training |
| `h200`        | 1× H200 141GB    | $5.00 | very large models |

Tip: validate the script end-to-end on `cpu-basic` (with a tiny model or a `--smoke` flag) before spending GPU minutes.

### Part 3: Launch

Use detached mode so you get the job id back immediately, then stream logs:

```sh
hf jobs uv run --flavor l4x1 --timeout 30m --secrets HF_TOKEN -d job.py --model Qwen/Qwen3-1.7B
```

- `--secrets HF_TOKEN` forwards the user's token into the job (needed for gated/private models or pushing results). Add `--secrets HF_TOKEN` whenever the script touches the Hub.
- `--with <pkg>` installs extra packages not in the inline block (one `--with` per package).
- `--env KEY=value` for non-secret config; `--image` only if a custom base image with `uv` is required.
- For a plain (non-uv) command in an existing image, use `hf jobs run <image> <cmd...>` with the same flags instead.

`-d` prints the job id. Then:

```sh
hf jobs logs -f <job_id>      # stream until the job finishes
hf jobs ps -a                 # list jobs (running + recent)
hf jobs inspect <job_id>      # status, flavor, timing
hf jobs cancel <job_id>       # stop a running job
```

Run the launch yourself if the user has authed and approved the spend; otherwise print the exact command for them to run. Always surface the job id and the `logs -f` command.

### Part 4: Persisting outputs (only if results must survive)

The job filesystem is wiped on exit. To keep artifacts, mount a volume with `-v`:

- `-v hf://datasets/<user>/<ds>:/data` — read a dataset (read-only).
- `-v hf://buckets/<user>/<bucket>:/out` — read+write scratch bucket; write results to `/out`.

Or have the script `push_to_hub` / `upload_file` to a repo (needs `--secrets HF_TOKEN`). For a quick experiment, printing results to stdout and reading them from `hf jobs logs` is usually enough — don't add a volume unless outputs are large or must persist.

### Part 5: Iterate

Edit the local script and re-run the same `hf jobs uv run` command — no rebuild step. Tighten the loop: smaller model + `cpu-basic` while debugging logic, then one final run on the real flavor. Remind the user to `hf jobs cancel` any job they abandon so it stops billing.

### Notes

- `--timeout` accepts `s`/`m`/`h`/`d` suffixes (e.g. `30m`, `2h`). Always set it.
- Schedule a recurring job with `hf jobs scheduled run "<cron>" ...` — only if the user explicitly wants a recurring/cron job, not for one-off experiments.
- Don't `uv add` anything locally for the job's deps — they live in the script's inline block and resolve on the worker.
