---
name: bootstrap-llm-experiment
description: Use this skill when the user asks to bootstrap, scaffold, or start a new local LLM inference/generation experiment. Creates a new top-level directory with a runnable run.py (transformers or vLLM), a README, and a notes.md.
---

You will scaffold a new local LLM inference experiment as a top-level directory in this repo, matching the layout of existing experiments (`open-env/`, `sae/`, `harness/`).

This skill is only for **local inference / generation** with `transformers` or `vLLM`. It is not for training, RL, eval harnesses, or interp pipelines.

### Part 1: Resolve the experiment name

- If the user supplied a name in the invocation, use it.
- Otherwise, ask via `AskUserQuestion` for a kebab-case name.
- Reject names containing `/`, whitespace, or starting with `.`.
- If `./<name>/` already exists, ask the user to pick a different name (do not overwrite without explicit confirmation).

### Part 2: Pick the inference backend

Ask via `AskUserQuestion`:

- `transformers` — light, single-GPU, `AutoModelForCausalLM` + `.generate()`.
- `vLLM` — batched, faster, heavier deps.

### Part 3: Pick the model id

Ask via `AskUserQuestion` for the HF model id, defaulting to `Qwen/Qwen3-1.7B`. Accept whatever the user provides verbatim — do not validate against the Hub.

### Part 4: Create the directory

Make `./<name>/` and write three files by reading each template from `.claude/skills/bootstrap-llm-experiment/templates/` and substituting placeholders with plain `str.replace`:

| Destination          | Template                                     |
| -------------------- | -------------------------------------------- |
| `./<name>/run.py`    | `templates/run_transformers.py` or `templates/run_vllm.py` |
| `./<name>/README.md` | `templates/README.md`                        |
| `./<name>/notes.md`  | `templates/notes.md`                         |

Placeholders to replace in all templates:

- `{{NAME}}` → the experiment name
- `{{MODEL_ID}}` → the chosen HF model id
- `{{DATE}}` → today's date in `YYYY-MM-DD` (use the current date from the conversation context)

### Part 5: Report

Print the created paths and the exact first command to try, e.g.:

```
Created ./<name>/{run.py,README.md,notes.md}
Try: uv run ./<name>/run.py --prompt "hello"
```

Do not execute the command.
