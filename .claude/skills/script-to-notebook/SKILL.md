---
name: script-to-notebook
description: Use this skill when asked to convert a Python script (.py) into a Jupyter notebook (.ipynb). Handles single files or glob patterns and writes the .ipynb next to the source script by default.
---

You will be given one or more paths to `.py` files (or a glob like `scripts/*.py`). Convert each into an `.ipynb` notebook.

### Part 1: Resolve inputs

- Expand any glob and verify each path ends in `.py` and exists. If the user gave a directory, treat it as `<dir>/*.py`.
- Default output path: same directory as the source, with `.py` replaced by `.ipynb`. If the user specified an output path or directory, honor that instead.
- If the target `.ipynb` already exists, ask before overwriting (unless the user said "force" / "overwrite").

### Part 2: Inspect the script first

Read the source `.py` to decide cell boundaries:

- **`# %%` markers present** → script is already in [percent format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html); each `# %%` starts a new code cell, `# %% [markdown]` starts a markdown cell. Use `jupytext`.
- **No markers** → treat the whole file as a single code cell, OR split on top-level `def`/`class`/blank-line groups only if the user explicitly asks for "smart" splitting. Default is whole-file single cell.

### Part 3: Convert

Use `jupytext` via `uv run` (project rule: prefer `uv` for Python tooling):

```sh
uv run jupytext --to notebook <script.py> --output <output.ipynb>
```

- Omit `--output` to write `<script>.ipynb` next to the source.
- If `jupytext` is not in the project env, fall back to `uvx jupytext --to notebook ...` or instruct the user to `uv add --dev jupytext`.

For multiple files, run per file so per-file errors stay isolated.

### Part 4: Post-process (only if asked)

- **Set kernel:** `uv run jupytext --set-kernel python3 <output.ipynb>` (or a named kernel the user specifies).
- **Pair the files so edits to either stay in sync:** `uv run jupytext --set-formats ipynb,py:percent <output.ipynb>`.
- **Execute cells after conversion:** add `--execute` to the convert command (requires a kernel — slow, only do if the user asks).

### Part 5: Report

For each script converted, print one line: `wrote <output.ipynb> (<N> cells)`. If any conversions failed, list failures with the jupytext error message. Do not paste the generated notebook JSON back into chat.
