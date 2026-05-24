---
name: notebook-to-script
description: Use this skill when asked to convert a Jupyter notebook (.ipynb) into a runnable Python script (.py). Handles single files or glob patterns and writes the .py next to the source notebook by default.
---

You will be given one or more paths to `.ipynb` files (or a glob like `notebooks/*.ipynb`). Convert each into a `.py` script.

### Part 1: Resolve inputs

- Expand any glob and verify each path ends in `.ipynb` and exists. If the user gave a directory, treat it as `<dir>/*.ipynb`.
- Default output path: same directory as the source, with the `.ipynb` extension replaced by `.py`. If the user specified an output path or directory, honor that instead.
- If the target `.py` already exists, ask before overwriting (unless the user said "force" / "overwrite").

### Part 2: Convert

Use `jupyter nbconvert` via `uv run` (project rule: prefer `uv` for Python tooling):

```sh
uv run jupyter nbconvert --to script <notebook.ipynb> --output <basename-without-ext> --output-dir <dir>
```

Notes:
- `--output` takes the basename **without** the `.py` extension; nbconvert appends it.
- `--output-dir` controls the directory. Omit both flags to drop the `.py` next to the `.ipynb`.
- If `jupyter` is not available in the project env, fall back to `uvx --from jupyter-core --with nbconvert jupyter nbconvert ...` or instruct the user to `uv add --dev nbconvert ipykernel`.

For multi-file conversions, run the command per file (don't shell-glob into a single command — nbconvert handles one path at a time cleanly and per-file errors stay isolated).

### Part 3: Post-process (only if asked)

Apply these only when the user explicitly requests them:
- **Strip markdown cells:** add `--TemplateExporter.exclude_markdown=True`.
- **Strip cell outputs / metadata noise:** nbconvert script export already drops outputs; nothing extra needed.
- **Remove `# In[ ]:` cell markers:** post-process the `.py` with a sed/grep pass, e.g. `sed -i '/^# In\[/d' <file.py>`.
- **Preserve markdown as comments:** that's the default behavior — markdown cells become `# ` comment blocks.

### Part 4: Report

For each notebook converted, print one line: `wrote <output.py> (<N> cells)`. If any conversions failed, list the failures with the nbconvert error message. Do not paste the generated `.py` contents back into chat.
