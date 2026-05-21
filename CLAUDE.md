# Project Instructions

## Keep Code Simple

- Start with the simplest implementation that works. Do not over-engineer.
- Avoid abstractions, helpers, or utilities unless they are needed by more than one caller.
- Do not add configuration, feature flags, or extensibility hooks speculatively.
- Do not add error handling for cases that cannot happen in practice.
- Three similar lines of code is better than a premature abstraction.
- Add complexity only when the task actually requires it.

## Package Manager

- Use `uv` for all Python package management (not pip, poetry, or conda).
- Add dependencies with `uv add <package>`.
- Run scripts with `uv run <script>`.

## Available Skills

Skills are defined in `.claude/skills/` and run inline:

- At the start of a task, check whether any project skill matches the request by reading the frontmatter in `.claude/skills/*/SKILL.md`.
- If a skill matches, read its full `SKILL.md` and follow those instructions inline.
- Current skills:
  - `read-arxiv-paper`: Use when asked to read or summarize an arXiv paper from an arXiv URL.
