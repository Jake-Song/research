---
name: learn-package
description: Use this skill when asked to learn, document, or build knowledge of the basic usage of a package, library, framework, SDK, or CLI tool. Produces a reusable knowledge file under `./knowledge/` covering install, imports, core API, and minimal examples.
---

You will be given the name of a package (e.g. `polars`, `httpx`, `transformers`, `Next.js`). The goal is to produce a compact, reusable reference of its **basic usage** at `./knowledge/package_{tag}.md` so future conversations can load it without re-fetching docs.

### Part 1: Pick a tag

Derive a short kebab-case tag from the package name (e.g. `Next.js` → `nextjs`, `polars` → `polars`, `@anthropic-ai/sdk` → `anthropic-sdk`). Check that `./knowledge/package_{tag}.md` does not already exist — if it does, ask the user whether to overwrite or pick a new tag.

### Part 2: Fetch current documentation

Use the `ctx7` CLI (see global ctx7 rules) — do NOT rely on training-data knowledge of the API:

1. `npx ctx7@latest library "<package name>" "basic usage, installation, getting started"` — pick the best-matching `/org/project` ID.
2. `npx ctx7@latest docs <libraryId> "installation, basic usage, common API, minimal example"` — fetch the docs.

If the first query is too vague, re-query for the specific things you still need (e.g. "how to make an async request", "how to read a CSV"). Cap at 3 `ctx7` calls per skill run.

### Part 3: Write the knowledge file

Write `./knowledge/package_{tag}.md` with this structure — keep it **terse and example-led**, not a docs dump:

```markdown
# {Package Name} — basic usage

- **What it is:** one sentence.
- **Version referenced:** {version from ctx7 if available, else "latest as of {date}"}
- **Docs:** {official docs URL}

## Install

{exact install command(s), e.g. `uv add polars` — prefer `uv` for Python per project rules}

## Import / Entry

{minimal import or setup snippet}

## Core concepts

{2–5 bullets naming the primary objects/functions a user touches first}

## Minimal example

{a single end-to-end runnable snippet covering the most common task — the "hello world"}

## Common patterns

{3–6 short snippets for the next-most-common operations, each with a one-line caption}

## Gotchas

{anything the docs flag as easy to get wrong — version-specific behavior, footguns, deprecations}
```

Rules for the content:

- Every code snippet must come from the ctx7 docs you fetched, not from memory. If ctx7 didn't return an example for something, omit that section rather than guess.
- No prose paragraphs — bullets and code only.
- Prefer the modern/recommended API when docs show multiple options; note the older one only if it's still common in the wild.
- Total length target: ~80–150 lines. If you find yourself writing more, you're including too much.

### Part 4: Report

Tell the user the file path you wrote and give a one-line summary of what's in it. Do not paste the file contents back into chat.
