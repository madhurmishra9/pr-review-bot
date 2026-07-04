# pr-review-bot

Review GitHub pull requests with a **local LLM** — no code ever leaves your machine
except the GitHub API calls needed to read the PR. Point it at a PR URL with a
personal access token and it produces a structured, line-anchored review.

```
pr-review-bot https://github.com/owner/repo/pull/123 --pat ghp_xxx
```

## How it works

```
PR URL + PAT
    │
    ▼
1. Fetch context ──── PR metadata, changed files + patches, repo file tree,
    │                 README, full content of changed files (GitHub REST API)
    ▼
2. Review pass ────── one deterministic LLM call per changed file (Ollama,
    │                 temperature 0 / top_k 1 / fixed seed, JSON-schema-
    │                 constrained output, diff annotated with explicit
    │                 new-side line numbers)
    ▼
3. Grounding ──────── every candidate finding is mechanically validated
    │                 against the parsed diff: the file must be in the PR,
    │                 the line must be an ADDED line, and the quoted
    │                 evidence must actually appear on that line.
    │                 Anything that fails is discarded, never reported.
    ▼
4. Verification ───── each surviving finding goes through an independent
    │                 second LLM pass that must CONFIRM or REJECT it.
    │                 Unverifiable findings fail closed (rejected).
    ▼
5. Report ─────────── markdown to stdout, optional JSON (--json), optional
                      posting to the PR as a review with line comments (--post)
```

## Accuracy and consistency — read this

Honest engineering note: **no LLM-based reviewer (local or hosted) can be
guaranteed 100% correct** — it may still miss real bugs (false negatives),
and a confirmed finding can still occasionally be debatable. Anyone claiming
otherwise is selling something. What this bot *does* guarantee by construction:

- **100% reproducible**: greedy decoding (temperature 0, top_k 1) with a fixed
  seed, stable file ordering, and byte-stable prompts — the same PR at the same
  commit produces the same review every run.
- **Zero fabricated locations**: a finding is only reported if its file, line
  number, and quoted evidence all match the real parsed diff. The model cannot
  invent files, point at lines the PR didn't add, or misquote code — such
  findings are mechanically discarded (visible with `--show-discarded`).
- **Zero unvetted claims**: every finding must survive an independent
  verification pass; if verification errors out, the finding is dropped
  (fail closed), so errors can't leak speculative findings into the report.
- **Schema-valid output, always**: generation is constrained to a JSON schema,
  so there is no free-text parsing that can silently go wrong.

The result: precision is engineered as high as the stack allows, and what the
bot *reports* is always anchored to code that really exists in the diff.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a code-capable model:

```bash
ollama pull qwen2.5-coder:7b     # default model; 14b/32b are more accurate
```

- A GitHub personal access token with read access to the target repo
  (classic: `repo` scope; fine-grained: *Contents: read* + *Pull requests: read*,
  plus *Pull requests: write* if you use `--post`).

## Install

```bash
pip install .
# or for development
pip install -e ".[dev]" && pytest
```

## Usage

```bash
# Print the review to stdout
pr-review-bot https://github.com/owner/repo/pull/123 --pat ghp_xxx

# Token from the environment instead of a flag
export GITHUB_TOKEN=ghp_xxx
pr-review-bot https://github.com/owner/repo/pull/123

# Bigger model, structured JSON output, progress logging
pr-review-bot https://github.com/owner/repo/pull/123 \
    --model qwen2.5-coder:14b --json review.json -v

# Post the review (summary + line comments) back to the PR
pr-review-bot https://github.com/owner/repo/pull/123 --post

# Show what the grounding/verification layers filtered out
pr-review-bot https://github.com/owner/repo/pull/123 --show-discarded
```

Exit codes: `0` clean review, `3` confirmed findings exist (useful for CI
gating), `1` runtime error, `2` bad invocation.

## Options

| Flag | Default | Purpose |
|---|---|---|
| `--pat` | `$GITHUB_TOKEN` / `$GH_TOKEN` | GitHub personal access token |
| `--model` | `qwen2.5-coder:7b` | Ollama model name |
| `--ollama-url` | `http://localhost:11434` | Ollama server |
| `--num-ctx` | `16384` | context window given to the model |
| `--post` | off | post review + line comments to the PR |
| `--json FILE` | off | write structured JSON result (`-` = stdout) |
| `--show-discarded` | off | include filtered candidates in the report |
| `-v` | off | progress logs on stderr |

## Security notes

- The PAT is only ever sent to `api.github.com`, over HTTPS.
- Prefer `GITHUB_TOKEN`/`GH_TOKEN` env vars over `--pat` so the token doesn't
  land in shell history or process listings.
- All model inference happens on your machine via Ollama; no repository
  content is sent to any third-party service.

## Project layout

```
src/pr_review_bot/
├── cli.py              # argument parsing, entry point
├── github_client.py    # GitHub REST client (PAT auth, retries, pagination)
├── diff_parser.py      # unified-diff → line anchors (the ground truth)
├── context_builder.py  # bounded repo/PR context for small local models
├── llm.py              # Ollama client: deterministic + schema-constrained
├── reviewer.py         # orchestration: review → ground → verify
├── grounding.py        # mechanical validation of findings against the diff
├── report.py           # markdown / JSON rendering, posting to GitHub
└── models.py           # dataclasses and enums
```
