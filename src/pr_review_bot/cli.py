"""Command-line entry point.

    pr-review-bot https://github.com/owner/repo/pull/123 --pat ghp_xxx
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .github_client import GitHubClient, GitHubError, parse_pr_url
from .llm import DEFAULT_MODEL, DEFAULT_URL, LLMError, OllamaClient
from .report import post_to_github, to_json, to_markdown
from .reviewer import Reviewer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pr-review-bot",
        description="Review a GitHub pull request with a local LLM (Ollama). "
                    "Findings are deterministic, diff-anchored, and verified "
                    "in a second pass before being reported.",
    )
    p.add_argument("pr_url", help="Pull request URL, e.g. https://github.com/o/r/pull/1")
    p.add_argument("--pat", help="GitHub personal access token "
                                 "(or set GITHUB_TOKEN / GH_TOKEN)")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    p.add_argument("--ollama-url", default=DEFAULT_URL,
                   help=f"Ollama server URL (default: {DEFAULT_URL})")
    p.add_argument("--num-ctx", type=int, default=16384,
                   help="Model context window in tokens (default: 16384)")
    p.add_argument("--post", action="store_true",
                   help="Post the review to the PR (default: print to stdout only)")
    p.add_argument("--json", dest="json_out", metavar="FILE",
                   help="Also write the structured result as JSON to FILE ('-' for stdout)")
    p.add_argument("--show-discarded", action="store_true",
                   help="Include candidates rejected by grounding/verification "
                        "in the markdown output")
    p.add_argument("-v", "--verbose", action="store_true", help="Log progress to stderr")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
    )

    token = args.pat or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("error: provide a PAT via --pat or the GITHUB_TOKEN env var",
              file=sys.stderr)
        return 2

    try:
        pr = parse_pr_url(args.pr_url)
    except GitHubError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    llm = OllamaClient(base_url=args.ollama_url, model=args.model,
                       num_ctx=args.num_ctx)
    try:
        llm.check_available()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    gh = GitHubClient(token)
    try:
        result = Reviewer(gh, llm).review(pr)
    except (GitHubError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(to_markdown(result, include_discarded=args.show_discarded))

    if args.json_out:
        payload = to_json(result)
        if args.json_out == "-":
            print(payload)
        else:
            with open(args.json_out, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")

    if args.post:
        try:
            url = post_to_github(gh, result)
            print(f"\nPosted review: {url}", file=sys.stderr)
        except GitHubError as exc:
            print(f"error posting review: {exc}", file=sys.stderr)
            return 1

    # Non-zero exit when confirmed findings exist, for CI-style gating.
    return 3 if result.findings else 0


if __name__ == "__main__":
    sys.exit(main())
