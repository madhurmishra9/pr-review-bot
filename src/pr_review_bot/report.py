"""Render the review as markdown and (optionally) post it to the PR."""

from __future__ import annotations

import json
from dataclasses import asdict

from .github_client import GitHubClient
from .models import ReviewResult

_SEVERITY_BADGE = {
    "critical": "🔴 CRITICAL",
    "high": "🟠 HIGH",
    "medium": "🟡 MEDIUM",
    "low": "🔵 LOW",
    "info": "⚪ INFO",
}


def to_markdown(result: ReviewResult, include_discarded: bool = False) -> str:
    lines = [
        f"## PR review: {result.pr} — {result.title}",
        "",
        f"*Reviewed by a local LLM (`{result.model}`), deterministic decoding. "
        "Every finding below is anchored to an added line in the diff and "
        "survived an independent verification pass.*",
        "",
        "### Summary",
        result.summary or "(none)",
        "",
    ]
    if not result.findings:
        lines += ["### Findings", "",
                  "No confirmed issues found in the added lines. "
                  f"{len(result.files_reviewed)} file(s) reviewed."]
    else:
        lines += [f"### Findings ({len(result.findings)})", ""]
        for i, f in enumerate(result.findings, 1):
            lines += [
                f"**{i}. {_SEVERITY_BADGE[f.severity.value]}** `{f.file}:{f.line}` "
                f"— {f.summary}",
                "",
                f.detail,
            ]
            if f.suggestion:
                lines += ["", f"*Suggestion:* {f.suggestion}"]
            lines += ["", f"> `{f.evidence.strip()}`", ""]

    lines += ["", f"Files reviewed: {', '.join(result.files_reviewed) or '(none)'}"]

    if include_discarded and result.discarded:
        lines += ["", "<details><summary>Candidates discarded by grounding/"
                      f"verification ({len(result.discarded)})</summary>", ""]
        for f, reason in result.discarded:
            lines.append(f"- `{f.file}:{f.line}` {f.summary} — *{reason}*")
        lines += ["", "</details>"]
    return "\n".join(lines)


def to_json(result: ReviewResult) -> str:
    payload = {
        "pr": str(result.pr),
        "title": result.title,
        "model": result.model,
        "summary": result.summary,
        "files_reviewed": result.files_reviewed,
        "findings": [
            {**asdict(f), "severity": f.severity.value,
             "verdict": f.verdict.value if f.verdict else None}
            for f in result.findings
        ],
        "discarded": [
            {"file": f.file, "line": f.line, "summary": f.summary, "reason": reason}
            for f, reason in result.discarded
        ],
    }
    return json.dumps(payload, indent=2)


def post_to_github(gh: GitHubClient, result: ReviewResult) -> str:
    """Post the review to the PR; returns the review URL."""
    comments = [
        {
            "path": f.file,
            "line": f.line,
            "side": "RIGHT",
            "body": (f"**{_SEVERITY_BADGE[f.severity.value]}** ({f.category}) "
                     f"{f.summary}\n\n{f.detail}"
                     + (f"\n\n*Suggestion:* {f.suggestion}" if f.suggestion else "")),
        }
        for f in result.findings
    ]
    body = to_markdown(result)
    review = gh.post_review(result.pr, body, comments)
    return review.get("html_url", "")
