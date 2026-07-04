"""Assemble bounded repository/PR context for the local model.

Local models have small context windows, so everything here is budgeted:
the repo tree is truncated, the README is excerpted, and full file content
is only included around the changed hunks.
"""

from __future__ import annotations

from .diff_parser import annotated_patch
from .models import FileDiff, PullRequestRef

TREE_LIMIT = 300
README_LINES = 60
HUNK_CONTEXT_LINES = 30
MAX_FULL_FILE_LINES = 400


def build_repo_context(pr: PullRequestRef, pr_data: dict, tree: list[str],
                       readme: str | None) -> str:
    parts = [
        f"Repository: {pr.slug}",
        f"Pull request #{pr.number}: {pr_data.get('title', '')}",
        f"Base branch: {pr_data.get('base', {}).get('ref')} <- "
        f"Head branch: {pr_data.get('head', {}).get('ref')}",
    ]
    body = (pr_data.get("body") or "").strip()
    if body:
        parts.append("PR description:\n" + body[:2000])
    if readme:
        excerpt = "\n".join(readme.splitlines()[:README_LINES])
        parts.append("README excerpt:\n" + excerpt)
    if tree:
        listing = "\n".join(tree[:TREE_LIMIT])
        suffix = f"\n... ({len(tree) - TREE_LIMIT} more files)" if len(tree) > TREE_LIMIT else ""
        parts.append("Repository file tree:\n" + listing + suffix)
    return "\n\n".join(parts)


def build_file_context(fd: FileDiff, head_content: str | None) -> str:
    """Diff (with explicit line numbers) plus surrounding source context."""
    parts = [f"File: {fd.path} (status: {fd.status})"]
    if fd.previous_path:
        parts.append(f"Renamed from: {fd.previous_path}")
    parts.append("Diff with new-file line numbers "
                 "(+ marks added lines — only these may receive findings):")
    parts.append(annotated_patch(fd))

    if head_content and fd.new_lines:
        lines = head_content.splitlines()
        if len(lines) <= MAX_FULL_FILE_LINES:
            numbered = "\n".join(f"{i}: {l}" for i, l in enumerate(lines, 1))
            parts.append("Full file content after the change:\n" + numbered)
        else:
            parts.append("Surrounding file content after the change:\n"
                         + _hunk_windows(fd, lines))
    return "\n\n".join(parts)


def _hunk_windows(fd: FileDiff, lines: list[str]) -> str:
    """Excerpts of the head-version file around each changed region."""
    wanted: set[int] = set()
    for n in fd.new_lines:
        lo = max(1, n - HUNK_CONTEXT_LINES)
        hi = min(len(lines), n + HUNK_CONTEXT_LINES)
        wanted.update(range(lo, hi + 1))
    out: list[str] = []
    prev = 0
    for n in sorted(wanted):
        if n > len(lines):
            break
        if prev and n != prev + 1:
            out.append("    ...")
        out.append(f"{n}: {lines[n - 1]}")
        prev = n
    return "\n".join(out)
