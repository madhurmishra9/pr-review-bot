"""Review orchestration: per-file review pass, grounding, verification pass."""

from __future__ import annotations

import logging

from .context_builder import build_file_context, build_repo_context
from .diff_parser import annotated_patch, parse_patch
from .github_client import GitHubClient
from .grounding import ground_finding
from .llm import (OllamaClient, REVIEW_SCHEMA, SUMMARY_SCHEMA, VERIFY_SCHEMA,
                  LLMError)
from .models import (FileDiff, Finding, PullRequestRef, ReviewResult, Severity,
                     Verdict)

log = logging.getLogger("pr_review_bot")

REVIEW_SYSTEM = """\
You are a precise code reviewer. You review ONE file's diff from a pull request.

Rules — follow them exactly:
- Report only real, defensible problems in the ADDED lines (marked '+').
  Never comment on unchanged context lines or removed lines.
- For each finding, 'line' MUST be the L-number shown in the diff for an
  added line, and 'evidence' MUST be an exact verbatim fragment copied from
  that same line.
- Do NOT speculate. If you are not certain something is a problem given the
  code you can see, do not report it.
- No nitpicks about formatting that a linter would catch.
- If the diff has no real problems, return an empty findings list. An empty
  list is a good, correct answer for clean code.
Respond only with JSON matching the required schema."""

VERIFY_SYSTEM = """\
You are auditing a code-review finding for correctness. You are given a
file's diff and one candidate finding. Decide whether the finding is a real,
technically correct problem that a competent maintainer would agree with.

Reject the finding if:
- the stated problem does not actually exist in the code shown,
- the claim is speculative and cannot be confirmed from the code shown,
- it misreads the code, or the 'problem' is handled elsewhere in the visible
  code,
- it is a trivial style preference.

Confirm it only if the problem is clearly present. Respond only with JSON."""

SUMMARY_SYSTEM = """\
You summarise a pull request for a human reviewer in 2-4 sentences: what it
changes and any overall risk worth noting. Be factual; describe only what is
in the diff. Respond only with JSON."""


class Reviewer:
    def __init__(self, gh: GitHubClient, llm: OllamaClient):
        self.gh = gh
        self.llm = llm

    def review(self, pr: PullRequestRef) -> ReviewResult:
        pr_data = self.gh.get_pull_request(pr)
        head_ref = pr_data["head"]["sha"]

        raw_files = self.gh.list_pr_files(pr)
        # Stable order so prompts (and therefore output) are reproducible.
        raw_files.sort(key=lambda f: f["filename"])
        diffs = [
            parse_patch(f["filename"], f["status"], f.get("patch"),
                        f.get("previous_filename"))
            for f in raw_files
        ]

        tree = self.gh.get_tree_paths(pr, head_ref)
        readme = self.gh.get_readme(pr, head_ref)
        repo_context = build_repo_context(pr, pr_data, tree, readme)

        findings: list[Finding] = []
        discarded: list[tuple[Finding, str]] = []
        reviewed: list[str] = []

        for fd in diffs:
            if fd.status == "removed" or not fd.new_lines:
                log.info("skipping %s (%s, no reviewable added lines)", fd.path, fd.status)
                continue
            reviewed.append(fd.path)
            log.info("reviewing %s (%d added lines)", fd.path, len(fd.added_linenos))

            head_content = self.gh.get_file_content(pr, fd.path, head_ref)
            candidates = self._review_file(repo_context, fd, head_content)

            for c in candidates:
                reason = ground_finding(c, fd)
                if reason:
                    log.info("discarded ungrounded finding in %s: %s", fd.path, reason)
                    discarded.append((c, f"grounding failed: {reason}"))
                    continue
                verdict, why = self._verify_finding(fd, c)
                c.verdict = verdict
                c.rejection_reason = why if verdict is Verdict.REJECTED else ""
                if verdict is Verdict.CONFIRMED:
                    findings.append(c)
                else:
                    log.info("verification rejected finding in %s:%d: %s",
                             fd.path, c.line, why)
                    discarded.append((c, f"verification rejected: {why}"))

        findings.sort(key=Finding.sort_key)
        summary = self._summarise(pr_data, diffs)
        return ReviewResult(
            pr=pr,
            title=pr_data.get("title", ""),
            summary=summary,
            findings=findings,
            discarded=discarded,
            files_reviewed=reviewed,
            model=self.llm.model,
        )

    # -- passes ---------------------------------------------------------------

    def _review_file(self, repo_context: str, fd: FileDiff,
                     head_content: str | None) -> list[Finding]:
        user = (repo_context + "\n\n---\n\n" + build_file_context(fd, head_content)
                + "\n\nReview the added lines of this file now.")
        try:
            data = self.llm.chat_json(REVIEW_SYSTEM, user, REVIEW_SCHEMA)
        except LLMError as exc:
            log.warning("review pass failed for %s: %s", fd.path, exc)
            return []
        out: list[Finding] = []
        for item in data.get("findings", []):
            try:
                out.append(Finding(
                    file=fd.path,
                    line=int(item["line"]),
                    severity=Severity(item["severity"]),
                    category=str(item["category"]),
                    summary=str(item["summary"]).strip(),
                    detail=str(item["detail"]).strip(),
                    evidence=str(item["evidence"]),
                    suggestion=str(item.get("suggestion", "")).strip(),
                ))
            except (KeyError, ValueError, TypeError) as exc:
                log.info("dropping malformed finding in %s: %s", fd.path, exc)
        return out

    def _verify_finding(self, fd: FileDiff, finding: Finding) -> tuple[Verdict, str]:
        user = (
            f"File: {fd.path}\n\nDiff with new-file line numbers:\n"
            f"{annotated_patch(fd)}\n\n"
            "Candidate finding to audit:\n"
            f"- line: {finding.line}\n"
            f"- severity: {finding.severity.value}\n"
            f"- category: {finding.category}\n"
            f"- summary: {finding.summary}\n"
            f"- detail: {finding.detail}\n"
            f"- evidence: {finding.evidence!r}\n\n"
            "Is this finding correct? Respond with your verdict."
        )
        try:
            data = self.llm.chat_json(VERIFY_SYSTEM, user, VERIFY_SCHEMA)
            return Verdict(data["verdict"]), str(data.get("reason", "")).strip()
        except (LLMError, KeyError, ValueError) as exc:
            # Fail closed: an unverifiable finding is never reported.
            return Verdict.REJECTED, f"verification pass failed ({exc})"

    def _summarise(self, pr_data: dict, diffs: list[FileDiff]) -> str:
        changed = "\n".join(
            f"- {d.path} ({d.status}, +{len(d.added_linenos)} lines)" for d in diffs
        )
        user = (
            f"Title: {pr_data.get('title', '')}\n"
            f"Description: {(pr_data.get('body') or '')[:1500]}\n"
            f"Changed files:\n{changed}\n\nSummarise this pull request."
        )
        try:
            data = self.llm.chat_json(SUMMARY_SYSTEM, user, SUMMARY_SCHEMA)
            return str(data.get("summary", "")).strip()
        except LLMError as exc:
            log.warning("summary pass failed: %s", exc)
            return "(summary unavailable)"
