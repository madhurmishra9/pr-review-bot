"""Review orchestration: per-file review pass, grounding, verification pass."""

from __future__ import annotations

import logging

from .context_builder import build_file_context, build_repo_context
from .diff_parser import annotated_patch, parse_patch
from .github_client import GitHubClient
from .grounding import ground_finding
from .llm import (OllamaClient, AUDIT_SCHEMA, DETERMINISTIC_OPTIONS, REVIEW_SCHEMA,
                  SUMMARY_SCHEMA, VERIFY_SCHEMA, LLMError)
from .models import (FileDiff, Finding, PullRequestRef, ReviewResult, Severity,
                     Verdict)

log = logging.getLogger("pr_review_bot")

# Default number of times a review is (re)generated before it is reported
# as-is even though it is still flagged hallucinated. Kept small: each
# attempt re-runs every LLM pass for every changed file.
DEFAULT_MAX_ATTEMPTS = 2

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

AUDIT_SYSTEM = """\
You are the final quality gate before a code review is shown to a human. You
are given the full set of changed files' diffs, a summary written about the
pull request, and the list of findings that survived independent grounding
and verification passes.

Decide whether this review, taken as a whole, is hallucinated: does it state
or imply anything that is not actually supported by the diffs shown? Flag it
as hallucinated if:
- the summary describes a change, file, or behavior that is not present in
  the diffs,
- the summary contradicts what the findings say, or the findings contradict
  each other,
- a finding's explanation claims something about the surrounding code that
  is not visible in the diff provided,
- the review as a whole overstates certainty about something that cannot be
  confirmed from the code shown.

Do not flag it for being incomplete, for missing findings, or for style
preferences — only for stating something false or unsupported. Respond only
with JSON."""


class Reviewer:
    def __init__(self, gh: GitHubClient, llm: OllamaClient):
        self.gh = gh
        self.llm = llm

    def review(self, pr: PullRequestRef,
               max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> ReviewResult:
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

        # Fetch each reviewable file's content once; it does not change
        # between attempts, only the LLM's read of it might.
        heads: dict[str, str | None] = {}
        for fd in diffs:
            if fd.status == "removed" or not fd.new_lines:
                continue
            heads[fd.path] = self.gh.get_file_content(pr, fd.path, head_ref)

        result = None
        feedback = ""
        for attempt in range(1, max_attempts + 1):
            result = self._generate(pr, pr_data, diffs, heads, repo_context,
                                     feedback, seed_offset=attempt - 1)
            result.attempts = attempt
            hallucinated, reason = self._audit_review(diffs, result)
            result.hallucinated = hallucinated
            result.audit_reason = reason
            if not hallucinated:
                break
            log.warning(
                "review attempt %d/%d flagged as hallucinated, regenerating: %s",
                attempt, max_attempts, reason,
            )
            feedback = (
                "A previous draft of this review was rejected by an audit pass "
                f"for the following reason: {reason}\n"
                "Do not repeat that mistake. Only state what the diff actually shows."
            )
        return result

    # -- passes ---------------------------------------------------------------

    def _generate(self, pr: PullRequestRef, pr_data: dict, diffs: list[FileDiff],
                  heads: dict[str, str | None], repo_context: str,
                  feedback: str, seed_offset: int) -> ReviewResult:
        findings: list[Finding] = []
        discarded: list[tuple[Finding, str]] = []
        reviewed: list[str] = []

        for fd in diffs:
            if fd.path not in heads:
                log.info("skipping %s (%s, no reviewable added lines)", fd.path, fd.status)
                continue
            reviewed.append(fd.path)
            log.info("reviewing %s (%d added lines)", fd.path, len(fd.added_linenos))

            candidates = self._review_file(repo_context, fd, heads[fd.path],
                                            feedback, seed_offset)

            for c in candidates:
                reason = ground_finding(c, fd)
                if reason:
                    log.info("discarded ungrounded finding in %s: %s", fd.path, reason)
                    discarded.append((c, f"grounding failed: {reason}"))
                    continue
                verdict, why = self._verify_finding(fd, c, seed_offset)
                c.verdict = verdict
                c.rejection_reason = why if verdict is Verdict.REJECTED else ""
                if verdict is Verdict.CONFIRMED:
                    findings.append(c)
                else:
                    log.info("verification rejected finding in %s:%d: %s",
                             fd.path, c.line, why)
                    discarded.append((c, f"verification rejected: {why}"))

        findings.sort(key=Finding.sort_key)
        summary = self._summarise(pr_data, diffs, feedback, seed_offset)
        return ReviewResult(
            pr=pr,
            title=pr_data.get("title", ""),
            summary=summary,
            findings=findings,
            discarded=discarded,
            files_reviewed=reviewed,
            model=self.llm.model,
        )

    def _review_file(self, repo_context: str, fd: FileDiff,
                     head_content: str | None, feedback: str,
                     seed_offset: int) -> list[Finding]:
        user = (repo_context + "\n\n---\n\n" + build_file_context(fd, head_content)
                + "\n\nReview the added lines of this file now.")
        if feedback:
            user += f"\n\n{feedback}"
        try:
            data = self.llm.chat_json(REVIEW_SYSTEM, user, REVIEW_SCHEMA,
                                       seed=_seed(seed_offset))
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

    def _verify_finding(self, fd: FileDiff, finding: Finding,
                        seed_offset: int = 0) -> tuple[Verdict, str]:
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
            data = self.llm.chat_json(VERIFY_SYSTEM, user, VERIFY_SCHEMA,
                                       seed=_seed(seed_offset))
            return Verdict(data["verdict"]), str(data.get("reason", "")).strip()
        except (LLMError, KeyError, ValueError) as exc:
            # Fail closed: an unverifiable finding is never reported.
            return Verdict.REJECTED, f"verification pass failed ({exc})"

    def _summarise(self, pr_data: dict, diffs: list[FileDiff], feedback: str = "",
                   seed_offset: int = 0) -> str:
        changed = "\n".join(
            f"- {d.path} ({d.status}, +{len(d.added_linenos)} lines)" for d in diffs
        )
        user = (
            f"Title: {pr_data.get('title', '')}\n"
            f"Description: {(pr_data.get('body') or '')[:1500]}\n"
            f"Changed files:\n{changed}\n\nSummarise this pull request."
        )
        if feedback:
            user += f"\n\n{feedback}"
        try:
            data = self.llm.chat_json(SUMMARY_SYSTEM, user, SUMMARY_SCHEMA,
                                       seed=_seed(seed_offset))
            return str(data.get("summary", "")).strip()
        except LLMError as exc:
            log.warning("summary pass failed: %s", exc)
            return "(summary unavailable)"

    def _audit_review(self, diffs: list[FileDiff],
                      result: ReviewResult) -> tuple[bool, str]:
        """Holistic hallucination check over the assembled review.

        This runs after grounding and per-finding verification, as a last
        gate over the review as a whole (in particular over the summary,
        which — unlike findings — is otherwise never checked against the
        diff). Consistent with the rest of the pipeline this fails closed:
        an audit pass that errors out is treated as hallucinated, so an
        unaudited review is never posted silently.
        """
        if not result.findings and not result.summary:
            return False, ""
        diff_text = "\n\n".join(
            f"--- {fd.path} ---\n{annotated_patch(fd)}"
            for fd in diffs if fd.path in result.files_reviewed
        )
        findings_text = "\n".join(
            f"- {f.file}:{f.line} [{f.severity.value}/{f.category}] {f.summary} "
            f"— {f.detail} (evidence: {f.evidence!r})"
            for f in result.findings
        ) or "(none)"
        user = (
            f"Diffs of the files reviewed:\n{diff_text}\n\n"
            f"Summary written about the pull request:\n{result.summary or '(none)'}\n\n"
            f"Findings that survived grounding and verification:\n{findings_text}\n\n"
            "Audit this review now."
        )
        try:
            data = self.llm.chat_json(AUDIT_SYSTEM, user, AUDIT_SCHEMA)
            return bool(data["hallucinated"]), str(data.get("reason", "")).strip()
        except (LLMError, KeyError, ValueError) as exc:
            # Fail closed: if we cannot audit the review, treat it as
            # unproven rather than silently reporting it as clean.
            return True, f"audit pass failed ({exc})"


def _seed(offset: int) -> int:
    return DETERMINISTIC_OPTIONS["seed"] + offset
