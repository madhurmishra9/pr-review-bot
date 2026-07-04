"""Grounding: mechanical validation of every LLM finding against the diff.

This layer is what makes the bot's output trustworthy. The model can only
*propose* findings; a finding is reported only if all of these hold:

1. the file exists in the PR diff,
2. the line number is an ADDED line on the new side of that file's patch
   (we never blame pre-existing code the PR didn't touch),
3. the quoted evidence actually appears on that line.

Anything that fails is discarded and recorded, never silently reported.
"""

from __future__ import annotations

from .models import FileDiff, Finding

_EVIDENCE_MIN_CHARS = 3


def ground_finding(finding: Finding, fd: FileDiff) -> str | None:
    """Return None if the finding is anchored to the diff, else the reason
    it must be discarded."""
    if finding.file != fd.path:
        return f"finding file {finding.file!r} does not match diff file {fd.path!r}"
    line = fd.new_lines.get(finding.line)
    if line is None:
        return f"line {finding.line} is not part of the diff for {fd.path}"
    if not line.is_addition:
        return (f"line {finding.line} in {fd.path} is unchanged context, "
                "not a line added by this PR")

    evidence = finding.evidence.strip()
    if len(evidence) < _EVIDENCE_MIN_CHARS:
        return "evidence quote is too short to verify"
    # Whitespace-insensitive containment: models often normalise indentation.
    if _squash(evidence) not in _squash(line.content):
        return (f"evidence {evidence!r} does not appear on line "
                f"{finding.line} of {fd.path} (actual: {line.content.strip()!r})")
    return None


def _squash(s: str) -> str:
    return "".join(s.split())
