"""Unified-diff parsing.

GitHub's ``/pulls/{n}/files`` endpoint returns a ``patch`` string per file
(unified diff without the ``diff --git`` header). This module turns each patch
into a map of *new-side* line numbers to their content, which is the ground
truth every LLM finding must anchor to before it is reported.
"""

from __future__ import annotations

import re

from .models import DiffLine, FileDiff

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(path: str, status: str, patch: str | None,
                previous_path: str | None = None) -> FileDiff:
    """Parse one file's patch into a FileDiff with line anchors.

    Binary files and very large files come back with ``patch=None``; they are
    kept in the FileDiff (so the review can mention them) but carry no anchors,
    which means no line-level finding can ever attach to them.
    """
    fd = FileDiff(path=path, status=status, patch=patch, previous_path=previous_path)
    if not patch:
        return fd

    new_lineno = 0
    in_hunk = False
    for raw in patch.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            new_lineno = int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw.startswith("+"):
            fd.new_lines[new_lineno] = DiffLine(new_lineno, raw[1:], is_addition=True)
            new_lineno += 1
        elif raw.startswith("-"):
            fd.removed_lines.append(raw[1:])
        elif raw.startswith("\\"):  # "\ No newline at end of file"
            continue
        else:
            content = raw[1:] if raw.startswith(" ") else raw
            fd.new_lines[new_lineno] = DiffLine(new_lineno, content, is_addition=False)
            new_lineno += 1
    return fd


def annotated_patch(fd: FileDiff) -> str:
    """Render the patch with explicit new-side line numbers.

    Giving the model unambiguous line numbers (instead of asking it to count
    hunk offsets itself) removes the single most common source of wrong line
    references in LLM reviews.
    """
    if not fd.patch:
        return "(no textual diff available)"
    out: list[str] = []
    new_lineno = 0
    in_hunk = False
    for raw in fd.patch.splitlines():
        m = _HUNK_RE.match(raw)
        if m:
            new_lineno = int(m.group(3))
            in_hunk = True
            out.append(raw)
            continue
        if not in_hunk:
            out.append(raw)
            continue
        if raw.startswith("+"):
            out.append(f"L{new_lineno}: + {raw[1:]}")
            new_lineno += 1
        elif raw.startswith("-"):
            out.append(f"      - {raw[1:]}")
        elif raw.startswith("\\"):
            out.append(raw)
        else:
            out.append(f"L{new_lineno}:   {raw[1:] if raw.startswith(' ') else raw}")
            new_lineno += 1
    return "\n".join(out)
