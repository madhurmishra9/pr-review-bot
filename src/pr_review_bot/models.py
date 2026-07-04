"""Core data types shared across the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @classmethod
    def order(cls, value: "Severity") -> int:
        return [cls.CRITICAL, cls.HIGH, cls.MEDIUM, cls.LOW, cls.INFO].index(value)


class Verdict(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class PullRequestRef:
    owner: str
    repo: str
    number: int

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    def __str__(self) -> str:
        return f"{self.slug}#{self.number}"


@dataclass
class DiffLine:
    """One line on the *new* side of a unified diff (added or context)."""

    new_lineno: int
    content: str
    is_addition: bool


@dataclass
class FileDiff:
    """Parsed patch for a single changed file."""

    path: str
    status: str  # added | modified | removed | renamed
    patch: str | None
    previous_path: str | None = None
    # new-side line number -> DiffLine, only lines present in the patch
    new_lines: dict[int, DiffLine] = field(default_factory=dict)
    removed_lines: list[str] = field(default_factory=list)

    @property
    def added_linenos(self) -> list[int]:
        return sorted(n for n, l in self.new_lines.items() if l.is_addition)


@dataclass
class Finding:
    """A single review finding produced by the LLM."""

    file: str
    line: int
    severity: Severity
    category: str
    summary: str
    detail: str
    evidence: str  # verbatim quote of the offending line, used for grounding
    suggestion: str = ""
    verdict: Verdict | None = None
    rejection_reason: str = ""

    def sort_key(self) -> tuple:
        return (Severity.order(self.severity), self.file, self.line)


@dataclass
class ReviewResult:
    pr: PullRequestRef
    title: str
    summary: str
    findings: list[Finding]
    discarded: list[tuple[Finding, str]]  # finding, reason it was dropped
    files_reviewed: list[str]
    model: str
