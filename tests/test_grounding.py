from pr_review_bot.diff_parser import parse_patch
from pr_review_bot.grounding import ground_finding
from pr_review_bot.models import Finding, Severity

PATCH = """\
@@ -1,3 +1,4 @@
 import os
+password = "hunter2"
 def main():
     pass
"""


def make_finding(**overrides):
    base = dict(
        file="app.py",
        line=2,
        severity=Severity.HIGH,
        category="security",
        summary="Hardcoded credential",
        detail="A password literal is committed to source.",
        evidence='password = "hunter2"',
    )
    base.update(overrides)
    return Finding(**base)


def fd():
    return parse_patch("app.py", "modified", PATCH)


def test_valid_finding_passes():
    assert ground_finding(make_finding(), fd()) is None


def test_wrong_file_rejected():
    reason = ground_finding(make_finding(file="other.py"), fd())
    assert reason and "does not match" in reason


def test_line_outside_diff_rejected():
    reason = ground_finding(make_finding(line=99), fd())
    assert reason and "not part of the diff" in reason


def test_context_line_rejected():
    # line 1 ("import os") exists in the diff but was not added by the PR
    reason = ground_finding(make_finding(line=1, evidence="import os"), fd())
    assert reason and "unchanged context" in reason


def test_fabricated_evidence_rejected():
    reason = ground_finding(make_finding(evidence="secret_key = 'abc'"), fd())
    assert reason and "does not appear" in reason


def test_whitespace_differences_tolerated():
    assert ground_finding(make_finding(evidence='password="hunter2"'), fd()) is None


def test_too_short_evidence_rejected():
    reason = ground_finding(make_finding(evidence="p"), fd())
    assert reason and "too short" in reason
