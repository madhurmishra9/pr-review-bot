from pr_review_bot.models import PullRequestRef
from pr_review_bot.report import HallucinatedReviewError, post_to_github
from pr_review_bot.reviewer import Reviewer

PATCH = """\
@@ -1,2 +1,3 @@
 import os
+password = "hunter2"
 def main():
"""


class FakeGitHubClient:
    def __init__(self):
        self.posted = None

    def get_pull_request(self, pr):
        return {"title": "Add config", "body": "adds a config value",
                 "head": {"sha": "deadbeef", "ref": "feature"},
                 "base": {"ref": "main"}}

    def list_pr_files(self, pr):
        return [{"filename": "app.py", "status": "modified", "patch": PATCH}]

    def get_tree_paths(self, pr, ref, limit=400):
        return ["app.py"]

    def get_readme(self, pr, ref):
        return None

    def get_file_content(self, pr, path, ref):
        return 'import os\npassword = "hunter2"\ndef main():\n    pass\n'

    def post_review(self, pr, body, comments):
        self.posted = (body, comments)
        return {"html_url": "https://github.com/o/r/pull/1#review"}


class ScriptedLLM:
    """Fake Ollama client: returns whatever chat_json call comes next off the
    script for that "kind" of prompt, identified by a marker in the system
    prompt. Each attempt of the reviewer issues: 1 review call, 0+ verify
    calls, 1 summary call, 1 audit call — in that order."""

    model = "fake-model"

    def __init__(self, review, verify, summary, audit):
        self.review = list(review)
        self.verify = list(verify)
        self.summary = list(summary)
        self.audit = list(audit)
        self.calls = []

    def chat_json(self, system, user, schema, seed=None):
        self.calls.append((system[:20], seed))
        if "final quality gate" in system:
            return self.audit.pop(0)
        if "auditing a code-review finding" in system:
            return self.verify.pop(0)
        if "summarise a pull request" in system.lower():
            return self.summary.pop(0)
        return self.review.pop(0)


FINDING = {
    "line": 2,
    "severity": "high",
    "category": "security",
    "summary": "Hardcoded credential",
    "detail": "A password literal is committed to source.",
    "evidence": 'password = "hunter2"',
}


def make_reviewer(llm):
    return Reviewer(FakeGitHubClient(), llm)


def test_clean_audit_reports_immediately():
    llm = ScriptedLLM(
        review=[{"findings": [FINDING]}],
        verify=[{"verdict": "confirmed", "reason": "real secret"}],
        summary=[{"summary": "Adds a hardcoded password."}],
        audit=[{"hallucinated": False, "reason": ""}],
    )
    result = make_reviewer(llm).review(PullRequestRef("o", "r", 1))

    assert result.hallucinated is False
    assert result.attempts == 1
    assert len(result.findings) == 1


def test_hallucinated_review_regenerates_then_succeeds():
    llm = ScriptedLLM(
        review=[{"findings": [FINDING]}, {"findings": [FINDING]}],
        verify=[
            {"verdict": "confirmed", "reason": "real secret"},
            {"verdict": "confirmed", "reason": "real secret"},
        ],
        summary=[
            {"summary": "Adds a fabricated migration system."},  # hallucinated
            {"summary": "Adds a hardcoded password."},           # corrected
        ],
        audit=[
            {"hallucinated": True, "reason": "summary invents a migration system"},
            {"hallucinated": False, "reason": ""},
        ],
    )
    result = make_reviewer(llm).review(PullRequestRef("o", "r", 1), max_attempts=3)

    assert result.hallucinated is False
    assert result.attempts == 2
    assert result.summary == "Adds a hardcoded password."
    # Second review-file call must have received the audit's feedback appended.
    # (Grounding/verification content isn't re-checked here; behavior is
    # covered by test_grounding.py.)


def test_persistent_hallucination_exhausts_attempts_and_blocks_posting():
    llm = ScriptedLLM(
        review=[{"findings": []}, {"findings": []}],
        verify=[],
        summary=[{"summary": "bad"}, {"summary": "still bad"}],
        audit=[
            {"hallucinated": True, "reason": "nope"},
            {"hallucinated": True, "reason": "still nope"},
        ],
    )
    gh = FakeGitHubClient()
    result = Reviewer(gh, llm).review(PullRequestRef("o", "r", 1), max_attempts=2)

    assert result.hallucinated is True
    assert result.attempts == 2
    assert result.audit_reason == "still nope"

    try:
        post_to_github(gh, result)
        assert False, "expected HallucinatedReviewError"
    except HallucinatedReviewError:
        pass
    assert gh.posted is None
