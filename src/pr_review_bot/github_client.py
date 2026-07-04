"""Thin GitHub REST v3 client authenticated with a personal access token."""

from __future__ import annotations

import base64
import re
import time

import requests

from .models import PullRequestRef

_PR_URL_RE = re.compile(
    r"^https?://(?:www\.)?github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)

API_ROOT = "https://api.github.com"


class GitHubError(RuntimeError):
    pass


def parse_pr_url(url: str) -> PullRequestRef:
    m = _PR_URL_RE.match(url.strip())
    if not m:
        raise GitHubError(
            f"Not a GitHub pull request URL: {url!r} "
            "(expected https://github.com/<owner>/<repo>/pull/<number>)"
        )
    return PullRequestRef(m.group("owner"), m.group("repo"), int(m.group("number")))


class GitHubClient:
    def __init__(self, token: str, api_root: str = API_ROOT):
        self.api_root = api_root.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "pr-review-bot",
            }
        )

    def _get(self, path: str, **params) -> requests.Response:
        for attempt in range(4):
            resp = self.session.get(f"{self.api_root}{path}", params=params, timeout=30)
            if resp.status_code in (429, 502, 503) and attempt < 3:
                time.sleep(2 ** (attempt + 1))
                continue
            break
        if resp.status_code == 401:
            raise GitHubError("GitHub rejected the token (401). Check the PAT.")
        if resp.status_code == 404:
            raise GitHubError(
                f"GitHub returned 404 for {path}. The resource doesn't exist "
                "or the PAT lacks access to this repository."
            )
        if not resp.ok:
            raise GitHubError(f"GitHub API error {resp.status_code} for {path}: {resp.text[:300]}")
        return resp

    # -- pull request -------------------------------------------------------

    def get_pull_request(self, pr: PullRequestRef) -> dict:
        return self._get(f"/repos/{pr.slug}/pulls/{pr.number}").json()

    def list_pr_files(self, pr: PullRequestRef) -> list[dict]:
        """All changed files with their patches, following pagination."""
        files: list[dict] = []
        page = 1
        while True:
            batch = self._get(
                f"/repos/{pr.slug}/pulls/{pr.number}/files", per_page=100, page=page
            ).json()
            files.extend(batch)
            if len(batch) < 100:
                return files
            page += 1

    # -- repository context -------------------------------------------------

    def get_file_content(self, pr: PullRequestRef, path: str, ref: str) -> str | None:
        """Decoded text content of a file at a ref, or None if binary/missing."""
        try:
            data = self._get(f"/repos/{pr.slug}/contents/{path}", ref=ref).json()
        except GitHubError:
            return None
        if isinstance(data, list) or data.get("encoding") != "base64":
            return None
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (UnicodeDecodeError, ValueError):
            return None

    def get_readme(self, pr: PullRequestRef, ref: str) -> str | None:
        try:
            data = self._get(f"/repos/{pr.slug}/readme", ref=ref).json()
        except GitHubError:
            return None
        try:
            return base64.b64decode(data["content"]).decode("utf-8")
        except (UnicodeDecodeError, ValueError, KeyError):
            return None

    def get_tree_paths(self, pr: PullRequestRef, ref: str, limit: int = 400) -> list[str]:
        """Repo file listing at a ref (truncated to keep prompts bounded)."""
        try:
            data = self._get(f"/repos/{pr.slug}/git/trees/{ref}", recursive="1").json()
        except GitHubError:
            return []
        paths = [e["path"] for e in data.get("tree", []) if e.get("type") == "blob"]
        return paths[:limit]

    # -- posting the review ---------------------------------------------------

    def post_review(self, pr: PullRequestRef, body: str, comments: list[dict]) -> dict:
        """Post a PR review with line comments; falls back to a body-only
        review if GitHub rejects the comment anchors."""
        payload: dict = {"event": "COMMENT", "body": body}
        if comments:
            payload["comments"] = comments
        resp = self.session.post(
            f"{self.api_root}/repos/{pr.slug}/pulls/{pr.number}/reviews",
            json=payload,
            timeout=30,
        )
        if resp.status_code == 422 and comments:
            # Anchors should always be valid (they are grounded against the
            # diff), but never lose the review text if GitHub disagrees.
            resp = self.session.post(
                f"{self.api_root}/repos/{pr.slug}/pulls/{pr.number}/reviews",
                json={"event": "COMMENT", "body": body},
                timeout=30,
            )
        if not resp.ok:
            raise GitHubError(f"Failed to post review: {resp.status_code} {resp.text[:300]}")
        return resp.json()
