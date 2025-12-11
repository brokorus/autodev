import os
import requests


def _get_token() -> str:
    token = (
        os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
        or os.getenv("GITHUB_PAT")
    )
    if not token:
        raise RuntimeError(
            "No GitHub token found in env (GITHUB_TOKEN / GH_TOKEN / GITHUB_PAT)."
        )
    return token


def last_workflow_status(repo: str) -> dict:
    """
    Return a summary of the most recent GitHub Actions workflow run
    for the given repo (format: 'owner/name').
    """
    url = f"https://api.github.com/repos/{repo}/actions/runs?per_page=1"
    headers = {
        "Authorization": f"Bearer {_get_token()}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.get(url, headers=headers, timeout=15)
    if resp.status_code != 200:
        return {"error": f"GitHub API status {resp.status_code}", "body": resp.text}
    data = resp.json()
    runs = data.get("workflow_runs") or []
    if not runs:
        return {"error": "no workflow runs"}
    run = runs[0]
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "html_url": run.get("html_url"),
        "workflow_id": run.get("workflow_id"),
    }
