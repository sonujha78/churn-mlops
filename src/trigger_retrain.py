"""
Simulates what a Prometheus Alertmanager webhook receiver would do:
when the HighDataDrift alert fires, this script calls the GitHub API
to trigger the automated retraining workflow via repository_dispatch.

In a full production setup, Alertmanager would be configured with a
webhook receiver that calls this automatically. Here we trigger it
manually to demonstrate the same end-to-end flow.
"""
import os
import sys
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "sonujha78/churn-mlops"

def trigger_retraining(reason="manual_test"):
    if not GITHUB_TOKEN:
        print("ERROR: GITHUB_TOKEN environment variable not set.")
        sys.exit(1)

    url = f"https://api.github.com/repos/{REPO}/dispatches"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
    }
    payload = {
        "event_type": "drift_alert",
        "client_payload": {"reason": reason}
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 204:
        print(f"✅ Successfully triggered retraining workflow (reason: {reason})")
    else:
        print(f"❌ Failed to trigger workflow: {response.status_code} - {response.text}")


if __name__ == "__main__":
    reason = sys.argv[1] if len(sys.argv) > 1 else "manual_test"
    trigger_retraining(reason)
