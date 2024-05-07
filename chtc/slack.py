import os
import subprocess

import slack_sdk
from dotenv import load_dotenv

load_dotenv()


def send_slack_message(channel: str, message: str) -> None:
    """Send a message to TQDM Slack channel for monitoring."""

    TQDM_SLACK_TOKEN = os.getenv("TQDM_SLACK_TOKEN")
    client = slack_sdk.WebClient(TQDM_SLACK_TOKEN)

    client.chat_postMessage(
        channel=channel,
        text=message,
    )


def run_bash_command(command: str):
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"


def main():
    """Monitor condor_q and send a message to Slack channel."""
    output = run_bash_command("condor_q")
    send_slack_message(channel="#htcondor", message=output)


if __name__ == "__main__":
    main()
