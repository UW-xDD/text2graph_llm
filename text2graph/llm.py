import requests
import os
from dotenv import load_dotenv
load_dotenv()

def ask_mixtral(messages: list[dict]) -> dict:
    """Ask mixtral with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]
    """
    url = os.getenv("MIXTRAL_URL")
    user = os.getenv("MIXTRAL_USER")
    password = os.getenv("MIXTRAL_PASSWORD")
    data = {
        "model": "mixtral",
        "messages": messages,
        "stream": False,  # set to True to get a stream of responses token-by-token
    }
    # Non-streaming mode
    response = requests.post(
        url, auth=requests.auth.HTTPBasicAuth(user, password), json=data
    )
    response.raise_for_status()
    return response.json()["message"]["content"]