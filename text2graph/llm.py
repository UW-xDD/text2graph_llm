import requests
import os
from enum import Enum
from dotenv import load_dotenv
load_dotenv()

class OpenSourceModel(Enum):
    """Supported open-source language models."""
    MIXTRAL = "mixtral"
    OPENHERMES = "openhermes"

def ask_llm(messages: list[dict], model: OpenSourceModel | str = "mixtral", temperature: float = 0.0) -> dict:
    """Ask model with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]
    """
    # Validate supported models
    if isinstance(model, str):
        try:
            model = OpenSourceModel(model.lower())
        except ValueError:
            raise ValueError(f"Model '{model}' is not supported.")

    url = os.getenv("OLLAMA_URL")
    user = os.getenv("OLLAMA_USER")
    password = os.getenv("OLLAMA_PASSWORD")
    data = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    # Non-streaming mode
    response = requests.post(url, auth=requests.auth.HTTPBasicAuth(user, password), json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]