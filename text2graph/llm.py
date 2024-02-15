import json
import logging
import os
from enum import Enum

import requests
from dotenv import load_dotenv

from .prompt import V0Prompt

load_dotenv()


class OpenSourceModel(Enum):
    """Supported open-source language models."""

    MIXTRAL = "mixtral"
    OPENHERMES = "openhermes"


PROMPT_MAPPING = {0: V0Prompt}


def ask_llm(
    messages: list[dict],
    model: OpenSourceModel | str = "mixtral",
    temperature: float = 0.0,
) -> dict:
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
    response = requests.post(
        url, auth=requests.auth.HTTPBasicAuth(user, password), json=data
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# API layer function logic
def llm_graph(text: str, model: str, version: int = 0) -> str:
    """Core function for llm_graph endpoint."""
    prompt_creator = PROMPT_MAPPING[version]()
    messages = prompt_creator.get_messages(text)
    raw_response = ask_llm(messages, model)

    # Post-process response
    try:
        contents = json.loads(raw_response)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode response: {raw_response}")
        contents = {}

    return {"locations": contents}
