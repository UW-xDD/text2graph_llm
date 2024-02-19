import json
import logging
import os
from enum import Enum

import requests
from dotenv import load_dotenv
from openai import OpenAI


from .prompt import V0Prompt, IainPrompt

load_dotenv()


class OpenSourceModel(Enum):
    """Supported open-source language models."""

    MIXTRAL = "mixtral"
    OPENHERMES = "openhermes"


class OpenAIModel(Enum):
    GPT3T = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4T = "gpt-4-turbo-preview"


PROMPT_MAPPING = {
    "mixtral": V0Prompt,
    "openhermes": V0Prompt,
    "gpt-3.5-turbo": IainPrompt,
    "gpt-4": IainPrompt,
    "gpt-4-turbo-preview": IainPrompt
}


def ask_llm(
    messages: list[dict],
    model: OpenSourceModel | OpenAIModel | str = "gpt-3.5-turbo",
    temperature: float = 0.0,
) -> dict:
    """Ask model with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]
    """

    # Validate supported models
    if isinstance(model, str):
        if model in [model.value for model in OpenSourceModel]:
            model = OpenSourceModel(model)
        elif model in [model.value for model in OpenAIModel]:
            model = OpenAIModel(model)
        else:
            raise ValueError(f"Model '{model}' is not supported.")

    if isinstance(model, OpenSourceModel):
        return query_ollama(model, messages, temperature)

    if isinstance(model, OpenAIModel):
        return query_openai(model, messages, temperature)


def query_openai(
    model: OpenAIModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query OpenAI API for language model completion."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model.value, messages=messages, temperature=temperature, stream=False
    )
    return completion.choices[0].message.content


def query_ollama(
    model: OpenSourceModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query self-hosted OLLAMA for language model completion."""
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
def llm_graph(text: str, model: str) -> str:
    """Core function for llm_graph endpoint."""
    prompt_creator = PROMPT_MAPPING[model]()
    messages = prompt_creator.get_messages(text)
    raw_response = ask_llm(messages, model)

    # Post-process response
    try:
        contents = json.loads(raw_response)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode response: {raw_response}")
        contents = raw_response

    return {"locations": contents}
