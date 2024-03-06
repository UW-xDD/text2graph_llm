import json
import logging
import os
from enum import Enum

import requests
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

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


class AnthropicModel(Enum):
    CLAUDE3OPUS = "claude-3-opus-20240229"
    CLAUDE3SONNET = "claude-3-sonnet-20240229"
    CLAUDE3HAIKU = "claude-3-haiku-xxxx"


AVAILABLE_PROMPTS = {"v0": V0Prompt, "v1": IainPrompt, "latest": IainPrompt}


def ask_llm(
    messages: list[dict],
    model: OpenSourceModel | OpenAIModel | AnthropicModel | str = "gpt-3.5-turbo",
    temperature: float = 0.0,
) -> str:
    """Ask model with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]
    """

    # Validate supported models
    if isinstance(model, str):
        if model in [model.value for model in OpenSourceModel]:
            model = OpenSourceModel(model)
        elif model in [model.value for model in OpenAIModel]:
            model = OpenAIModel(model)
        elif model in [model.value for model in AnthropicModel]:
            model = AnthropicModel(model)
        else:
            raise ValueError(f"Model '{model}' is not supported.")

    if isinstance(model, OpenSourceModel):
        return query_ollama(model, messages, temperature)

    if isinstance(model, OpenAIModel):
        return query_openai(model, messages, temperature)

    if isinstance(model, AnthropicModel):
        return query_anthropic(model, messages, temperature)


def query_openai(
    model: OpenAIModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query OpenAI API for language model completion."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model.value,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=temperature,
        stream=False,
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
        "format": "json",
    }
    # Non-streaming mode
    response = requests.post(
        url, auth=requests.auth.HTTPBasicAuth(user, password), json=data
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def query_anthropic(
    model: AnthropicModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query Anthropic for language model completion."""

    client = Anthropic()

    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
            messages.remove(message)

    response = client.messages.create(
        model=model.value,
        max_tokens=4096,
        messages=messages,
        system=system_message,
        temperature=temperature,
        stream=False,
    )

    return response.content[0].text


# API layer function logic
def llm_graph(text: str, model: str, prompt_version: str = "latest") -> dict:
    """Core function for llm_graph endpoint."""

    logging.info(f"Querying model '{model}' with prompt version '{prompt_version}'")

    # Create prompt and messages format
    prompt_creator = AVAILABLE_PROMPTS[prompt_version]()
    messages = prompt_creator.get_messages(text)

    # Query language model
    raw_response = ask_llm(messages, model)

    # Post-process response
    try:
        contents = json.loads(raw_response)
    except json.JSONDecodeError:
        logging.error(f"Failed to decode response: {raw_response}")
        contents = raw_response

    return contents
