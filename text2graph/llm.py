import json
import logging
import asyncio
import os
from datetime import datetime
from enum import Enum

import requests
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic

from .prompt import V0Prompt, V1Prompt, V2Prompt
from .schema import RelationshipTriplet, GraphOutput, Provenance, ModelResponse

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


AVAILABLE_PROMPTS = {"v0": V0Prompt, "v1": V1Prompt, "v2": V2Prompt}


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
) -> ModelResponse:
    """Query OpenAI API for language model completion."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model.value,
        response_format={"type": "json_object"},
        messages=messages,
        temperature=temperature,
        stream=False,
    )
    return ModelResponse(
        result=completion.choices[0].message.content,
        # prompt=prompt,
        provenance=Provenance(
            source_name="OpenAI",
            source_url=None,
            source_version=model.value,
            requested=datetime.now(),
        ),
    )


def query_ollama(
    model: OpenSourceModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query self-hosted OLLAMA for language model completion."""
    url = os.getenv("OLLAMA_URL")
    if not url:
        raise ValueError("OLLAMA_URL is not set.")
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

    # Extract system message
    for message in messages:
        if message["role"] == "system":
            system_message = message["content"]
            messages.remove(message)
        else:
            system_message = None

    # Format user message
    user_messages = []
    for message in messages:
        if message["role"] == "user":
            user_messages.append(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": message["content"]}],
                }
            )

    kwargs = {
        "model": model.value,
        "max_tokens": 4096,
        "messages": user_messages,
        "temperature": temperature,
        "stream": False,
    }
    if system_message:
        kwargs["system"] = system_message

    response = client.messages.create(**kwargs)
    return response.content[0].text


# API layer function logic
def llm_graph(text: str, model: str, prompt_version: str = "latest") -> GraphOutput:
    """Core function for llm_graph endpoint."""

    logging.info(f"Querying model '{model}' with prompt version '{prompt_version}'")

    # Create prompt and messages format
    prompt_creator = AVAILABLE_PROMPTS[prompt_version]()
    messages = prompt_creator.get_messages(text)

    # Query language model
    triplets = ask_llm(messages, model)

    # Post-process response (must be JSON)
    triplets = json.loads(triplets)

    if "triplets" not in triplets:
        raise ValueError("Response does not contain 'triplets' key.")
    triplets = [to_triplet(triplet) for triplet in triplets["triplets"]]
    output = GraphOutput(triplets=triplets)
    asyncio.run(output.hydrate())
    return output


def to_triplet(triplet: dict) -> RelationshipTriplet:
    """Inject attributes into RelationshipTriples model. Must be {"subject": "x", "object": "y", "predicate": "z"} tuple."""
    return RelationshipTriplet(
        subject=triplet["subject"],
        object=triplet["object"],
        predicate=triplet["predicate"],
    )
