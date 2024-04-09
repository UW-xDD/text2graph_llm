import json
import logging
import os
from enum import Enum
from functools import partial

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from .alignment import AlignmentHandler
from .prompt import PromptHandler, to_handler
from .schema import GraphOutput, RelationshipTriplet, Stratigraphy, Provenance

load_dotenv()


class OpenSourceModel(Enum):
    """Supported open-source language models via Ollama."""

    MIXTRAL = "mixtral"
    OPENHERMES = "openhermes"


class OpenAIModel(Enum):
    GPT3T = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4T = "gpt-4-turbo-preview"


class AnthropicModel(Enum):
    CLAUDE3OPUS = "claude-3-opus-20240229"
    CLAUDE3SONNET = "claude-3-sonnet-20240229"
    CLAUDE3HAIKU = "claude-3-haiku-20240307"


def to_model(model: str) -> OpenSourceModel | OpenAIModel | AnthropicModel:
    """Convert string to model enum."""
    if model in [model.value for model in OpenSourceModel]:
        return OpenSourceModel(model)
    elif model in [model.value for model in OpenAIModel]:
        return OpenAIModel(model)
    elif model in [model.value for model in AnthropicModel]:
        return AnthropicModel(model)
    else:
        raise ValueError(f"Model '{model}' is not supported.")


async def ask_llm(
    text: str,
    prompt_handler: PromptHandler | str = "v3",
    model: OpenSourceModel | OpenAIModel | AnthropicModel | str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    to_triplets: bool = True,
    alignment_handler: AlignmentHandler | None = None,
    doc_ids: list[str] | None = None,
) -> str | GraphOutput:
    """Ask model with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]
    """
    if not doc_ids:
        doc_ids = []

    # Convert model string to enum
    if isinstance(model, str):
        model = to_model(model)

    # Convert prompt handler string to object
    if isinstance(prompt_handler, str):
        prompt_handler = to_handler(prompt_handler)

    messages = prompt_handler.get_gpt_messages(text)

    if isinstance(model, OpenSourceModel):
        raw_output = query_ollama(model, messages, temperature)

    if isinstance(model, OpenAIModel):
        raw_output = query_openai(model, messages, temperature)

    if isinstance(model, AnthropicModel):
        raw_output = query_anthropic(model, messages, temperature)

    if not to_triplets:
        return raw_output

    ask_llm_provenance = Provenance(
        source_name=model.__class__.__name__,
        source_version=model.name,
        additional_values=dict(
            temperature=temperature, prompt=prompt_handler.version, doc_ids=doc_ids
        ),
    )

    return await post_process(
        raw_llm_output=raw_output,
        prompt_handler=prompt_handler,
        alignment_handler=alignment_handler,
        llm_provenance=ask_llm_provenance,
    )


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


def to_triplet(
    triplet: dict,
    subject_key: str,
    object_key: str,
    predicate_key: str,
    llm_provenance: Provenance,
) -> RelationshipTriplet:
    """Inject attributes into RelationshipTriples model. Must be {"subject": "x", "object": "y", "predicate": "z"} tuple."""
    return RelationshipTriplet(
        subject=triplet[subject_key],
        object=triplet[object_key],
        predicate=triplet[predicate_key],
        provenance=llm_provenance,
    )


async def post_process(
    raw_llm_output: str,
    prompt_handler: PromptHandler,
    alignment_handler: AlignmentHandler | None = None,
    threshold: float = 0.95,
    llm_provenance: Provenance | None = None,
) -> GraphOutput:
    """Post-process raw output to GraphOutput model."""
    triplets = json.loads(raw_llm_output)

    if "triplets" not in triplets:
        raise ValueError("Response does not contain 'triplets' key.")

    triplet_format_func = partial(
        to_triplet,
        subject_key=prompt_handler.subject_key,
        object_key=prompt_handler.object_key,
        predicate_key=prompt_handler.predicate_key,
        llm_provenance=llm_provenance,
    )

    triplets = [triplet_format_func(triplet) for triplet in triplets["triplets"]]

    if alignment_handler:
        for triplet in triplets:
            # Only apply to strat_name because location is not in Macrostrat
            name = triplet.object.strat_name
            closest = alignment_handler.get_closest_known_entity(
                name, threshold=threshold
            )

            # Update triplet object if closest known entity is different
            if closest != name:
                logging.info("Swaping", name, "with", closest)
                triplet.object = Stratigraphy(strat_name=closest)
                triplet.object.provenance.additional_values[
                    "alignment_old_strat_name"
                ] = name
                triplet.object.provenance.additional_values[
                    "alignment_handler_model"
                ] = alignment_handler.model_name

    output = GraphOutput(triplets=triplets)
    await output.hydrate()
    return output
