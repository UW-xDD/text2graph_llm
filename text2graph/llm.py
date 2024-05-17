import json
import logging
import os
import sqlite3
from enum import Enum
from functools import partial

import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from requests.auth import HTTPBasicAuth

from text2graph.alignment import (
    AlignmentHandler,
)
from text2graph.askxdd import Retriever
from text2graph.geolocation.geocode import RateLimitedClient
from text2graph.gkm.gkm import to_ttl
from text2graph.macrostrat import EntityType
from text2graph.prompt import PromptHandler, get_prompt_handler
from text2graph.schema import (
    GraphOutput,
    Location,
    Mineral,
    Provenance,
    RelationshipTriplet,
    Stratigraphy,
)

load_dotenv()


class OpenSourceModel(Enum):
    """Supported open-source language models via Ollama."""

    MIXTRAL = "mixtral"
    OPENHERMES = "openhermes"


class OpenAIModel(Enum):
    GPT3T = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4T = "gpt-4-turbo-preview"
    GPTO = "gpt-4o"


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


def query_openai(
    model: OpenAIModel, messages: list[dict[str, str]], temperature: float = 0.0
) -> str:
    """Query OpenAI API for language model completion."""
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model.value,
        response_format={"type": "json_object"},
        messages=messages,  # type: ignore
        temperature=temperature,
        stream=False,
    )  # type: ignore
    return completion.choices[0].message.content


def query_local_ollama(
    model: OpenSourceModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query self-hosted OLLAMA for language model completion."""
    url = os.getenv("OLLAMA_URL")
    if not url:
        raise ValueError("OLLAMA_URL is not set.")
    user = os.getenv("OLLAMA_USER", "")
    password = os.getenv("OLLAMA_PASSWORD", "")
    data = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "format": "json",
    }
    # Non-streaming mode
    response = requests.post(url, auth=HTTPBasicAuth(user, password), json=data)
    response.raise_for_status()
    return response.json()["message"]["content"]


def query_llm_queue(
    model: OpenSourceModel, messages: list[dict], temperature: float = 0.0
) -> str:
    """Query CHTC Ollama proxy."""

    CHTC_LLM_API_URL = (
        f"http://{os.getenv('CHTC_LLM_HOST')}:{os.getenv('CHTC_LLM_PORT')}"
    )
    auth_headers = {"Api-Key": os.getenv("CHTC_LLM_API_KEY")}

    # This is vanilla Ollama API style
    data = {
        "model": model.value,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
        "format": "json",
    }
    # Non-streaming mode
    response = requests.post(
        f"{CHTC_LLM_API_URL}/api/chat", headers=auth_headers, json=data
    )
    response.raise_for_status()
    return (
        json.loads(response.json()["_content"])["message"]["content"]
        .strip()
        .replace("\\", "")
    )


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
    prompt_handler: PromptHandler,
    llm_provenance: Provenance | None = None,
) -> RelationshipTriplet:
    """Inject attributes into RelationshipTriples model. Must be {"subject": "x", "object": "y", "predicate": "z"} tuple."""

    # TODO: Probably should handle all subject, object, and predicate more gracefully
    type_map = {
        EntityType.STRAT_NAME: Stratigraphy,
        EntityType.MINERAL: Mineral,
    }

    s = triplet[prompt_handler.subject_key]
    o = triplet[prompt_handler.object_key]
    p = triplet[prompt_handler.predicate_key]

    return RelationshipTriplet(
        subject=Location(name=s),
        object=type_map[prompt_handler.object_entity_type](name=o),
        predicate=p,
        provenance=llm_provenance,
    )


async def post_process(
    raw_llm_output: str,
    prompt_handler: PromptHandler,
    alignment_handler: AlignmentHandler | None = None,
    threshold: float = 0.95,
    hydrate: bool = True,
    provenance: Provenance | None = None,
) -> GraphOutput:
    """Post-process raw output to GraphOutput model."""
    triplets = json.loads(raw_llm_output)

    # Handle different response formats form different LLMs
    if "triplets" not in triplets:
        logging.info(f"unexpected triplet format: {triplets}, attempting to fix.")
        triplets = {"triplets": triplets}

    triplet_format_func = partial(
        to_triplet,
        prompt_handler=prompt_handler,
        llm_provenance=provenance,
    )

    try:
        # Convert triplets to RelationshipTriplet objects
        safe_triplets = []
        for triplet in triplets["triplets"]:
            try:
                safe_triplets.append(triplet_format_func(triplet))
            except ValidationError:
                logging.error(f"ValidationError: when converting {triplet}")

    except KeyError:
        logging.info(f"unexpected triplet format: {triplets}")
        raise ValueError("Unexpected triplet format")

    if alignment_handler:
        for triplet in safe_triplets:
            # Only apply to strat_name because location is not in Macrostrat
            name = triplet.object.name
            closest = alignment_handler.get_closest_known_entity(
                name, threshold=threshold
            )

            # Update triplet object if closest known entity is different from original
            if closest != name:
                logging.info("Swapping", name, "with", closest)
                triplet.object = Stratigraphy(name=closest)

    output = GraphOutput(triplets=safe_triplets)
    if hydrate:
        await output.hydrate(
            client=RateLimitedClient(interval=1.5, count=1, timeout=30)
        )
    return output


async def ask_llm(
    text: str,
    prompt_handler: PromptHandler | str = "stratname_v3",
    alignment_handler: AlignmentHandler | None = None,
    model: OpenSourceModel | OpenAIModel | AnthropicModel | str = "gpt-3.5-turbo",
    temperature: float = 0.0,
    to_triplets: bool = True,  # For debugging intermediate steps
    doc_ids: list[str] | None = None,
    hydrate: bool = True,
    provenance: Provenance | None = None,
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
        prompt_handler = get_prompt_handler(prompt_handler)

    messages = prompt_handler.get_gpt_messages(text)

    use_chtc = int(os.getenv("USE_LLM_QUEUE", 0))

    if isinstance(model, OpenSourceModel):
        if use_chtc:
            raw_output = query_llm_queue(model, messages, temperature)
        else:
            raw_output = query_local_ollama(model, messages, temperature)

    if isinstance(model, OpenAIModel):
        raw_output = query_openai(model, messages, temperature)

    if isinstance(model, AnthropicModel):
        raw_output = query_anthropic(model, messages, temperature)

    logging.debug(f"Raw llm output: {raw_output}")

    if not to_triplets:
        return raw_output

    ask_llm_provenance = Provenance(
        source_name=model.__class__.__name__,
        source_version=model.value,
        additional_values=dict(
            temperature=temperature,
            prompt=prompt_handler.version,
            doc_ids=doc_ids,
        ),
        previous=provenance,
    )

    logging.debug(f"Provenance: {ask_llm_provenance}")
    return await post_process(
        raw_llm_output=raw_output,
        prompt_handler=prompt_handler,
        alignment_handler=alignment_handler,
        hydrate=hydrate,
        provenance=ask_llm_provenance,
    )


async def llm_graph_from_search(
    query: str,
    top_k: int,
    model: str,
    alignment_handler: AlignmentHandler,
    prompt_handler: PromptHandler,
    hydrate: bool = False,
    ttl: bool = True,
) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search."""

    r = Retriever()
    paragraphs = r.query(query, top_k=top_k)
    graphs = []
    for paragraph in paragraphs:
        graph = await ask_llm(
            text=paragraph.text_content,
            prompt_handler=prompt_handler,
            alignment_handler=alignment_handler,
            model=model,
            temperature=0.0,
            to_triplets=True,
            doc_ids=[
                paragraph.paper_id
            ],  # TODO: Confirm with Iain if this is the correct usage. It's unclear why a paragraph from one document requires a list.
            provenance=paragraph.provenance,
            hydrate=hydrate,
        )

        # Add paragraph level information
        assert isinstance(graph, GraphOutput)
        graph.id = paragraph.id
        graph.paper_id = paragraph.paper_id
        graph.hashed_text = paragraph.hashed_text
        graph.text_content = paragraph.text_content

        graphs.append(graph)

    logging.info(paragraphs)

    if not ttl:
        return graphs

    return [to_ttl(graph) for graph in graphs]


def get_graph_from_cache(
    ids: list[str] | tuple[str],
) -> list[GraphOutput]:
    """Get cached graph from sqlite database."""

    GRAPH_CACHE = os.getenv("GRAPH_SQLITE")
    assert GRAPH_CACHE is not None, "GRAPH_SQLITE environment variable must be set."

    with sqlite3.connect(GRAPH_CACHE) as conn:
        cursor = conn.cursor()
        formatted_ids = ", ".join([f"'{id}'" for id in ids])
        cursor.execute(
            f"SELECT id, paper_id, hashed_text, triplets FROM triplets WHERE id IN ({formatted_ids});"
        )
        rows = cursor.fetchall()

    # Validate to GraphOutput
    graphs = []
    for row in rows:
        try:
            triplets = json.loads(row[3])["triplets"]
        except json.JSONDecodeError:
            logging.error(f"Error loading graph from cache: {row}")
            continue

        data = {
            "id": row[0],
            "paper_id": row[1],
            "hashed_text": row[2],
            "triplets": triplets,
        }

        try:
            graphs.append(GraphOutput.model_validate(data))
        except Exception as e:
            logging.error(f"Error loading graph from cache: {e}")
            pass

    return graphs


async def fast_llm_graph_from_search(
    query: str,
    top_k: int,
    ttl: bool = True,
    hydrate: bool = False,
    with_text: bool = False,
) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search using locally cached."""

    r = Retriever()
    paragraphs = r.query(query, top_k=top_k)
    id2text = {paragraph.id: paragraph.text_content for paragraph in paragraphs}

    graphs = get_graph_from_cache([paragraph.id for paragraph in paragraphs])

    gps_client = RateLimitedClient(interval=1.2)
    for graph in graphs:
        # Get text
        if with_text and graph.id:
            try:
                graph.text_content = id2text[graph.id]
            except KeyError:
                continue
        # Hydrate
        if hydrate:
            await graph.hydrate(client=gps_client)

    if not ttl:
        return graphs

    # TTL returns
    return [to_ttl(graph) for graph in graphs]
