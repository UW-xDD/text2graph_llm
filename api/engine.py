from pydantic import BaseModel

import text2graph.llm as llm
from text2graph.pipeline import ExtractionPipeline, get_handlers
from text2graph.schema import GraphOutput

# Data models


class TextToGraphRequest(BaseModel):
    text: str
    model: str
    extraction_pipeline: ExtractionPipeline


class SearchToGraphRequest(BaseModel):
    query: str
    top_k: int
    ttl: bool
    hydrate: bool
    extraction_pipeline: ExtractionPipeline


async def text_to_graph(text: str, model: str, extraction_pipeline: ExtractionPipeline):
    """Business logic layer for llm graph extraction."""

    prompt_handler, alignment_handler = get_handlers(extraction_pipeline)
    return await llm.ask_llm(
        text=text,
        prompt_handler=prompt_handler,
        alignment_handler=alignment_handler,
        model=model,
        temperature=0.0,
        to_triplets=True,
    )


async def search_to_graph_slow(**kwargs) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search."""
    # Add more API customization logic here if needed.

    extraction_pipeline = kwargs.pop("extraction_pipeline")
    prompt_handler, alignment_handler = get_handlers(extraction_pipeline)
    return await llm.llm_graph_from_search(
        model="mixtral",
        prompt_handler=prompt_handler,
        alignment_handler=alignment_handler,
        **kwargs,
    )


async def search_to_graph_fast(**kwargs) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search using cached graph."""
    # Add more API customization logic here if needed.

    extraction_pipeline = kwargs.pop("extraction_pipeline")
    if extraction_pipeline != ExtractionPipeline.LOCATION_STRATNAME:
        raise NotImplementedError(
            f"Fast search to graph only supports {ExtractionPipeline.LOCATION_STRATNAME}"
        )

    return await llm.fast_llm_graph_from_search(**kwargs)
