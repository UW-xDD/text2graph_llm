import text2graph.llm as llm
from text2graph.alignment import EntityType, get_alignment_handler
from text2graph.prompt import StratPromptHandlerV3
from text2graph.schema import GraphOutput


async def llm_graph(text: str, model: str):
    """Business logic layer for llm graph extraction."""

    return await llm.ask_llm(
        text=text,
        prompt_handler=StratPromptHandlerV3(),
        alignment_handler=get_alignment_handler(EntityType.STRAT_NAME),
        model=model,
        temperature=0.0,
        to_triplets=True,
    )


async def slow_llm_graph_from_search(**kwargs) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search."""
    # Add more API customization logic here if needed.
    return await llm.llm_graph_from_search(model="mixtral", **kwargs)


async def fast_llm_graph_from_search(**kwargs) -> list[str] | list[GraphOutput]:
    """Business logic layer for llm graph extraction from search using cached graph."""
    # Add more API customization logic here if needed.
    return await llm.fast_llm_graph_from_search(**kwargs)
