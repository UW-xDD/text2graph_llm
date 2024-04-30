import text2graph.llm as llm
from text2graph.prompt import PromptHandlerV3


async def llm_graph(
    text: str,
    model: str,
):
    """Business logic layer for llm graph extraction."""

    return await llm.ask_llm(
        text=text,
        prompt_handler=PromptHandlerV3(),
        model=model,
        temperature=0.0,
        to_triplets=True,
    )


async def llm_graph_from_search(query: str, top_k: int, model: str, ttl: bool = True):
    """Business logic layer for llm graph extraction from search."""

    return await llm.llm_graph_from_search(
        query=query, top_k=top_k, model=model, ttl=ttl
    )
