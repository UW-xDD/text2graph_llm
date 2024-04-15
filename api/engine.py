import logging
from pathlib import Path

from text2graph.alignment import AlignmentHandler
from text2graph.llm import ask_llm
from text2graph.macrostrat import get_all_strat_names
from text2graph.prompt import PromptHandlerV3


def generate_known_entity_embeddings() -> None:
    """Generate known entity embeddings for alignment."""

    if Path("data/known_entity_embeddings/all-MiniLM-L6-v2/model.txt").exists():
        logging.info("Known entity embeddings already exist.")
        return
    handler = AlignmentHandler(known_entity_names=get_all_strat_names())
    handler.save()


async def llm_graph(
    text: str,
    model: str,
):
    """Business logic layer for llm graph extraction."""

    return await ask_llm(
        text=text,
        prompt_handler=PromptHandlerV3(),
        model=model,
        temperature=0.0,
        to_triplets=True,
        alignment_handler=AlignmentHandler.load(
            "data/known_entity_embeddings/all-MiniLM-L6-v2"
        ),
    )
