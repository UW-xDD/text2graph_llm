import pytest
import asyncio

from text2graph.alignment import get_alignment_handler
from text2graph.prompt import get_prompt_handler
from text2graph.llm import llm_graph_from_search, fast_llm_graph_from_search


@pytest.mark.slow()
def test_fast_llm_graph_from_search_empty_result_creates_empty_triplet():
    y = asyncio.run(
        fast_llm_graph_from_search("iron mines in Minnesota", top_k=3, ttl=False)
    )
    assert not [x.triplets for x in y if not x.triplets][0]


@pytest.mark.slow()
def test_llm_graph_from_search_to_ttl_creates_string_for_all_triplets():
    y = asyncio.run(
        llm_graph_from_search(
            query="Mines in Minnesota, including iron, aluminium, and copper.",
            top_k=3,
            prompt_handler=get_prompt_handler("mineral_v0"),
            alignment_handler=get_alignment_handler("mineral"),
            model="mixtral",
            hydrate=False,
        )
    )
    assert all([isinstance(x, str) for x in y])
