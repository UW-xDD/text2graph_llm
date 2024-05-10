import asyncio

from text2graph.llm import fast_llm_graph_from_search
from text2graph.schema import GraphOutput


def test_fast_llm_graph_from_search_default():
    ttl = asyncio.run(
        fast_llm_graph_from_search(query="gold mines", top_k=2, hydrate=False, ttl=True)
    )
    assert isinstance(ttl, str)
    assert ttl.startswith("@prefix")


def test_fast_llm_graph_from_search_no_ttl():
    graph = asyncio.run(
        fast_llm_graph_from_search(
            query="gold mines", top_k=2, hydrate=False, ttl=False
        )
    )
    assert isinstance(graph, GraphOutput)
    assert len(graph.triplets) >= 0


# Test Hydration
# Caution: Hydration is super slow for larger graph.
def test_fast_llm_graph_from_search_hydrate():
    ttl = asyncio.run(
        fast_llm_graph_from_search(query="gold mines", top_k=1, hydrate=True, ttl=True)
    )
    assert isinstance(ttl, str)
    assert ttl.startswith("@prefix")
    assert "https://geocode.maps.co" in ttl
    assert "xdd:GeocodeAPIQuery" in ttl
