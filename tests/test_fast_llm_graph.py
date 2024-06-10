import pytest
import asyncio

from text2graph.llm import fast_llm_graph_from_search
from text2graph.schema import GraphOutput


@pytest.mark.slow()
def test_fast_llm_graph_from_search_default():
    ttls = asyncio.run(
        fast_llm_graph_from_search(query="gold mines", top_k=2, hydrate=False, ttl=True)
    )
    assert isinstance(ttls, list)

    ttl = ttls[0]
    assert isinstance(ttl, str)
    assert ttl.startswith("@prefix")


@pytest.mark.slow()
def test_fast_llm_graph_from_search_no_ttl():
    graphs = asyncio.run(
        fast_llm_graph_from_search(
            query="gold mines", top_k=2, hydrate=False, ttl=False
        )
    )

    assert isinstance(graphs, list)

    graph = graphs[0]
    assert isinstance(graph, GraphOutput)
    assert len(graph.triplets) >= 0


# Test Hydration
# Caution: Hydration is super slow for larger graph.
@pytest.mark.slow()
def test_fast_llm_graph_from_search_hydrate():
    ttls = asyncio.run(
        fast_llm_graph_from_search(query="gold mines", top_k=1, hydrate=True, ttl=True)
    )
    ttl = ttls[0]
    assert isinstance(ttl, str)
    assert ttl.startswith("@prefix")
    assert "https://geocode.maps.co" in ttl
    assert "xdd:GeocodeAPIQuery" in ttl


@pytest.mark.slow()
def test_with_text():
    graphs = asyncio.run(
        fast_llm_graph_from_search(
            query="gold mines", top_k=1, hydrate=False, ttl=False, with_text=True
        )
    )

    graph = graphs[0]
    assert isinstance(graph, GraphOutput)
    assert graph.text_content is not None
    assert len(graph.text_content) > 0
