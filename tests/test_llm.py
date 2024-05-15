import asyncio

from text2graph.llm import post_process
from text2graph.schema import GraphOutput, Stratigraphy

SMITHVILLE = {
    "strat_name": "Smithville",
    "strat_name_long": "Smithville Formation",
    "rank": "Fm",
    "strat_name_id": 5933,
    "concept_id": 3863,
    "bed": "",
    "bed_id": 0,
    "mbr": "",
    "mbr_id": 0,
    "fm": "Smithville",
    "fm_id": 5933,
    "subgp": "",
    "subgp_id": 0,
    "gp": "",
    "gp_id": 0,
    "sgp": "",
    "sgp_id": 0,
    "b_age": 471.2834,
    "t_age": 465.5,
    "b_period": "Ordovician",
    "t_period": "Ordovician",
    "c_interval": "",
    "t_units": 1,
    "ref_id": 1,
}


def test_post_processor(raw_llm_output, stratname_prompt_handler_v3):
    graph = asyncio.run(
        post_process(
            raw_llm_output=raw_llm_output, prompt_handler=stratname_prompt_handler_v3
        )
    )

    assert graph is not None
    assert isinstance(graph, GraphOutput)

    # check all triplets are RelationshipTriplet
    assert all(
        [
            triplet.__class__.__name__ == "RelationshipTriplet"
            for triplet in graph.triplets
        ]
    )

    # check all triplets have subject, object, and predicate
    assert all([triplet.subject is not None for triplet in graph.triplets])
    assert all([triplet.object is not None for triplet in graph.triplets])
    assert all([triplet.predicate is not None for triplet in graph.triplets])

    # Check `Smithville` is hydrated
    for triplet in graph.triplets:
        assert isinstance(triplet.object, Stratigraphy)
        if triplet.object.name == "Smithville":
            for k, v in SMITHVILLE.items():
                assert getattr(triplet.object, k) == v


def test_post_processor_with_alignment(
    raw_llm_output, stratname_prompt_handler_v3, stratname_alignment_handler
):
    graph = asyncio.run(
        post_process(
            raw_llm_output=raw_llm_output,
            prompt_handler=stratname_prompt_handler_v3,
            alignment_handler=stratname_alignment_handler,
        )
    )

    assert graph is not None
    assert isinstance(graph, GraphOutput)
