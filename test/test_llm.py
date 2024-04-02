from text2graph.llm import ask_llm, post_process
from text2graph.schema import GraphOutput

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


def test_post_processor(raw_llm_output, prompt_handler_v3):
    graph = post_process(
        raw_llm_output=raw_llm_output, prompt_handler=prompt_handler_v3
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
        if triplet.object.strat_name == "Smithville":
            for k, v in SMITHVILLE.items():
                assert getattr(triplet.object, k) == v


def test_openai(text, prompt_handler_v3):
    raw_output = ask_llm(
        text=text,
        prompt_handler=prompt_handler_v3,
        model="gpt-3.5-turbo",
        to_triplets=False,
    )
    assert "Smithville" in raw_output


def test_anthropic(text, prompt_handler_v3):
    raw_output = ask_llm(
        text=text,
        prompt_handler=prompt_handler_v3,
        model="claude-3-haiku-20240307",
        to_triplets=False,
    )
    assert "Smithville" in raw_output
