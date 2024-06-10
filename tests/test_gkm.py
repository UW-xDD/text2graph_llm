import pytest
from rdflib import Graph
from rdflib.compare import to_isomorphic, IsomorphicGraph
from datetime import timezone, datetime

from text2graph.schema import (
    RelationshipTriplet,
    Location,
    Mineral,
    Stratigraphy,
    Provenance,
)
from text2graph.gkm.convert import triplet_to_rdf


def isomorphicgraph_from_ttl_file(filename: str) -> IsomorphicGraph:
    with open(filename, "r") as f:
        ttl_lines = f.readlines()

    return to_isomorphic(Graph().parse(format="ttl", data="".join(ttl_lines)))


@pytest.fixture
def mineral_triplet() -> RelationshipTriplet:
    return RelationshipTriplet(
        subject=Location(
            name="Margnac Mine",
            lat=45.98611,
            lon=1.30833,
            provenance=Provenance(
                source_name="geocodingAPI",
                source_url="https://geocoder.api/",
                source_version="v1.1",
                requested=datetime(2024, 5, 29, 20, 1, 46, 967742, tzinfo=timezone.utc),
            ),
        ),
        predicate="contains",
        object=Mineral(
            mineral="Agrinierite",
            mineral_id=58,
            formula="K2(Ca,Sr)[(UO2)3O3(OH)2]2·5H2O",
            provenance=Provenance(
                source_name="macrostrat",
                source_url="https://macrostrat.org/api/defs/minerals?mineral_id=58",
                source_version="v2",
                requested=datetime(2024, 5, 29, 20, 1, 46, 967742, tzinfo=timezone.utc),
            ),
        ),
    )


@pytest.fixture
def stratigraphy_triplet() -> RelationshipTriplet:
    macrostrat_waldron_shale_dct = {
        "strat_name": "Waldron Shale",
        "strat_name_long": "Waldron Shale",
        "rank": "Fm",
        "strat_name_id": 4260,
        "concept_id": 4273,
        "bed": "",
        "bed_id": 0,
        "mbr": "",
        "mbr_id": 0,
        "fm": "Waldron Shale",
        "fm_id": 4260,
        "subgp": "",
        "subgp_id": 0,
        "gp": "Wayne",
        "gp_id": 2700,
        "sgp": "",
        "sgp_id": 0,
        "b_age": 429.65,
        "t_age": 427.4,
        "b_period": "Silurian",
        "t_period": "Silurian",
        "c_interval": "",
        "t_units": 9,
        "ref_id": 1,
        "macrostrat_version": 2,
    }

    return RelationshipTriplet(
        subject=Location(
            name="Arkabulta and Franks Rd, MI",
            lat=34.685,
            lon=-90.146,
            provenance=Provenance(
                source_name="geocodingAPI",
                source_url="https://geocoder.api/",
                source_version="v1.1",
                requested=datetime(2024, 5, 20, 10, 00),
            ),
        ),
        predicate="is found near",
        object=Stratigraphy(
            **macrostrat_waldron_shale_dct,
            provenance=Provenance(
                source_name="macrostrat",
                source_url="https://macrostrat.org/api/defs/strat_names?strat_name_id=4260",
                source_version="v2",
                requested=datetime(2024, 5, 20, 10, 00),
            ),
        ),
    )


@pytest.mark.parametrize(
    "filename, triplet",
    [
        ("tests/fixtures/test_triplet_to_rdf_mineral.ttl", "mineral_triplet"),
        ("tests/fixtures/test_triplet_to_rdf_stratigraphy.ttl", "stratigraphy_triplet"),
    ],
)
def test_triplet_to_rdf(filename, triplet, request):
    expected = isomorphicgraph_from_ttl_file(filename)
    test = to_isomorphic(triplet_to_rdf(request.getfixturevalue(triplet)))
    assert test == expected
