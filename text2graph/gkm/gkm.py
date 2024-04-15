import logging
from pathlib import Path
from typing import Protocol

from pydantic import ValidationError
from rdflib import Graph, URIRef

from text2graph.gkm.features import (
    default_rdf_graph,
    define_object_node,
    deposition_age,
    spatial_location,
    stratigraphic_label,
    stratigraphic_rank_relations,
    stratigraphic_type,
    time_span,
    triplet_provenance,
)
from text2graph.schema import GraphOutput, RelationshipTriplet


class GraphFeatureGenerator(Protocol):
    def __call__(
        self, g: Graph, triplet: RelationshipTriplet, object_node: URIRef
    ) -> Graph: ...


def triplet_to_rdf(triplet: dict | RelationshipTriplet) -> Graph | None:
    """
    Convert RelationshipTriples object to an RDF graph
    :param triplet: RelationshipTriples object to convert
    :return: RDF graph of RelationshipTriples object
    """
    if isinstance(triplet, dict):
        try:
            triplet = RelationshipTriplet(**triplet)
        except ValidationError:
            logging.warning(f"failed to validate RelationshipTriplet from: {triplet}")
            return

    g = default_rdf_graph()
    object_node = define_object_node(triplet_object=triplet.object)
    graph_features: list[GraphFeatureGenerator] = [
        stratigraphic_type,
        stratigraphic_label,
        triplet_provenance,
        spatial_location,
        stratigraphic_rank_relations,
        deposition_age,
        time_span,
    ]
    for feat in graph_features:
        try:
            g = feat(g=g, triplet=triplet, object_node=object_node)
        except Exception as e:
            logging.info(
                f"failed to add {feat.__name__} to graph with error:{e} for {triplet=}"
            )
    return g


def graph_to_ttl_string(g: Graph, filename: Path | None = None) -> str:
    """
    serialize graph to RDF Turtle (ttl) string
    :param g: graph to serialize
    :param filename: Path or None, if Path write TTL to disk at given filename
    :return str: serialized graph
    """
    output = g.serialize(format="turtle")
    if filename:
        with open(filename, "w") as f:
            f.write(output)
    return output


def to_ttl(graph: GraphOutput) -> str:
    """Convert GraphOutput object to TTL string."""

    ttls = []
    for triplet in graph.triplets:
        try:
            g = triplet_to_rdf(triplet)
            ttl = graph_to_ttl_string(g)
            ttls.append(ttl)
        except Exception as e:
            logging.error(f"Error converting triplet to RDF: {e}")
            continue

    return "\n".join(ttls)
