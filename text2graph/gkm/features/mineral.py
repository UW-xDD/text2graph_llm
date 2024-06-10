from rdflib import Graph, Literal, RDF, RDFS, URIRef

from text2graph.schema import Mineral, RelationshipTriplet
from text2graph.gkm.namespace import GSOG, GSRM, XDD
from text2graph.gkm.features.general import entity_name, add_macrostrat_query_and_entity


def object_node_mineral(triplet_object: Mineral) -> URIRef:
    """
    create mineral node/URIRef for object/mineral dict
    :param triplet_object: subject/mineral dict
    :return: URIRef
    """
    object_name = entity_name(triplet_object.mineral)
    return URIRef(object_name, XDD)


def mineral_type(g: Graph, triplet: RelationshipTriplet, object_node: URIRef) -> Graph:
    """
    add object node to the graph with type as mineral type
    """
    g.add((object_node, RDF.type, GSOG["Rock_Material"]))
    g.add((object_node, RDF.type, GSRM[entity_name(triplet.object.mineral)]))
    g.add(
        (
            object_node,
            RDFS.label,
            Literal(
                f"{triplet.object.name}, {triplet.object.mineral_type}, {triplet.object.formula}",
                lang="en",
            ),
        )
    )
    g = add_macrostrat_query_and_entity(g, triplet, object_node)
    return g
