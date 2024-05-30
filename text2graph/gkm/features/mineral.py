from rdflib import Graph, Literal, RDF, RDFS, URIRef

from text2graph.schema import Mineral, RelationshipTriplet
from text2graph.gkm.namespace import GSOG, GSRM, XDD
from text2graph.gkm.features.general import add_macrostrat_query_and_entity


def mineral_name(m: Mineral) -> str:
    """
    create appropriate formatted entity name
    :param: m: Mineral object
    :return: formatted entity name string
    """
    return m.mineral.strip().title().replace(" ", "").replace('"', "")


def object_node_mineral(triplet_object: Mineral) -> URIRef:
    """
    create mineral node/URIRef for object/mineral dict
    :param triplet_object: subject/mineral dict
    :return: URIRef
    """
    object_name = mineral_name(triplet_object)
    object_node = URIRef(object_name, XDD)
    return object_node


def mineral_type(g: Graph, triplet: RelationshipTriplet, object_node: URIRef) -> Graph:
    """
    add object node to the graph with type as mineral type
    """
    g.add((object_node, RDF.type, GSOG["Rock_Material"]))
    g.add((object_node, RDF.type, GSRM[mineral_name(triplet.object)]))
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
