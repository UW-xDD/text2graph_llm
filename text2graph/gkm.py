import logging
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass

from pydantic import ValidationError
from rdflib import Graph, Literal, RDF, RDFS, XSD, Namespace, URIRef, BNode

from text2graph.macrostrat import get_all_intervals
from text2graph.schema import RelationshipTriplet, Stratigraphy


class Rank(IntEnum):
    BED = 0
    MEMBER = 1
    FORMATION = 2
    GROUP = 3
    SUPERGROUP = 4


STRAT_RANK_EXPANSION = {
    "Bed": "Bed",
    "Mbr": "Member",
    "Fm": "Formation",
    "Gp": "Group",
    "SGp": "Supergroup",
}

STRAT_RANK_LOOKUP = {
    "Bed": Rank.BED,
    "Mbr": Rank.MEMBER,
    "Fm": Rank.FORMATION,
    "Gp": Rank.GROUP,
    "SGp": Rank.SUPERGROUP,
}

GSOC = Namespace("https://w3id.org/gso/1.0/common/")
GSOG = Namespace("https://w3id.org/gso/geology/")
GSGU = Namespace("https://w3id.org/gso/geologicunit/")
GSPR = Namespace("https://w3id.org/gso/geologicprocess/")
GST = Namespace("https://w3id.org/gso/geologictime/")
PROV = Namespace("http://www.w3.org/ns/prov#")
MSL = Namespace("https://macrostrat.org/lexicon/")


RANK_LOOKUP = {
    "Bed": GSGU.Bed,
    "Fm": GSGU.Formation,
    "Mbr": GSGU.Member,
    "Gp": GSGU.Group,
    "SGp": GSGU.Supergroup,
}

BASE_URL = "https://macrostrat.org/api"


def create_interval_lookup(intervals: list[dict]) -> dict:
    """
    create a rdflib namespace GST class for every interval in macrostrat
    :param intervals: list of macrostrat interval dictionaries
    :return: dictionary key: interval name: value: GST.interval class
    """
    lookup = {}
    for interval in intervals:
        interval_name = (
            interval["name"].strip().title().replace(" ", "").replace('"', "")
        )
        interval_type = interval["int_type"].title().replace(" ", "")
        interval_class_name = interval_name + interval_type
        gst_class = GST[interval_class_name]
        lookup[interval_name] = gst_class

    return lookup


INTERVAL_LOOKUP = create_interval_lookup(intervals=get_all_intervals())


def default_rdf_graph() -> Graph:
    """
    Initialize graph with all namespaces bound
    see reference https://loop3d.org/GKM/geology.html
    :return: graph
    """
    g = Graph()
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("gsoc", GSOC)
    g.bind("gsog", GSOG)
    g.bind("gsgu", GSGU)
    g.bind("gst", GST)
    g.bind("gspr", GSPR)
    g.bind("msl", MSL)
    g.bind("prov", PROV)
    return g


def stratigraphic_type(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    g.add((object_node, RDF.type, RANK_LOOKUP[triplet.object.rank]))
    return g


def stratigraphic_label(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    try:
        label = triplet.object.strat_name_long
        if label and label != "None":
            g.add((object_node, RDFS.label, Literal(label, lang="en")))
        else:
            logging.warning(f"{label} not a valid stratname")
    except (KeyError, ValueError):
        g.add((object_node, RDFS.label, Literal(triplet.object.strat_name, lang="en")))
    return g


@dataclass
class RankRelation:
    """
    Represents a stratigraphic name and it's rank
    """

    name: str
    rank: Rank
    rdftype: Namespace
    entity_name: str = ""

    def __post_init__(self):
        self.entity_name = self.name.replace(" ", "")

    @classmethod
    def from_subject_data(cls, subject_data: dict | Stratigraphy):
        if isinstance(subject_data, Stratigraphy):
            subject_data = subject_data.model_dump()
        return cls(
            name=subject_data["strat_name_long"],
            rank=STRAT_RANK_LOOKUP[subject_data["rank"]],
            rdftype=RANK_LOOKUP[subject_data["rank"]],
        )


def stratigraphic_rank_relations(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    Add stratigraphic rank relationships in subject_data to graph
    """
    subject_data = triplet.object.model_dump()
    subject_rank_relation = RankRelation.from_subject_data(subject_data)
    for rank in STRAT_RANK_LOOKUP.keys():
        rank_relation_name = subject_data[rank.lower()]
        rank_relation_dct = dict(
            strat_name_long=rank_relation_name + STRAT_RANK_EXPANSION[rank], rank=rank
        )
        if rank_relation_name:
            relator_rank_relation = RankRelation.from_subject_data(rank_relation_dct)
            if relator_rank_relation.rank < subject_rank_relation.rank:
                # print(f"{relator_rank_relation.name} is part of {subject_rank_relation.name}")
                rank_relation_node = URIRef(relator_rank_relation.entity_name, MSL)
                g.add((rank_relation_node, RDF.type, relator_rank_relation.rdftype))
                g.add((rank_relation_node, GSOC.isPartOf, object_node))

            if relator_rank_relation.rank > subject_rank_relation.rank:
                # print(f"{subject_rank_relation.name} is part of {relator_rank_relation.name} ")
                rank_relation_node = URIRef(relator_rank_relation.entity_name, MSL)
                g.add((rank_relation_node, RDF.type, relator_rank_relation.rdftype))
                g.add((object_node, GSOC.isPartOf, rank_relation_node))
    return g


def deposition_age(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    Add subject deposition age to subject in graph
    """
    subject_data = triplet.object.model_dump()
    period_keys = ["t_period", "b_period"]
    unique_periods = set([subject_data[k] for k in period_keys])
    for period in unique_periods:
        if period and period != "None":
            bnode_deposition = BNode()
            g.add((bnode_deposition, RDF.type, GSPR.Deposition))
            g.add(
                (
                    bnode_deposition,
                    RDFS.label,
                    Literal(f"Deposition during {period}", lang="en"),
                )
            )
            g.add(
                (bnode_deposition, GSOC.occupiesTimeDirectly, INTERVAL_LOOKUP[period])
            )
            g.add((object_node, GSOC.isParticipantIn, bnode_deposition))

    return g


def time_span(g: Graph, triplet: RelationshipTriplet, object_node: URIRef) -> Graph:
    """
    add age range to subject in graph
    """
    if (
        triplet.object.t_age
        and triplet.object.b_age
        and triplet.object.t_age != "None"
        and triplet.object.b_age != "None"
    ):
        bnode_interval = BNode()
        g.add((bnode_interval, RDF.type, GSOG.Geologic_Time_Interval))
        bnode_interval_location = BNode()
        g.add((bnode_interval_location, RDF.type, GSOC.Time_Interval_Location))
        bnode_range = BNode()
        g.add((bnode_range, RDF.type, GSOC.Temporal_Range))
        bnode_range_end = BNode()
        g.add((bnode_range_end, RDF.type, GSOC.Time_Numeric_Value))
        g.add(
            (bnode_range_end, GSOC.hasDataValue, Literal(float(triplet.object.t_age)))
        )
        bnode_range_start = BNode()
        g.add((bnode_range_start, RDF.type, GSOC.Time_Numeric_Value))
        g.add(
            (
                bnode_range_start,
                GSOC.hasDataValue,
                Literal(float(triplet.object.b_age)),
            )
        )

        g.add((object_node, GSOC.occupiesTimeDirectly, bnode_interval))
        g.add((bnode_interval, GSOC.hasQuality, bnode_interval_location))
        g.add((bnode_interval_location, GSOC.hasValue, bnode_range))
        g.add((bnode_range, GSOC.hasEndValue, bnode_range_end))
        g.add((bnode_range, GSOC.hasStartValue, bnode_range_start))

        # provenance
        macrostrat_api = URIRef(triplet.object.provenance.source_name, MSL)
        g.add((macrostrat_api, RDF.type, PROV.entity))

        fetch_activity = URIRef(
            triplet.object.provenance.source_name
            + "_fetch"
            + str(triplet.object.provenance.id),
            MSL,
        )
        g.add(
            (
                fetch_activity,
                PROV.requestedOn,
                Literal(
                    triplet.object.provenance.requested.isoformat(),
                    datatype=XSD.dateTime,
                ),
            )
        )
        g.add((bnode_range, PROV.wasGeneratedBy, macrostrat_api))
        g.add((bnode_range, PROV.wasGeneratedBy, macrostrat_api))
    return g


def spatial_location(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    Add spatial location to subject_node in graph
    """
    WGS84 = URIRef("https://epsg.io/4326")
    bnode_sl = BNode()
    (g.add((bnode_sl, RDF.type, GSOC.SpatialLocation)),)
    g.add((object_node, GSOC.hasQuality, bnode_sl))
    bnode_slv = BNode()
    g.add((bnode_slv, RDF.type, GSOC.SpatialValue))
    g.add((bnode_slv, GSOC.hasDataValue, Literal(triplet.subject.name, lang="en")))
    g.add((bnode_sl, GSOC.hasValue, bnode_slv))
    if (
        triplet.subject.lon
        and triplet.subject.lat
        and triplet.subject.lon != "None"
        and triplet.subject.lat != "None"
    ):
        bnode_slwkt = BNode()
        g.add((bnode_slwkt, RDF.type, GSOC.WKT_Value))
        g.add(
            (
                bnode_slwkt,
                GSOC.hasDataValue,
                Literal(f"( POINT {triplet.subject.lon} {triplet.subject.lat} )"),
            )
        )
        g.add((bnode_slwkt, GSOC.hasReferenceSystem, WGS84))
        g.add((bnode_sl, GSOC.hasValue, bnode_slwkt))
        g.add((WGS84, RDF.type, GSOC.Geographic_Coordinate_System))
    return g


def define_object_node(object_stratigraphy: Stratigraphy) -> URIRef:
    """
    create strat_name node/URIRef for subject/strat_name dict
    :param object_stratigraphy: subject/strat_name dict
    :return: URIRef
    """
    try:
        subject_name = object_stratigraphy.strat_name_long.replace(" ", "")
    except (KeyError, AttributeError):
        subject_name = object_stratigraphy.strat_name.replace(" ", "")
    subject = URIRef(subject_name, MSL)
    return subject


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
    object_node = define_object_node(object_stratigraphy=triplet.object)
    graph_features = [
        stratigraphic_type,
        stratigraphic_label,
        spatial_location,
        stratigraphic_rank_relations,
        deposition_age,
        time_span,
    ]
    for feat in graph_features:
        # try:
        g = feat(g=g, triplet=triplet, object_node=object_node)
        # except Exception as e:
        #     logging.warning(f"failed to add {feat.__name__} to graph with error:{e}"
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
