from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
from rdflib import Graph, Literal, RDF, RDFS, Namespace, URIRef, BNode

from text2graph.schema import RelationshipTriples
from text2graph.macrostrat import get_all_intervals


class Rank(IntEnum):
    BED = 0
    MEMBER = 1
    FORMATION = 2
    GROUP = 3
    SUPERGROUP = 4


STRAT_RANK_LOOKUP = {
    'Bed': Rank.BED,
    'Mbr': Rank.MEMBER,
    'Fm': Rank.FORMATION,
    'Gp': Rank.GROUP,
    'SGp': Rank.SUPERGROUP
}

DCT = Namespace('http://purl.org/dc/terms/')
GSOC = Namespace("https://w3id.org/gso/1.0/common/")
GSOG = Namespace("https://w3id.org/gso/geology/")
GSGU = Namespace("https://w3id.org/gso/geologicunit/")
GSPR = Namespace("https://w3id.org/gso/geologicprocess/")
GST = Namespace("https://w3id.org/gso/geologictime/")
MSL = Namespace("http://w3id.org/gso/1.0/ex-macrostratlexicon#")
OWL = Namespace("http://www.w3.org/2002/07/owl#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")

ONTOLOGY_RANK_LOOKUP = {
    'Bed': GSGU.Bed,
    'Fm': GSGU.Formation,
    'Mbr': GSGU.Member,
    'Gp': GSGU.Group,
    'SGp': GSGU.Supergroup,
}


def create_interval_ontology_lookup(intervals: list[dict]) -> dict:
    """
    Build list of GKM ontology interval clases for all intervals in macrostrat
    """
    lookup = {}
    for interval in intervals:
        interval_name = interval["name"].strip().title().replace(" ", "").replace('"', "")
        interval_type = interval['int_type'].title().replace(" ", "")
        interval_class_name = interval_name + interval_type
        gst_class = GST[interval_class_name]
        lookup[interval_name] = gst_class

    return lookup


INTERVAL_ONTOLOGY_LOOKUP = create_interval_ontology_lookup(intervals=get_all_intervals())


@dataclass
class RankRelation:
    name: str
    rank: Rank
    rdftype: Namespace
    entity_name: str = ""

    def __post_init__(self):
        self.entity_name = self.name.replace(" ", "")

    @classmethod
    def from_subject_data(cls, subject_data: dict):
        return cls(
            name=subject_data["strat_name_long"],
            rank=STRAT_RANK_LOOKUP[subject_data["rank"]],
            rdftype=ONTOLOGY_RANK_LOOKUP[subject_data["rank"]]
        )


def triplet_to_rdf(triplet: dict | RelationshipTriples) -> Graph:
    """
    Convert RelationshipTriples object to an RDF graph
    :param triplet: RelationshipTriples object to convert
    :return: RDF graph of RealtionshipTriples object
    """
    if isinstance(triplet, RelationshipTriples):
        triplet = triplet.dict()

    subject_data = triplet["subject"]
    predicate_data = triplet["predicate"]
    object_data = triplet["object"]
    subject_name = subject_data["strat_name_long"].replace(" ", "")
    subject = URIRef(subject_name, MSL)

    g = default_rdf_graph()
    g.add((subject, RDF.type, ONTOLOGY_RANK_LOOKUP[subject_data["rank"]]))
    g.add((subject, RDFS.label, Literal(subject_data["strat_name_long"], lang="en")))
    # g = spatial_location(g=g, object_data=object_data, subject=subject)
    # g = stratigraphic_rank_relations(g=g, subject_data=subject_data, subject=subject)
    # g = deposition_age(g=g, subject_data=subject_data, subject=subject)
    # g = time_span(g=g, subject_data=subject_data, subject=subject)
    # g = macrostrat_ontology_defintion(g=g)
    return g


def macrostrat_ontology_defintion(g: Graph) -> Graph:
    """
    add ontolgy definition to graph
    """
    ontology_node = URIRef("https://w3id.org/gso/1.0/ex-macrostratlexicon/ontology")
    cc_license = URIRef("https://creativecommons.org/licenses/by/4.0/legalcode")
    master_ontology = URIRef("https://w3id.org/gso/1.0/master/ontology")
    g.add((ontology_node, RDF.type, OWL.Ontology))
    g.add((ontology_node, DCT.created, Literal("2024-03-20", datatype=XSD.date)))
    g.add((ontology_node, DCT.modified, Literal("2024-03-20", datatype=XSD.date)))
    g.add((ontology_node, DCT.license, cc_license))
    g.add((ontology_node, RDFS.label, Literal("Macrostrat stratigraphic lexicon", lang="en")))
    g.add((ontology_node, OWL.imports, master_ontology))
    return g


def stratigraphic_rank_relations(g: Graph, subject_data: dict, subject: URIRef) -> Graph:
    """
    Add stratigraphic rank relationships in subject_data to graph
    """
    subject_rank_relation = RankRelation.from_subject_data(subject_data)
    for k in STRAT_RANK_LOOKUP.keys():
        rank_relation_name = subject_data[k.lower()]
        rank_relation_dct = dict(strat_name_long=rank_relation_name + k, rank=k)
        if rank_relation_name:
            relator_rank_relation = RankRelation.from_subject_data(rank_relation_dct)
            if relator_rank_relation.rank < subject_rank_relation.rank:
                # print(f"{relator_rank_relation.name} is part of {subject_rank_relation.name}")
                rank_relation_node = URIRef(relator_rank_relation.entity_name, MSL)
                g.add((rank_relation_node, RDF.type, relator_rank_relation.rdftype))
                g.add((rank_relation_node, GSOC.isPartOf, subject))

            if relator_rank_relation.rank > subject_rank_relation.rank:
                # print(f"{subject_rank_relation.name} is part of {relator_rank_relation.name} ")
                rank_relation_node = URIRef(relator_rank_relation.entity_name, MSL)
                g.add((rank_relation_node, RDF.type, relator_rank_relation.rdftype))
                g.add((subject, GSOC.isPartOf, rank_relation_node))
    return g


def deposition_age(g: Graph, subject_data: dict, subject: URIRef) -> Graph:
    """
    Add subject deposition age to subject in graph
    """
    period_keys = ["t_period", "b_period"]
    unique_periods = set([subject_data[k] for k in period_keys])
    for period in unique_periods:
        if period:
            bnode_deposition = BNode()
            g.add((bnode_deposition, RDF.type, GSPR.Deposition))
            g.add((bnode_deposition, RDFS.label, Literal(f"Deposition during {period}", lang="en")))
            g.add((bnode_deposition, GSOC.occupiesTimeDirectly, INTERVAL_ONTOLOGY_LOOKUP[period]))
            g.add((subject, GSOC.isParticipantIn, bnode_deposition))

    return g


def time_span(g: Graph, subject_data: dict, subject: URIRef) -> Graph:
    """
    add age range to subject in graph
    """
    bnode_interval = BNode()
    g.add((bnode_interval, RDF.type, GSOG.Geologic_Time_Interval))
    bnode_interval_location = BNode()
    g.add((bnode_interval_location, RDF.type, GSOC.Time_Interval_Location))
    bnode_range = BNode()
    g.add((bnode_range, RDF.type, GSOC.Temporal_Range))
    bnode_range_end = BNode()
    g.add((bnode_range_end, RDF.type, GSOC.Time_Numeric_Value))
    g.add((bnode_range_end, GSOC.hasDataValue, Literal(float(subject_data['t_age']))))
    bnode_range_start = BNode()
    g.add((bnode_range_start, RDF.type, GSOC.Time_Numeric_Value))
    g.add((bnode_range_start, GSOC.hasDataValue, Literal(float(subject_data['b_age']))))

    g.add((subject, GSOC.occupiesTimeDirectly, bnode_interval))
    g.add((bnode_interval, GSOC.hasQuality, bnode_interval_location))
    g.add((bnode_interval_location, GSOC.hasValue, bnode_range))
    g.add((bnode_range, GSOC.hasEndValue, bnode_range_end))
    g.add((bnode_range, GSOC.hasStartValue, bnode_range_start))
    return g


def spatial_location(g: Graph, object_data: dict, subject: URIRef) -> Graph:
    """
    Add spatial location to subject in graph
    """
    wgs84 = URIRef("https://epsg.io/4326")
    bnode_sl = BNode()
    g.add((bnode_sl, RDF.type, GSOC.SpatialLocation)),
    g.add((subject, GSOC.hasQuality, bnode_sl))
    bnode_slv = BNode()
    g.add((bnode_slv, RDF.type, GSOC.SpatialValue))
    g.add((bnode_slv, GSOC.hasDataValue, Literal(object_data['name'], lang="en")))
    g.add((bnode_sl, GSOC.hasValue, bnode_slv))
    bnode_slwkt = BNode()
    g.add((bnode_slwkt, RDF.type, GSOC.WKT_Value))
    g.add((bnode_slwkt, GSOC.hasDataValue, Literal(f"( POINT {object_data['lon']} {object_data['lat']} )")))
    g.add((bnode_slwkt, GSOC.hasReferenceSystem, wgs84))
    g.add((bnode_sl, GSOC.hasValue, bnode_slwkt))
    g.add((wgs84, RDF.type, GSOC.Geographic_Coordinate_System))
    return g


def default_rdf_graph() -> Graph:
    """
    Initialize graph with all namespaces bound
    :return: graph
    """
    # https://loop3d.org/GKM/geology.html
    g = Graph()

    g.bind("msl", MSL)
    g.bind("dct", DCT)
    g.bind("gsoc", GSOC)
    g.bind("gsog", GSOG)
    g.bind("gsgu", GSGU)
    g.bind("gst", GST)
    g.bind("gspr", GSPR)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

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
        with open(filename, 'w') as f:
            f.write(output)
    return output
