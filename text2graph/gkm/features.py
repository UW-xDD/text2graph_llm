from enum import IntEnum
from dataclasses import dataclass
from rdflib import Graph, Literal, RDF, RDFS, XSD, Namespace, URIRef, BNode

from text2graph.macrostrat import get_all_intervals
from text2graph.schema import Stratigraphy, RelationshipTriplet


def define_object_node(triplet_object: Stratigraphy) -> URIRef:
    """
    create strat_name node/URIRef for object/strat_name dict
    :param triplet_object: subject/strat_name dict
    :return: URIRef
    """
    try:
        object_name = triplet_object.strat_name_long.replace(" ", "")
    except (KeyError, AttributeError):
        object_name = triplet_object.strat_name.replace(" ", "")
    object_node = URIRef(object_name, MSL)
    return object_node


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

STRAT_RANK_CONTRACTION = {v: k for k, v in STRAT_RANK_EXPANSION.items()}

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
PAV = Namespace("http://purl.org/pav/")
MSL = Namespace("https://macrostrat.org/lexicon/")
XDD = Namespace("https://xdd.wisc.edu/lexicon/")

RANK_LOOKUP = {
    "Bed": GSGU.Bed,
    "Fm": GSGU.Formation,
    "Mbr": GSGU.Member,
    "Gp": GSGU.Group,
    "SGp": GSGU.Supergroup,
}


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
    g.bind("xdd", XDD)
    g.bind("prov", PROV)
    g.bind("pav", PAV)
    return g


def add_macrostrat_query_and_entity(
    g: Graph, triplet: RelationshipTriplet, attributed_node: URIRef | BNode
) -> Graph:
    """add macrostrat provenance to attributed node in Graph"""
    macrostrat_entity_node = URIRef(triplet.object.provenance.source_name, MSL)
    g.add((macrostrat_entity_node, RDF.type, PROV.entity))
    g.add(
        (
            macrostrat_entity_node,
            PAV.version,
            Literal(triplet.object.provenance.source_version, datatype=XSD.string),
        )
    )

    macrostrat_query_node = URIRef(triplet.object.provenance.source_name + "Query", MSL)
    g.add((macrostrat_query_node, RDF.type, PROV.activity))
    g.add((macrostrat_query_node, PROV.used, macrostrat_entity_node))
    g.add(
        (
            macrostrat_query_node,
            PROV.atLocation,
            Literal(triplet.object.provenance.source_url, datatype=XSD.anyURI),
        )
    )
    g.add(
        (
            macrostrat_query_node,
            PROV.requestedAt,
            Literal(
                triplet.object.provenance.requested.isoformat(), datatype=XSD.dateTime
            ),
        )
    )

    g.add((attributed_node, PROV.wasGeneratedBy, macrostrat_query_node))
    return g


def add_serpapi_query_and_entity(
    g: Graph, triplet: RelationshipTriplet, attributed_node: URIRef | BNode
) -> Graph:
    """add serpapi provenance to attributed node in Graph"""
    serpapi_entity_node = URIRef(triplet.subject.provenance.source_name, XDD)
    g.add((serpapi_entity_node, RDF.type, PROV.entity))
    g.add(
        (
            serpapi_entity_node,
            PAV.version,
            Literal(triplet.subject.provenance.source_version, datatype=XSD.string),
        )
    )

    serapi_query_node = URIRef(triplet.subject.provenance.source_name + "Query", XDD)
    g.add((serapi_query_node, RDF.type, PROV.activity))
    g.add((serapi_query_node, PROV.used, serpapi_entity_node))
    g.add(
        (
            serapi_query_node,
            PROV.atLocation,
            Literal(triplet.subject.provenance.source_url, datatype=XSD.anyURI),
        )
    )
    g.add(
        (
            serapi_query_node,
            PROV.requestedAt,
            Literal(
                triplet.subject.provenance.requested.isoformat(), datatype=XSD.dateTime
            ),
        )
    )

    g.add((attributed_node, PROV.wasGeneratedBy, serapi_query_node))
    return g


def stratigraphic_type(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    add object node to the graph with type as macrostrat stratigraphic rank
    """
    # Stratigprahy has Macrostrat defined rank
    try:
        g.add((object_node, RDF.type, RANK_LOOKUP[triplet.object.rank]))
        return g
    except KeyError:
        pass

    # Stratigraphy name has rank as last word in name
    try:
        name_from_title = triplet.object.strat_name.split()[-1].title()
        if len(name_from_title) > 3:
            name_from_title = STRAT_RANK_CONTRACTION[name_from_title]
        g.add((object_node, RDF.type, RANK_LOOKUP[name_from_title]))
        return g
    except KeyError:
        pass

    # No specific stratigraphic rank
    g.add((object_node, RDF.type, GSGU.StratigraphicUnit))
    return g


def stratigraphic_label(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    add strat_name_long as label to subject node if strat_name_long is present, else use strat_name
    """
    try:
        label = triplet.object.strat_name_long
        if label and label != "None":
            g.add((object_node, RDFS.label, Literal(label, lang="en")))
            return g
    except (KeyError, AttributeError):
        pass
    g.add((object_node, RDFS.label, Literal(triplet.object.strat_name, lang="en")))
    return g


def triplet_provenance(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    add triplet and object provenance to the object node
    """
    hybrid_api_provenance = triplet.provenance.find("Ask_xDD_hybrid_API")

    # xdd textpreprocessor HayStack
    xdd_text_processor_node = URIRef("XDDTextPreProcessor", XDD)
    g.add((xdd_text_processor_node, RDF.type, PROV.entity))
    g.add(
        (
            xdd_text_processor_node,
            RDFS.label,
            Literal(
                hybrid_api_provenance.additional_values["preprocessor_id"], lang="en"
            ),
        )
    )
    g.add(
        (
            xdd_text_processor_node,
            PAV.version,
            Literal(
                hybrid_api_provenance.additional_values["preprocessor_id"],
                datatype=XSD.string,
            ),
        )
    )

    # XDDCorpus/DocIDS
    doc_ids_corpus_node = URIRef("XDDCorpus", XDD)
    g.add((doc_ids_corpus_node, RDF.type, PROV.entity))
    g.add((doc_ids_corpus_node, RDFS.label, Literal("xDD document ids", lang="en")))
    g.add(
        (
            doc_ids_corpus_node,
            XDD.docID,
            Literal(
                hybrid_api_provenance.additional_values["paper_id"], datatype=XSD.string
            ),
        )
    )
    g.add(
        (
            doc_ids_corpus_node,
            XDD.docURL,
            Literal(
                hybrid_api_provenance.additional_values["url"], datatype=XSD.anyURI
            ),
        )
    )
    g.add((doc_ids_corpus_node, PROV.used, xdd_text_processor_node))

    # RAG App/Hybrid endpoint
    rag_hybrid_retriever_node = URIRef(
        hybrid_api_provenance.source_name.replace(" ", ""), XDD
    )
    g.add((rag_hybrid_retriever_node, RDF.type, PROV.entity))
    g.add(
        (
            rag_hybrid_retriever_node,
            RDFS.label,
            Literal(hybrid_api_provenance.source_name, lang="en"),
        )
    )
    g.add(
        (
            rag_hybrid_retriever_node,
            PAV.version,
            Literal(hybrid_api_provenance.source_version, datatype=XSD.string),
        )
    )
    g.add((rag_hybrid_retriever_node, PROV.used, doc_ids_corpus_node))

    # LLM Model
    llm_entity_node = URIRef(triplet.provenance.source_name, XDD)
    g.add((llm_entity_node, RDF.type, PROV.entity))
    g.add(
        (
            llm_entity_node,
            PAV.version,
            Literal(triplet.provenance.source_version, datatype=XSD.string),
        )
    )

    # LLM Query
    llm_query_node = URIRef(triplet.provenance.source_name + "_query", XDD)
    g.add((llm_query_node, RDF.type, PROV.activity))
    g.add(
        (
            llm_query_node,
            PROV.startedAtTime,
            Literal(triplet.provenance.requested.isoformat(), datatype=XSD.dateTime),
        )
    )
    g.add((llm_query_node, PROV.used, llm_entity_node))
    g.add((llm_query_node, PROV.used, rag_hybrid_retriever_node))

    g.add((object_node, PROV.wasGeneratedBy, llm_query_node))
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
    Add stratigraphic rank relationships in subject_data to graph by using macrostrat Stratigraphy Member, Formation, Group and Supergroup values
    """
    subject_data = triplet.object.model_dump()
    try:
        subject_rank_relation = RankRelation.from_subject_data(subject_data)
    except KeyError:
        return g
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
                g = add_macrostrat_query_and_entity(
                    g=g, triplet=triplet, attributed_node=rank_relation_node
                )

            if relator_rank_relation.rank > subject_rank_relation.rank:
                # print(f"{subject_rank_relation.name} is part of {relator_rank_relation.name} ")
                rank_relation_node = URIRef(relator_rank_relation.entity_name, MSL)
                g.add((rank_relation_node, RDF.type, relator_rank_relation.rdftype))
                g.add((object_node, GSOC.isPartOf, rank_relation_node))
                g = add_macrostrat_query_and_entity(
                    g=g, triplet=triplet, attributed_node=rank_relation_node
                )

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
            g = add_macrostrat_query_and_entity(
                g=g, triplet=triplet, attributed_node=bnode_deposition
            )

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
        g = add_macrostrat_query_and_entity(
            g=g, triplet=triplet, attributed_node=bnode_interval
        )

    return g


def spatial_location(
    g: Graph, triplet: RelationshipTriplet, object_node: URIRef
) -> Graph:
    """
    Add spatial location to subject_node in graph
    """
    WGS84 = URIRef("https://epsg.io/4326")
    bnode_spatial_location = BNode()
    (g.add((bnode_spatial_location, RDF.type, GSOC.SpatialLocation)),)
    g.add((object_node, GSOC.hasQuality, bnode_spatial_location))
    bnode_spatial_location_value = BNode()
    g.add((bnode_spatial_location_value, RDF.type, GSOC.SpatialValue))
    g.add(
        (
            bnode_spatial_location_value,
            GSOC.hasDataValue,
            Literal(triplet.subject.name, lang="en"),
        )
    )
    g.add((bnode_spatial_location, GSOC.hasValue, bnode_spatial_location_value))
    if (
        triplet.subject.lon
        and triplet.subject.lat
        and triplet.subject.lon != "None"
        and triplet.subject.lat != "None"
    ):
        bnode_spatial_location_wkt_value = BNode()
        g.add((bnode_spatial_location_wkt_value, RDF.type, GSOC.WKT_Value))
        g.add(
            (
                bnode_spatial_location_wkt_value,
                GSOC.hasDataValue,
                Literal(f"( POINT {triplet.subject.lon} {triplet.subject.lat} )"),
            )
        )
        g.add((bnode_spatial_location_wkt_value, GSOC.hasReferenceSystem, WGS84))
        g.add((bnode_spatial_location, GSOC.hasValue, bnode_spatial_location_wkt_value))
        g.add((WGS84, RDF.type, GSOC.Geographic_Coordinate_System))
        g = add_serpapi_query_and_entity(
            g=g, triplet=triplet, attributed_node=bnode_spatial_location_wkt_value
        )
    return g
