from rdflib import URIRef, BNode, Graph, Literal, RDF, RDFS, XSD

from text2graph.schema import RelationshipTriplet
from text2graph.gkm.namespace import GSOC, PROV, PAV, MSL, XDD


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


def add_geolocation_provenance_query_and_entity(
    g: Graph, triplet: RelationshipTriplet, attributed_node: URIRef | BNode
) -> Graph:
    """add geolocation provenance to attributed node in Graph"""
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
        g = add_geolocation_provenance_query_and_entity(
            g=g, triplet=triplet, attributed_node=bnode_spatial_location_wkt_value
        )
    return g
