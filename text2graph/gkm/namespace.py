from rdflib import Namespace, Graph, RDF, RDFS

GSOC = Namespace("https://w3id.org/gso/1.0/common/")
GSOG = Namespace("https://w3id.org/gso/geology/")
GSGU = Namespace("https://w3id.org/gso/geologicunit/")
GSPR = Namespace("https://w3id.org/gso/geologicprocess/")
GSRM = Namespace("https://w3id.org/gso/1.0/rockmaterial/")
GST = Namespace("https://w3id.org/gso/geologictime/")
PROV = Namespace("http://www.w3.org/ns/prov#")
PAV = Namespace("http://purl.org/pav/")
MSL = Namespace("https://macrostrat.org/lexicon/")
XDD = Namespace("https://xdd.wisc.edu/lexicon/")


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
