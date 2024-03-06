

def ttl_general() -> str:
    gen_ttl = """@prefix xdd: <https://w3id.org/gso/1.0/ex-xddlexicon#> .
@prefix dct: <http://purl.org/dc/terms/> .
@prefix gsel: <https://w3id.org/gso/1.0/element/> .
@prefix gsen: <https://w3id.org/gso/1.0/geologicsetting/> .
@prefix gsgu: <https://w3id.org/gso/1.0/geologicunit/> .
@prefix gsmin: <https://w3id.org/gso/1.0/mineral/> .
@prefix gsoc: <https://w3id.org/gso/1.0/common/> .
@prefix gsog: <https://w3id.org/gso/1.0/geology/> .
@prefix gsoq: <https://w3id.org/gso/1.0/quality/> .
@prefix gspd: <https://w3id.org/gso/1.0/perdurant/> .
@prefix gspr: <https://w3id.org/gso/1.0/geologicprocess/> .
@prefix gsps: <https://w3id.org/gso/1.0/lithology/particleroundness/> .
@prefix gsrm: <https://w3id.org/gso/1.0/rockmaterial/> .
@prefix gstime: <https://w3id.org/gso/1.0/ischart/> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix ppbc: <https://w3id.org/gso/1.0/ex-petrophysics-bc#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix schema: <https://schema.org/> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix turtle: <http://www.semanticweb.org/owl/owlapi/turtle#> .
@prefix unit: <http://qudt.org/vocab/unit/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

"""
    return gen_ttl


RDFTYPE_LOOKUP = {
    "Mbr": "gsgu:Member",
    "Fm": "gsgu:Formation",
    "Gp": "gsgu:Group",
    "SGp": "gsgu:Supergroup",
}

class gkm_xdd_entity():
    def __init__(self, name):
        """
        build GKM strings
        """
        # self.gkm: str = ""
        self.properties: dict[str: list[str]] = {}
        self.additional_entities: list[gkm_xdd_entity] = []
        self.newline_indent: str = " ;\n\t"
        self.entity_separator: str = "\n.\n"
        self.name: str = name.replace(" ", "")
        self.header = "xdd:" + self.name + self.newline_indent

    def __add__(self, type, item) -> None:
        try:
            self.properties[type].append(item)
        except KeyError:
            self.properties[type] = [item]

    def add_property(self, type, item) -> None:
        self.__add__(type, item)

    def __add_additional_entity__(self, item) -> None:
        self.additional_entities.append(item)

    def add_lithology(self, lith_name) -> None:
        lith_entity_name = self.name + "-" + lith_name
        self.__add__("gsoc:hasConstituent", f"xdd:{lith_entity_name}")

        lith_entity = gkm_xdd_entity(lith_entity_name)
        lith_entity.__add__("rdf:type", f"gsrm:{lith_entity_name}")
        self.__add_additional_entity__(lith_entity)

    def __total_current_locations__(self):
        return sum([1 for x in self.additional_entities if 'location' in x.name])

    def add_location_coords(self,
                            lat: float,
                            lon: float,
                            reference_system_name: str | None = None,
                            reference_system_val: str | None = None
                            ) -> None:
        location_name = self.name + f"-location-{self.__total_current_locations__()}"
        self.__add__("gsoc:hasQuality", f"xdd:{location_name}")
        location_entity = gkm_xdd_entity(location_name)
        location_entity.__add__("rdf:type", "gsoc:SpatialLocation")
        location_entity.__add__(f"gsoc:hasSpatialValue", f"[gsoc:WKTValue: POINT ({lat} {lon})]")

        reference_system_entity = None
        if reference_system_name:
            specific_system_name = self.name + "-" + reference_system_name.replace(" ", "")
            location_entity.__add__(f"gsoc:hasReferenceSystem", f"{specific_system_name}")
            reference_system_entity = gkm_xdd_entity(name=specific_system_name)
            reference_system_entity.__add__("rdf:type", "gsoc:GeographicCoordinateSystem")
            if not reference_system_val:
                reference_system_val = reference_system_name
            reference_system_entity.__add__("gsoc:hasValue", f"{reference_system_val}")

        self.__add_additional_entity__(location_entity)
        if reference_system_entity:
            self.__add_additional_entity__(reference_system_entity)

    def add_location_name(self, name: str) -> None:
        location_name = self.name + f"-location-{self.__total_current_locations__()}"
        self.__add__("gsoc:hasQuality", f"xdd:{location_name}")
        location_entity = gkm_xdd_entity(location_name)
        location_entity.__add__("rdf:type", "gsoc:SpatialLocation")
        location_entity.__add__("gsoc:hasValue", f"{name}")
        self.__add_additional_entity__(location_entity)

    def __gkm_str__(self) -> str:
        self_gkm_string = self.header
        for k, v in self.properties.items():
            for item in v:
                self_gkm_string += f"{k}: {item}" + self.newline_indent
        return self_gkm_string[:-2]

    def full_gkm_str(self) -> str:
        all_additional_entity_strings = [x.__gkm_str__() + x.entity_separator for x in self.additional_entities]
        return (
                self.__gkm_str__()
                + self.entity_separator
                + "".join(all_additional_entity_strings)
        )


def gkm_xdd_entity_from_macrostrat_unit_dict(macrostrat_unit_dict) -> gkm_xdd_entity:
    gxe = gkm_xdd_entity(name=macrostrat_unit_dict["unit_name"])

    for k, v in RDFTYPE_LOOKUP.items():
        if macrostrat_unit_dict[k]:
            gxe.__add__("rdf:type", v)

    if macrostrat_unit_dict["lith"]:
        lith_name = macrostrat_unit_dict['lith'][0]['name']
        gxe.add_lithology(lith_name)

    if macrostrat_unit_dict["clat"] and macrostrat_unit_dict["clng"]:
        gxe.add_location_coords(lat=macrostrat_unit_dict["clat"], lon=macrostrat_unit_dict["clng"],
                                reference_system_name="WGS 84")

    return gxe


def final_gkm_file(gkm_xdd_entities: list[gkm_xdd_entity]) -> str:
    final = ttl_general()
    for gkm_xdd_entity in gkm_xdd_entities:
        final += gkm_xdd_entity.full_gkm_str()

    return final
