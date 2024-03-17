from __future__ import annotations


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


def ttl_prefixes() -> str:
    prefixes = """@prefix xdd: <https://w3id.org/gso/1.0/ex-xddlexicon#> .
@prefix gsgu: <https://w3id.org/gso/1.0/geologicunit/> .
@prefix gsoc: <https://w3id.org/gso/1.0/common/> .
@prefix gsrm: <https://w3id.org/gso/1.0/rockmaterial/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

"""
    return prefixes

RANK_LOOKUP = {
    "Mbr": "Member",
    "Fm": "Formation",
    "Gp": "Group",
    "SGp": "Supergroup",
}

RDFTYPE_LOOKUP = {
    "Mbr": "gsgu:Member",
    "Fm": "gsgu:Formation",
    "Gp": "gsgu:Group",
    "SGp": "gsgu:Supergroup",
}


class GKMxDDEntity:
    def __init__(self, name: str, entity_type: str | None = None) -> None:
        """
        build GKM strings
        :param name: the name of the xdd gkm entity to start building

        public methods
        add_property: add an arbitrary property to this entity
        add_lithology: add a lithology to this entity
        add_location_coords: add a location to this entity using GPS coordinates
        add_location_name: add a locaiton to this property using a named location
        full_gkm_str: return the full GKM string of this entity
        """
        self.__newline_indent__: str = " ;\n\t"
        self.__entity_separator__: str = "\n.\n"
        self.__properties__: dict[str: list[str]] = {}
        self.__additional_entities__: list[GKMxDDEntity] = []

        self.name: str = name.replace(" ", "")
        self.prefix = "xdd"
        self.name_with_prefix = self.prefix + ":" + self.name
        self.__header__ = self.name_with_prefix + self.__newline_indent__
        if entity_type:
            self.add_property(
                type="rdf:type",
                item=f"{entity_type}"
            )
        if not entity_type:
            type = name.split()[-1]
            try:
                processed_type = RANK_LOOKUP[type]
            except KeyError:
                processed_type = type
            self.add_property(
                type="rdf:type",
                item=f"gsgu:{processed_type}"
            )

    def __add__(self, type: str, item: str) -> None:
        """manage process for adding new properties"""
        try:
            self.__properties__[type].append(item)
        except KeyError:
            self.__properties__[type] = [item]

    def add_property(self, type: str, item: str) -> GKMxDDEntity:
        """add any property to this entity, type=<prefix>:<type> item=<value>"""
        self.__add__(type, item)
        return self

    def __add_additional_entity__(self, item: GKMxDDEntity) -> None:
        self.__additional_entities__.append(item)

    def add_lithology(self, lith_name: str) -> GKMxDDEntity:
        """
        Add a lithology to this entity
        :param lith_name:
        """
        self.add_property("gsoc:hasConstituent", f"xdd:{lith_name}")
        return self

    def add_location_coords(
        self,
        lat: float,
        lon: float,
        reference_system_name: str | None = None,
    ) -> GKMxDDEntity:
        """
        Add a point location to this entity with lat lon coordinates
        """
        location_item = (
            "gsoc: SpatialLocation ["
            + "\n\t\tgsoc:hasSpatialValue ["
            + f"\n\t\t\tgsoc:WKTValue: POINT ({lat} {lon})"
            + "\n\t\t]"
            + "\n\t\tgsoc:hasReferenceSystem ["
            + "\n\t\t\tgsoc:GeographicCoordinateSystem gsoc:hasValue ["
            + f"\n\t\t\t\"{reference_system_name}\""
            + "\n\t\t]"
            + "\n\t]"
        )
        self.add_property(
            type="gsoc:hasQuality",
            item=location_item
        )
        return self

    def __total_current_locations__(self) -> int:
        return sum([1 for x in self.__additional_entities__ if 'location' in x.name])

    def add_location_coords_with_addnl_location_entity(
        self,
        lat: float,
        lon: float,
        reference_system_name: str | None = None,
        reference_system_val: str | None = None
    ) -> GKMxDDEntity:
        """
        Add a point location to this entity with lat lon coordinates
        """
        location_name = f"xdd:{self.name}-location-{self.__total_current_locations__()}"
        self.add_property(
            type="gsoc:hasQuality",
            item=f"{location_name}"
        )
        location_entity = GKMxDDEntity(
            name=location_name,
            entity_type="gsoc:SpatialLocation"
        )
        location_entity.add_property(
            type=f"gsoc:hasSpatialValue",
            item=f"[gsoc:WKTValue: POINT ({lat} {lon})]"
        )
        reference_system_entity = None
        if reference_system_name:
            specific_system_name = reference_system_name.replace(" ", "")
            reference_system_entity = GKMxDDEntity(
                name=specific_system_name,
                entity_type="gsoc:GeographicCoordinateSystem"
            )
            location_entity.add_property(
                type=f"gsoc:hasReferenceSystem",
                item=f"{reference_system_entity.name}"
            )
            reference_system_entity.add_property(
                type="gsoc:hasValue",
                item=f"\"{reference_system_val if reference_system_val else reference_system_name}\""
            )

        self.__add_additional_entity__(location_entity)
        if reference_system_entity:
            self.__add_additional_entity__(reference_system_entity)
        return self

    def add_location_name(self, name: str) -> GKMxDDEntity:
        """
        Add a location by place name to this entity
        """
        # location_name = self.name + f"-location-{self.__total_current_locations__()}"
        spatial_location_value = (
                "gsoc:SpatialLocation ["
                + f"\n\t\tgsoc:hasValue \"{name}\" ;"
                + f"\n\t]"
        )
        self.add_property(type="gsoc:hasQuality", item=spatial_location_value)
        return self

    def __gkm_str__(self) -> str:
        self_gkm_string = self.__header__
        for k, v in self.__properties__.items():
            for item in v:
                self_gkm_string += f"{k} {item}" + self.__newline_indent__
        return self_gkm_string[:-2]

    def full_gkm_str(self) -> str:
        all_additional_entity_strings = [x.__gkm_str__() + x.__entity_separator__ for x in self.__additional_entities__]
        return (
                self.__gkm_str__()
                + self.__entity_separator__
                + "".join(all_additional_entity_strings)
        )
