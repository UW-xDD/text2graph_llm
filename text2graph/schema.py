from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

import httpx
import pytz
from pydantic import AnyUrl, BaseModel, BeforeValidator, ConfigDict, Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from text2graph import macrostrat
from text2graph.geolocation.geocode import RateLimitedClient, get_gps
from text2graph.usgs import USGS_FORMULAS


class Provenance(BaseModel):
    """class for collecting data source information"""

    id: UUID = Field(default_factory=uuid4)
    source_name: str
    source_url: str | None = None
    source_version: str | int | float | None = None
    requested: datetime = datetime.now(pytz.utc)
    additional_values: dict[str, str | float | int | list[str] | None] = Field(
        default_factory=dict
    )
    previous: Provenance | None = None

    def find(self, source_name: str) -> Provenance | None:
        """
        return provenance object from provenance chain with matching source name if no matching source name returns None
        :param source_name: exact match for provenance chain source name
        :return: Provenance object or None
        """
        if self.source_name == source_name:
            return self
        elif not self.previous:
            return None
        else:
            return self.previous.find(source_name=source_name)


class Paragraph(BaseModel):
    """Enriched Ask-xDD Retriever results."""

    id: str
    paper_id: str
    preprocessor_id: str
    doc_type: str
    topic_list: list[str]
    text_content: str
    hashed_text: str
    cosmos_object_id: str | None
    distance: float
    url: AnyUrl
    provenance: Provenance


class Lithology(BaseModel):
    name: str
    lith_id: int | None = None
    type: str | None = None
    group: str | None = None
    _class: str | None = None
    color: str | None = None
    fill: int | None = None
    t_units: int | None = None
    provenance: Provenance | None = None

    async def hydrate(self) -> None:
        """Hydrate Lithology from macrostrat."""
        try:
            hit = await macrostrat.get_records(
                entity_type=macrostrat.EntityType.LITHOLOGY, name=self.name, exact=True
            )
            hit = hit[0]
        except (ValueError, IndexError):
            logging.warning(f"No records found for lithology '{self.name}'")
            return

        # Load data into model
        for k, v in hit.items():
            if k == "class":
                setattr(
                    self, "_class", v
                )  # Avoid clashing with reserved keyword 'class'
            else:
                setattr(self, k, v)

        self.provenance = Provenance(
            source_name="Macrostrat",
            source_url=f"{macrostrat.BASE_URL}'/defs/lithologies?lith_id={hit['lith_id']}",
            previous=self.provenance,
        )


class Stratigraphy(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    strat_name: str = Field(alias="name")
    strat_name_long: str | None = None
    rank: str | None = None
    strat_name_id: int | None = None
    concept_id: int | None = None
    bed: str | None = None
    bed_id: int | None = None
    mbr: str | None = None
    mbr_id: int | None = None
    fm: str | None = None
    fm_id: int | None = None
    subgp: str | None = None
    subgp_id: int | None = None
    gp: str | None = None
    gp_id: int | None = None
    sgp: str | None = None
    sgp_id: int | None = None
    b_age: float | None = None
    t_age: float | None = None
    b_period: str | None = None
    t_period: str | None = None
    c_interval: str | None = None
    t_units: int | None = None
    ref_id: int | None = None
    provenance: Provenance | None = None

    @property
    def name(self) -> str:
        return self.strat_name

    async def hydrate(self) -> None:
        """Hydrate Stratigraphy from macrostrat."""
        try:
            hit = await macrostrat.get_records(
                entity_type=macrostrat.EntityType.STRAT_NAME,
                name=self.strat_name,
                exact=False,
            )
            hit = hit[0]
        except (ValueError, IndexError):
            logging.info(f"No records found for stratigraphy '{self.name}'")
            return

        macrostrat_version = hit.pop("macrostrat_version")
        # Load data into model
        for k, v in hit.items():
            setattr(self, k, v)

        self.provenance = Provenance(
            source_name="Macrostrat",
            source_version=macrostrat_version,
            source_url=f"{macrostrat.BASE_URL}/defs/strat_names?strat_name_id={hit['strat_name_id']}",
            previous=self.provenance,
        )


class Element(Enum):
    Hydrogen = "H"
    Helium = "He"
    Lithium = "Li"
    Beryllium = "Be"
    Boron = "B"
    Carbon = "C"
    Nitrogen = "N"
    Oxygen = "O"
    Fluorine = "F"
    Neon = "Ne"
    Sodium = "Na"
    Magnesium = "Mg"
    Aluminium = "Al"
    Silicon = "Si"
    Phosphorus = "P"
    Sulfur = "S"
    Chlorine = "Cl"
    Argon = "Ar"
    Potassium = "K"
    Calcium = "Ca"
    Scandium = "Sc"
    Titanium = "Ti"
    Vanadium = "V"
    Chromium = "Cr"
    Manganese = "Mn"
    Iron = "Fe"
    Cobalt = "Co"
    Nickel = "Ni"
    Copper = "Cu"
    Zinc = "Zn"
    Gallium = "Ga"
    Germanium = "Ge"
    Arsenic = "As"
    Selenium = "Se"
    Bromine = "Br"
    Krypton = "Kr"
    Rubidium = "Rb"
    Strontium = "Sr"
    Yttrium = "Y"
    Zirconium = "Zr"
    Niobium = "Nb"
    Molybdenum = "Mo"
    Technetium = "Tc"
    Ruthenium = "Ru"
    Rhodium = "Rh"
    Palladium = "Pd"
    Silver = "Ag"
    Cadmium = "Cd"
    Indium = "In"
    Tin = "Sn"
    Antimony = "Sb"
    Tellurium = "Te"
    Iodine = "I"
    Xenon = "Xe"
    Caesium = "Cs"
    Barium = "Ba"
    Lanthanum = "La"
    Cerium = "Ce"
    Praseodymium = "Pr"
    Neodymium = "Nd"
    Promethium = "Pm"
    Samarium = "Sm"
    Europium = "Eu"
    Gadolinium = "Gd"
    Terbium = "Tb"
    Dysprosium = "Dy"
    Holmium = "Ho"
    Erbium = "Er"
    Thulium = "Tm"
    Ytterbium = "Yb"
    Lutetium = "Lu"
    Hafnium = "Hf"
    Tantalum = "Ta"
    Tungsten = "W"
    Rhenium = "Re"
    Osmium = "Os"
    Iridium = "Ir"
    Platinum = "Pt"
    Gold = "Au"
    Mercury = "Hg"
    Thallium = "Tl"
    Lead = "Pb"
    Bismuth = "Bi"
    Polonium = "Po"
    Astatine = "At"
    Radon = "Rn"
    Francium = "Fr"
    Radium = "Ra"
    Actinium = "Ac"
    Thorium = "Th"
    Protactinium = "Pa"
    Uranium = "U"
    Neptunium = "Np"
    Plutonium = "Pu"
    Americium = "Am"
    Curium = "Cm"
    Berkelium = "Bk"
    Californium = "Cf"
    Einsteinium = "Es"
    Fermium = "Fm"
    Mendelevium = "Md"
    Nobelium = "No"
    Lawrencium = "Lr"
    Rutherfordium = "Rf"
    Dubnium = "Db"
    Seaborgium = "Sg"
    Bohrium = "Bh"
    Hassium = "Hs"
    Meitnerium = "Mt"
    Darmstadtium = "Ds"
    Roentgenium = "Rg"
    Copernicium = "Cn"
    Nihonium = "Nh"
    Flerovium = "Fl"
    Moscovium = "Mc"
    Livermorium = "Lv"
    Tennessine = "Ts"
    Oganesson = "Og"


class Mineral(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mineral: str = Field(alias="name")
    mineral_id: int | None = None
    mineral_type: str | None = None
    formula: str | None = None
    formula_tags: str | None = None
    url: AnyUrl | None = None
    hardness_min: int | None = None
    hardness_max: int | None = None
    crystal_form: str | None = None
    mineral_color: str | None = None
    lustre: str | None = None
    provenance: Provenance | None = None
    elements: list[Element] | None = None

    @property
    def name(self) -> str:
        return self.mineral

    @staticmethod
    def to_elements(formula: str) -> list[Element]:
        """Convert a chemical formula to a list of elements."""

        legal_element_suffix = "abcdefghiklmnoprstuvy"

        elements = []
        i = 0
        while i < len(formula):
            if formula[i].isupper():
                # 2-letter element
                if i + 1 < len(formula) and formula[i + 1] in legal_element_suffix:
                    elements.append(formula[i : i + 2])
                    i += 2
                # 1-letter element
                else:
                    elements.append(formula[i])
                    i += 1
            else:
                # Just skip the trash
                i += 1

        # Deduplicate and sort elements
        elements = sorted(set(elements))

        # Validation against `Element` enum
        valid_elements = []
        for element in elements:
            try:
                valid_elements.append(Element(element))
            except ValueError:
                logging.warning(
                    f"When processing formula: {formula}, element: '{element}' not recognized"
                )
                pass

        return valid_elements

    async def hydrate(self) -> None:
        """Hydrate Mineral from macrostrat."""
        hit = await macrostrat.get_records(
            entity_type=macrostrat.EntityType.MINERAL, name=self.name, exact=True
        )

        if hit:
            # Process Macrostrat Mineral
            hit = hit[0]
            macrostrat_version = hit.pop("macrostrat_version")

            # Load data into model
            for k, v in hit.items():
                setattr(self, k, v)

            self.provenance = Provenance(
                source_name="Macrostrat",
                source_version=macrostrat_version,
                source_url=f"{macrostrat.BASE_URL}/defs/minerals?mineral_id={hit['mineral_id']}",
                previous=self.provenance,
            )

        else:
            # Process USGS Exclusive Minerals
            self.formula = USGS_FORMULAS.get(self.name.lower())
            if not self.formula:
                return

            self.provenance = Provenance(
                source_name="Mindat",
                source_version=1.0,
                source_url="https://api.mindat.org/",
                previous=self.provenance,
            )

        # Convert formula to elements
        if self.formula:
            self.elements = self.to_elements(self.formula)


def validate_longitude(v: float) -> float:
    assert -180 <= v <= 180, f"{v} is not a valid longitude"
    return v


def validate_latitude(v: float) -> float:
    assert -90 <= v <= 90, f"{v} is not a valid latitude"
    return v


def validate_location_name(v: str | list) -> str:
    """Force location name to be a string."""
    if isinstance(v, list):
        return ", ".join(v)
    return v


class Location(BaseModel):
    name: Annotated[str, BeforeValidator(validate_location_name)]
    lat: Annotated[float, AfterValidator(validate_latitude)] | None = None
    lon: Annotated[float, AfterValidator(validate_longitude)] | None = None
    provenance: Provenance | None = None

    async def hydrate(self, client: httpx.AsyncClient) -> None:
        """
        Hydrate Location (from geocode API)
        client: httpx.AsyncClient for geocode API use RateLimitedClient
        """
        self.lat, self.lon, request_url = await get_gps(self.name, client=client)
        if self.lat and self.lon:
            self.provenance = Provenance(
                source_name="GeocodeAPI",
                source_url=request_url,
                requested=datetime.now(),
                previous=self.provenance,
            )


class RelationshipTriplet(BaseModel):
    """Relationship between stratigraphy and location.

    Usage:
        raw_llm_output = ("Shakopee", "Minnesota", "is_in")
        subject, object, predicate = raw_llm_output
        triplet = RelationshipTriples(subject=subject, object=object, predicate=predicate)
    """

    subject: Location
    predicate: str  # relationship, str for now...
    object: Stratigraphy | Mineral
    provenance: Provenance | None = None


class GraphOutput(BaseModel):
    """LLM output should follow this format."""

    id: str | None = None
    paper_id: str | None = None
    hashed_text: str | None = None
    text_content: str | None = None
    triplets: list[RelationshipTriplet]

    async def hydrate(self, client: RateLimitedClient) -> None:
        """Hydrate all objects in the graph."""

        await asyncio.gather(
            *[triplet.subject.hydrate(client=client) for triplet in self.triplets],  # type: ignore
            *[triplet.object.hydrate() for triplet in self.triplets],  # type: ignore
        )
