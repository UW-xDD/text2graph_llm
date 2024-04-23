from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from uuid import UUID, uuid4

import httpx
from pydantic import AnyUrl, BaseModel, BeforeValidator, Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from text2graph import macrostrat
from .geolocation.geocode import get_gps, RateLimitedClient
from .macrostrat import get_lith_records, get_strat_records


class Provenance(BaseModel):
    """class for collecting data source information"""

    id: UUID = Field(default_factory=uuid4)
    source_name: str
    source_url: str | None = None
    source_version: str | int | float | None = None
    requested: datetime = datetime.now(UTC)
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
            hit = await get_lith_records(self.name, exact=True)[0]
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
    strat_name: str
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

    async def hydrate(self) -> None:
        """Hydrate Stratigraphy from macrostrat."""
        try:
            hit = await get_strat_records(self.strat_name, exact=False)
            hit = hit[0]
        except (ValueError, IndexError):
            logging.info(f"No records found for stratigraphy '{self.strat_name}'")
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

    subject: str | Location
    predicate: str  # relationship, str for now...
    object: str | Stratigraphy
    provenance: Provenance | None = None

    def model_post_init(self, __context) -> None:
        if isinstance(self.subject, str):
            self.subject = Location(name=self.subject, provenance=self.provenance)
        if isinstance(self.object, str):
            self.object = Stratigraphy(
                strat_name=self.object, provenance=self.provenance
            )


class GraphOutput(BaseModel):
    """LLM output should follow this format."""

    triplets: list[RelationshipTriplet]

    async def hydrate(self) -> None:
        """Hydrate all objects in the graph."""

        async with RateLimitedClient(interval=1.0, count=1, timeout=30) as client:
            await asyncio.gather(
                *[triplet.subject.hydrate(client=client) for triplet in self.triplets],
                *[triplet.object.hydrate() for triplet in self.triplets],
            )
