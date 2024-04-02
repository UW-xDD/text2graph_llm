from __future__ import annotations
import asyncio
import logging
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from .geolocation.serpapi import get_gps
from .macrostrat import get_lith_records, get_strat_records


class Entity(BaseModel):
    """a thing in PROV, e.g. webpage, chart, spellchecker..."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    was_derived_from: Entity
    was_generated_by: Activity
    was_attributed_to: Agent


class Activity(BaseModel):
    """how a PROV entity comes into existence"""

    id: UUID = Field(default_factory=uuid4)
    name: str
    used: Entity
    was_associated_with: Agent


class Agent(BaseModel):
    """person, piece of software, an inanimate object, an organization, anything assigned responsibility in PROV"""

    id: UUID = Field(default_factory=uuid4)
    name: str


class Provenance(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str
    source_name: str
    source_url: str | None
    source_version: str | None
    started_at: datetime
    ended_at: datetime


class Lithology(BaseModel):
    name: str
    lith_id: int | None = None
    type: str | None = None
    group: str | None = None
    _class: str | None = None
    color: str | None = None
    fill: int | None = None
    t_units: int | None = None
    provenance: Provenance

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
    provenance: Provenance

    async def hydrate(self) -> None:
        """Hydrate Stratigraphy from macrostrat."""
        try:
            hit = await get_strat_records(self.strat_name, exact=False)
            hit = hit[0]
        except (ValueError, IndexError):
            logging.info(f"No records found for stratigraphy '{self.strat_name}'")
            return

        # Load data into model
        for k, v in hit.items():
            setattr(self, k, v)


def valid_longitude(v: float) -> float:
    assert -180 <= v <= 180, f"{v} is not a valid longitude"
    return v


def valid_latitude(v: float) -> float:
    assert -90 <= v <= 90, f"{v} is not a valid latitude"
    return v


class Location(BaseModel):
    name: str
    lat: Annotated[float, AfterValidator(valid_latitude)] | None = None
    lon: Annotated[float, AfterValidator(valid_longitude)] | None = None
    provenance: Provenance

    async def hydrate(self) -> None:
        self.lat, self.lon = await get_gps(self.name)


def strip_tuples_return_float(value: str | float | tuple[float] | list[float]) -> float:
    """
    take first value of any tuple passed and conver to float. if conversion failure return None
    :param value: value to 'un-tuple' and convert
    :return: float or None
    """
    try:
        untupled = value[0]
    except (TypeError, IndexError):
        untupled = value
    try:
        result = float(untupled)
    except (ValueError, TypeError):
        result = None
    return result


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
    provenance: Provenance

    def model_post_init(self, __context) -> None:
        if isinstance(self.subject, str):
            self.subject = Location(name=self.subject)
        if isinstance(self.object, str):
            self.object = Stratigraphy(strat_name=self.object)


class GraphOutput(BaseModel):
    """LLM output should follow this format."""

    triplets: list[RelationshipTriplet]

    async def hydrate(self) -> None:
        """Hydrate all objects in the graph."""
        await asyncio.gather(
            *[triplet.subject.hydrate() for triplet in self.triplets],
            *[triplet.object.hydrate() for triplet in self.triplets],
        )
