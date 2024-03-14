import logging
from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated
from .macrostrat import get_strat_records, get_lith_records


class Lithology(BaseModel):
    name: str
    lith_id: int | None = None
    type: str | None = None
    group: str | None = None
    _class: str | None = None
    color: str | None = None
    fill: int | None = None
    t_units: int | None = None

    def hydrate(self) -> None:
        """Hydrate Lithology from macrostrat."""
        try:
            hit = get_lith_records(self.name, exact=True)[0]
        except (ValueError, IndexError):
            logging.info(f"No records found for lithology '{self.name}'")
            return

        for k, v in hit.items():
            if k == "class":
                setattr(self, "_class", v)
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

    def model_post_init(self, __context) -> None:
        self.hydrate()

    def hydrate(self) -> None:
        """Hydrate Stratigraphy from macrostrat."""
        try:
            hit = get_strat_records(self.strat_name, exact=True)[0]
        except (ValueError, IndexError):
            logging.info(f"No records found for stratigraphy '{self.strat_name}'")
            return

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

    def hydrate(self) -> None:
        raise NotImplementedError  # TODO: Use any geocoding API to hydrate location, also add `model_post_init`


class RelationshipTriples(BaseModel):
    """Relationship between stratigraphy and location.

    Usage:
        raw_llm_output = ("Shakopee", "Minnesota", "is_in")
        subject, object, predicate = raw_llm_output
        triplet = RelationshipTriples(subject=subject, object=object, predicate=predicate)
    """

    subject: str | Stratigraphy
    predicate: str  # relationship, str for now...
    object: str | Location

    def model_post_init(self, __context) -> None:
        if isinstance(self.subject, str):
            self.subject = Stratigraphy(strat_name=self.subject)
        if isinstance(self.object, str):
            self.object = Location(name=self.object)


class GraphOutput(BaseModel):
    """LLM output should follow this format."""

    triplets: list[RelationshipTriples]
