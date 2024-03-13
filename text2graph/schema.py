from pydantic import BaseModel
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated


class Lithology(BaseModel):
    lith_id: int
    name: str
    type: str
    group: str
    _class: str
    color: str
    fill: int
    t_units: int


class Stratigraphy(BaseModel):
    strat_name: str
    strat_name_long: str | None
    rank: str | None
    strat_name_id: int | None
    concept_id: int | None
    bed: str | None
    bed_id: int | None
    mbr: str | None
    mbr_id: int | None
    fm: str | None
    fm_id: int | None
    subgp: str | None
    subgp_id: int | None
    gp: str | None
    gp_id: int | None
    sgp: str | None
    sgp_id: int | None
    b_age: float | None
    t_age: float | None
    b_period: str | None
    t_period: str | None
    c_interval: str | None
    t_units: int | None
    ref_id: int | None


def valid_longitude(v: float) -> float:
    assert -180 <= v <= 180, f"{v} is not a valid longitude"
    return v


def valid_latitude(v: float) -> float:
    assert -90 <= v <= 90, f"{v} is not a valid latitude"
    return v


class Location(BaseModel):
    name: str
    lat: Annotated[float, AfterValidator(valid_latitude)] | None
    lon: Annotated[float, AfterValidator(valid_longitude)] | None


class RelationshipTriples(BaseModel):
    subject: str | Stratigraphy
    predicate: str  # relationship, str for now...
    object: str | Location


class GraphOutput(BaseModel):
    """LLM output should follow this format."""

    triplets: list[RelationshipTriples]
