from pydantic import BaseModel
from enum import Enum

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
    strat_name_long: str
    rank: str
    strat_name_id: int
    concept_id: int
    bed: str
    bed_id: int
    mbr: str
    mbr_id: int
    fm: str
    fm_id: int
    subgp: str
    subgp_id: int
    gp: str
    gp_id: int
    sgp: str
    sgp_id: int
    b_age: float
    t_age: float
    b_period: str
    t_period: str
    c_interval: str
    t_units: int
    ref_id: int

class Location(BaseModel):
    name: str
    lat: float | None
    lon: float | None


class RelationshipTriples(BaseModel):
    subject: str | Stratigraphy
    predicate: str  # relationship, str for now...
    object: str | Location

class GraphOutput(BaseModel):
    """LLM output should follow this format."""
    triplets: list[RelationshipTriples]