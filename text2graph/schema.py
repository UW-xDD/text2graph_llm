
from pydantic import BaseModel, ValidationError
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

def valid_longitude(v: float) -> float:
    assert -180 <= v <= 180, f'{v} is not a valid longitude'
    return v

def valid_latitude(v: float) -> float:
    assert -90 <= v <= 90, f'{v} is not a valid latitude'
    return v

Latitude = Annotated[float, AfterValidator(valid_latitude)]
Longitude = Annotated[float, AfterValidator(valid_longitude)]

class Location(BaseModel):
    name: str
    lat: Latitude | None
    lon: Longitude | None
