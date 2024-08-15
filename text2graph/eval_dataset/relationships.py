import json
import requests
import pyproj
from datetime import datetime
import dataclasses
from dataclasses import dataclass
from pathlib import Path
import logging


from text2graph.schema import RelationshipTriplet
from text2graph.askxdd import get_paragraph


@dataclass
class SpatialRelationship:
    subject_name: str
    subject_lat: float
    subject_lon: float
    predicate: str
    object_name: str
    object_lat: float
    object_lon: float
    doc_id: str
    hashed_text: str
    original_paragraph: str | None = None

    def has_all_coords(self) -> bool:
        return (
            self.object_lat is not None
            and self.object_lon is not None
            and self.subject_lat is not None
            and self.subject_lon is not None
        )

    def distance(self, geod: pyproj.Geod) -> float:
        """return subject to object distance in miles."""
        _, _, distance_in_meters = geod.inv(
            self.subject_lon, self.subject_lat, self.object_lon, self.object_lat
        )
        meters_to_miles = 0.000621371
        return distance_in_meters * meters_to_miles

    def midpoint(self, geod: pyproj.Geod) -> tuple[float, float]:
        """return the midpoint lat, lon between subject and object."""
        return geod.npts(
            self.subject_lon, self.subject_lat, self.object_lon, self.object_lat, 1
        )[0]

    def within_extent(self, extent: tuple[float, float, float, float]) -> bool:
        """return True if the object and or location is within the extent."""
        return (
            extent[0] <= self.object_lon <= extent[1]
            and extent[2] <= self.object_lat <= extent[3]
        ) or (
            extent[0] <= self.subject_lon <= extent[1]
            and extent[2] <= self.subject_lat <= extent[3]
        )


def fetch_object_location(triplet: RelationshipTriplet):
    strat_name_id = triplet.object.strat_name_id
    url = (
        f"https://macrostrat.org/api/units?strat_name_id={strat_name_id}&response=long"
    )
    response = requests.get(url)
    response.raise_for_status()
    response_json = response.json()
    lat, lon = None, None
    try:
        lat = response_json["success"]["data"][0]["clat"]
        lon = response_json["success"]["data"][0]["clng"]
    except (KeyError, IndexError):
        logging.warning(f"Failed to get location for {triplet.object.strat_name_long}")

    return lat, lon


def triplet_to_spatial_relationship(
    triplet: RelationshipTriplet, table_pk_to_hashed_text_lookup: dict[str, str]
) -> SpatialRelationship:
    subject_name = triplet.subject.name
    subject_lat = triplet.subject.lat
    subject_lon = triplet.subject.lon
    predicate = triplet.predicate
    object_name = triplet.object.strat_name_long
    start = datetime.now()
    object_lat, object_lon = fetch_object_location(triplet)
    end = datetime.now()
    macrostrat_timedelta = end - start
    doc_id = triplet.provenance.additional_values["doc_ids"][0]
    hashed_text = table_pk_to_hashed_text_lookup[
        triplet.provenance.additional_values["triplet_db"]["triplet_db_pk"]
    ]
    start = datetime.now()
    paragraph = get_paragraph(hashed_text=hashed_text)
    end = datetime.now()
    weaviate_time_delta = end - start
    print(
        f"Macrostrat API call took {macrostrat_timedelta}"
        f"Weaviate  API  call took {weaviate_time_delta}",
        end="\r",
    )
    return SpatialRelationship(
        subject_name,
        subject_lat,
        subject_lon,
        predicate,
        object_name,
        object_lat,
        object_lon,
        doc_id,
        hashed_text,
        paragraph,
    )


def dump_all_spatial_relationships(
    spatial_relationships: list[SpatialRelationship], output_path: Path
):
    """
    Dump a list of SpatialRelationships to a JSON file.
    """
    json_strings = []
    for sr in spatial_relationships:
        json_strings.append(json.dumps(dataclasses.asdict(sr)))
    with open(output_path, "w") as f:
        f.write("\n".join(json_strings))


def load_all_spatial_relationships(output_path: Path) -> list[SpatialRelationship]:
    """
    Load a list of SpatialRelationships from a JSON file.
    """
    with open(output_path, "r") as f:
        json_strings = f.readlines()
    return [SpatialRelationship(**json.loads(json_str)) for json_str in json_strings]
