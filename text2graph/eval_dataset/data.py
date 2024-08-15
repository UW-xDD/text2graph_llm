import json
from pathlib import Path
from pydantic import TypeAdapter

from text2graph.schema import RelationshipTriplet


def load_triplet_from_json(json_dict: dict) -> RelationshipTriplet:
    """
    Load a RelationshipTriplet from a JSON dictionary.
    """
    triplet_adapter = TypeAdapter(RelationshipTriplet)
    triplet_db_info = json_dict["provenance"]["additional_values"].pop(
        "triplet_db", dict()
    )
    rt = triplet_adapter.validate_python(json_dict)
    rt.provenance.additional_values["triplet_db"] = triplet_db_info
    return rt


def load_triplets_from_dir(
    hydrated_json_paths: list[Path],
) -> list[RelationshipTriplet]:
    """
    Load all triplets from a directory of hydrated JSON files.
    """

    triplets = []
    for hydrated_json_path in hydrated_json_paths:
        with open(hydrated_json_path, "r") as f:
            for triplet_json in json.load(f):
                triplets.append(load_triplet_from_json(triplet_json))

    print(f"{len(triplets)} triplets loaded from {len(hydrated_json_paths)} files.")
    return triplets
