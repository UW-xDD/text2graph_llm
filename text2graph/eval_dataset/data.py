import json
from pathlib import Path
from pydantic import TypeAdapter

from text2graph.schema import RelationshipTriplet
from text2graph.eval_dataset.schema import SQuADStratPipelineQuestion


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


def preprocess_dirty_ndjson(lines: list[str]) -> list[str]:
    """
    Preprocess a lines from a delimited JSON file where some lines were some missing a newlines
    """
    lines = [line.strip() for line in lines]
    proper_lines = []
    for line in lines:
        line_splits = line.split("}{")
        for line_split in line_splits:
            if line_split[0] != "{":
                line_split = "{" + line_split
            if line_split[-1] != "}":
                line_split = line_split + "}"
            proper_lines.append(line_split)

    return proper_lines


def squadquestions_from_json(
    squad_questions_json_path: Path,
) -> list[SQuADStratPipelineQuestion]:
    """
    Deserialize SQuAD questions from a line delimited JSON file.
    """
    with open(squad_questions_json_path, "r") as f:
        lines = f.readlines()

    print(f"Loaded {len(lines)} lines from {squad_questions_json_path}")
    lines = preprocess_dirty_ndjson(lines)
    print(f"Preprocessed to {len(lines)} lines")
    squad_questions = []
    for line in lines:
        try:
            squad_questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error loading line: {line}")
            print(e)

    return [SQuADStratPipelineQuestion(**sq) for sq in squad_questions]


def label_impossible(
    questions=list[SQuADStratPipelineQuestion],
) -> list[SQuADStratPipelineQuestion]:
    """
    set impossible flag True on questions that are missing subject, predicate, or object, else set to False
    """
    labelled_questions = []
    impossible_count = 0
    possible_count = 0
    for q in questions:
        if not q.object or not q.subject or not q.predicate:
            q.impossible = True
            q.object = None
            q.subject = None
            q.predicate = None
            impossible_count += 1
        else:
            q.impossible = False
            possible_count += 1
        labelled_questions.append(q)
    print(f"Impossible: {impossible_count}, Possible: {possible_count}")
    return labelled_questions
