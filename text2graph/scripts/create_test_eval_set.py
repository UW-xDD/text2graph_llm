import json
from pathlib import Path

import typer
import sqlite3
import asyncio
from typing import Any
from tqdm.auto import tqdm
from pydantic import ValidationError

from text2graph.schema import GraphOutput, RelationshipTriplet
from text2graph.geolocation.geocode import RateLimitedClient


COLUMN_LOOKUP = {
    "index": 0,
    "id": 1,
    "job_id": 2,
    "hashed_txt": 3,
    "paper_id": 4,
    "triplets": 5,
}


def main(db_path: Path, output_path: Path, limit: int | None = None) -> None:
    if not db_path:
        db_path = Path("data/data_dump_240430.db")
    cur = get_db_cursor(db_path)
    triplet_db_tuples = get_all_db_table_tuples(cur)
    if limit:
        triplet_db_tuples = triplet_db_tuples[:limit]
    filtered_graphoutputs, _ = filter_triplet_db_tuples(triplet_db_tuples)
    keepers, notkeepers = asyncio.run(hydrate_and_split_triplets(filtered_graphoutputs))

    for data in [keepers, notkeepers]:
        with open(output_path, "w") as f:
            json.dump(data, f)


def get_db_cursor(db_path: str) -> sqlite3.Cursor:
    con = sqlite3.connect(db_path)
    return con.cursor()


def get_all_db_table_tuples(
    cur: sqlite3.Cursor,
) -> list[tuple[Any, Any, Any, Any, Any]]:
    res = cur.execute("""SELECT * FROM triplets;""")
    return res.fetchall()


def filter_triplet_db_tuples(
    triplet_db_tuples: list[tuple[Any, Any, Any, Any, Any]],
) -> tuple[list[GraphOutput], list[int]]:
    """
    read db tuples, ignore those with no triplets and those with validation errs
    :param triplet_db_tuples: listof tuples from sqlite3 db triplet table
    :return: list of graphoutput objects, list of triplet_tuples indexes with validation errors
    """
    filtered_graphoutputs = []
    triplet_counts = []
    validation_error_indexes = []
    for i, graphoutput in tqdm(enumerate(triplet_db_tuples)):
        if not graphoutput[COLUMN_LOOKUP["triplets"]]:
            continue
        try:
            go = GraphOutput.model_validate_json(graphoutput[COLUMN_LOOKUP["triplets"]])
        except ValidationError:
            validation_error_indexes.append(i)
            continue

        if go.triplets:
            for triplet in go.triplets:
                triplet.provenance.additional_values["triplet_db"] = {
                    "triplet_db_pk": graphoutput[COLUMN_LOOKUP["id"]],
                    "paragraph_hash": COLUMN_LOOKUP["hashed_txt"],
                    "xdd_id": graphoutput[COLUMN_LOOKUP["paper_id"]],
                }
            triplet_counts.append(len(go.triplets))
            filtered_graphoutputs.append(go)
    print(f"total_triplets_output: {sum(triplet_counts)}")
    return filtered_graphoutputs, validation_error_indexes


async def hydrate_and_split_triplets(
    filtered_graphoutputs: list[GraphOutput],
) -> tuple[list[RelationshipTriplet], list[RelationshipTriplet]]:
    keeper_triplets = []
    not_keeper_triplets = []
    keeper_counts = []
    not_keeper_counts = []
    for i, go in tqdm(enumerate(filtered_graphoutputs[::1000])):
        await go.hydrate(client=RateLimitedClient(interval=1.5))
        for triplet in go.triplets:
            if triplet.object.t_units and triplet.object.t_units >= 1:
                keeper_triplets.append(triplet)
            else:
                not_keeper_triplets.append(triplet)

        keeper_counts.append(len(keeper_triplets))
        not_keeper_counts.append(len(not_keeper_triplets))

    return keeper_triplets, not_keeper_triplets


if __name__ == "__main__":
    typer.run(main)
