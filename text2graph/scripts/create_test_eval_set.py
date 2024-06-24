import time
import json
import typer
import sqlite3
import httpx
import asyncio
import numpy as np
import aiofiles as aiof
from pathlib import Path
from dotenv import load_dotenv
from tqdm.autonotebook import tqdm
from tqdm.asyncio import tqdm_asyncio
from pydantic import ValidationError
from typing import Any, Generator
import logging

from text2graph.schema import GraphOutput
from text2graph.apiutils import RateLimitedClient


# FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
FORMAT = "%(asctime)s [%(threadName)s] [%(levelname)s]  %(message)s"
logger = logging.getLogger(__name__)
logging.basicConfig(
    # filename="logs/log.log",
    encoding="utf-8",
    level=logging.INFO,
    format=FORMAT,
)

load_dotenv()

COLUMN_LOOKUP = {
    "index": 0,
    "id": 1,
    "job_id": 2,
    "hashed_txt": 3,
    "paper_id": 4,
    "triplets": 5,
}


def main(
    db_path: Path,
    output_path: Path,
    chunksize: int = 100,
    geoloc_limit: int | None = None,
    limit: int | None = None,
    client_timeout: int = 30,
    skip_ahead: int | None = None,
) -> None:
    """
    fetch all triplets from sqllite database of graphoutputs, hydrate (geolocate with GeoCode API, and Macrostrat)
    all triplets, write triplets to disk. those with location information as keepers, those without - nonkeepers.
    :param db_path: Path to sqllitedb on disk
    :param output_path: dir to write results to
    :param chunksize: number of graphouputs to fetch from DB per loop
    :param geoloc_limit: whatever number of free gelocation api calls you have left this month
    :param limit: max number of graphouts to pull from the db on disk
    :param client_timeout: timeout value to use for httpx clients
    :param skip_ahead: graph_output index to skip to
    """
    if not db_path:
        db_path = Path("data/data_dump_240430.db")
    cur = get_db_cursor(db_path)
    if limit < chunksize:
        raise ValueError(f"{limit=} must be >= {chunksize}")
    db_triplet_chunks = get_all_db_table_tuples_chunks(
        cur=cur, chunksize=chunksize, limit=limit
    )
    total_triplet_count = 0
    for i, triplets_chunk in enumerate(db_triplet_chunks):
        logger.info(f"processing batch of triplets above #{total_triplet_count}")
        filtered_graphoutputs, _ = filter_triplet_db_tuples(triplets_chunk)
        triplets_in_batch_count = len(filtered_graphoutputs)
        filename_offset = i * chunksize

        if total_triplet_count + triplets_in_batch_count < skip_ahead:
            total_triplet_count += triplets_in_batch_count
            continue

        if total_triplet_count + triplets_in_batch_count >= geoloc_limit:
            object_client, subject_client = create_api_clients(timeout=client_timeout)
            filtered_graphoutputs = filtered_graphoutputs[
                : geoloc_limit - total_triplet_count
            ]
            _ = asyncio.run(
                run_all_triplets(
                    filtered_graphoutputs=filtered_graphoutputs,
                    object_client=object_client,
                    subject_client=subject_client,
                    output_path=output_path,
                    filename_offset=filename_offset,
                )
            )
            logger.info(f"Stopping! Hit specified geoloc limit: {geoloc_limit}")
            break

        else:
            object_client, subject_client = create_api_clients(timeout=client_timeout)
            _ = asyncio.run(
                run_all_triplets(
                    filtered_graphoutputs=filtered_graphoutputs,
                    object_client=object_client,
                    subject_client=subject_client,
                    output_path=output_path,
                    filename_offset=filename_offset,
                )
            )
            time.sleep(1.5)
        total_triplet_count += triplets_in_batch_count


def get_db_cursor(db_path: str) -> sqlite3.Cursor:
    con = sqlite3.connect(db_path)
    return con.cursor()


def get_all_db_table_tuples(
    cur: sqlite3.Cursor,
) -> list[tuple[Any, Any, Any, Any, Any]]:
    res = cur.execute("""SELECT * FROM triplets;""")
    return res.fetchall()


def get_all_db_table_tuples_chunks(
    cur: sqlite3.Cursor, chunksize: int, limit: int | None
) -> Generator[list[tuple[Any, Any, Any, Any, Any]], None, None]:
    """
    Return generator for reading consecutive chunks of data from a table as
    conn : DB connection object
    chunksize : int
        Number of rows to return in each call to the generator.
    """

    if not limit:
        limit = np.inf

    offset = 0
    while offset <= limit:
        query = f"SELECT * FROM triplets LIMIT {chunksize} OFFSET {offset}"
        result = cur.execute(query).fetchall()
        if len(result):
            if offset + len(result) >= limit:
                yield result[: limit - offset]
            yield result
        else:
            raise StopIteration
        offset += chunksize


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
    logger.info(f"total_triplets_output: {sum(triplet_counts)}")
    return filtered_graphoutputs, validation_error_indexes


async def run_all_triplets(
    filtered_graphoutputs: list[GraphOutput],
    object_client: RateLimitedClient,
    subject_client: RateLimitedClient,
    output_path: Path,
    filename_offset: int,
    mini_batch_size: int = 10,
) -> None:
    """
    run all api requests for each batch of filtered_graphoutputs in mini batches
    :param filtered_graphoutputs: graphoutputs to hydrate
    :param object_client: httpx client for Macrostrat
    :param subject_client: httpx client for Geocoding API
    :param output_path: where to write results
    :param filename_offset: track the super batch / the current graphoutput idx in the whole dataset
    :param mini_batch_size: number of graphoutputs to hydrate simultaneously
    """
    mini_batches = [
        filtered_graphoutputs[x : x + mini_batch_size]
        for x in range(0, len(filtered_graphoutputs), mini_batch_size)
    ]
    for mini_batch_i, filtered_graphoutputs_mini_batch in enumerate(mini_batches):
        await tqdm_asyncio.gather(
            *[
                hydrate_and_split_one_graphoutput(
                    go_idx=(mini_batch_size * mini_batch_i) + i + filename_offset,
                    output_path=output_path,
                    go=go,
                    object_client=object_client,
                    subject_client=subject_client,
                )
                for i, go in enumerate(filtered_graphoutputs_mini_batch)
            ]
        )


def create_api_clients(timeout: float) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
    """
    Create API clients from object(macrostrat) and subject(geocodeAPI)
    :param timeout: timeout limit for each client, suggest 30 seconds, but make a decision.
    :return: two async clients.
    """
    timeout = httpx.Timeout(timeout, read=120.0)
    object_client = RateLimitedClient(
        interval=0.1, count=1, timeout=timeout
    )  # Daven suggested macrostrat api rate limit
    subject_client = RateLimitedClient(
        interval=1.5, count=1, timeout=timeout
    )  # max geocode api rate is 1/s, use 1.5 to be safe
    return object_client, subject_client


async def hydrate_and_split_one_graphoutput(
    go_idx: int,
    output_path: Path,
    go: GraphOutput,
    object_client: RateLimitedClient,
    subject_client: RateLimitedClient,
) -> None:
    keeper_triplets = []
    not_keeper_triplets = []
    try:
        await go.hydrate(object_client=object_client, subject_client=subject_client)
    except Exception as e:
        logger.warning(
            f"failed to hydrate with {e} on triplet number: {go_idx} triplet: {go}"
        )
        return None

    for triplet in go.triplets:
        c = triplet.model_dump()
        if triplet.object.t_units and triplet.object.t_units >= 1:
            keeper_triplets.append(c)
        else:
            not_keeper_triplets.append(c)

    for name, data in zip(
        ["keeper", "not_keeper"], [keeper_triplets, not_keeper_triplets]
    ):
        if data:
            output_filename = output_path / name / (str(go_idx) + ".json")
            async with aiof.open(output_filename, "w") as out:
                logger.info(f"writing to {output_filename}")
                await out.write(json.dumps(data))
                await out.flush()


if __name__ == "__main__":
    typer.run(main)
