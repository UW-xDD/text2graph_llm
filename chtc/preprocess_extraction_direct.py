import argparse
import asyncio
import logging
import os
import pickle
import sqlite3
from pathlib import Path

import tenacity
import weaviate
from dotenv import load_dotenv
from tqdm import tqdm

from text2graph.askxdd import get_weaviate_client
from text2graph.llm import ask_llm

logging.basicConfig(level=logging.ERROR)

load_dotenv()

DB_NAME = "triplets"
DB_PATH = os.getenv("DB_PATH", "triplets.db")


def create_db() -> None:
    """Create local DB if it doesn't exist."""

    if Path(DB_PATH).exists():
        logging.debug("Database already exists.")
        return

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_NAME} (
            id CHAR(64) PRIMARY KEY,
            hashed_text CHAR(64),
            paper_id CHAR(32),
            triplets TEXT
        )
        """)
        conn.commit()


def insert_case(
    connection: sqlite3.Connection,
    id: str,
    hashed_text: str,
    paper_id: str,
    triplets: str,
) -> None:
    """Insert entity into local DB."""

    cur = connection.cursor()
    cur.execute(
        f"""
    INSERT INTO {DB_NAME} (id, hashed_text, paper_id, triplets)
    VALUES (?, ?, ?, ?)
    """,
        (id, hashed_text, paper_id, triplets),
    )
    connection.commit()


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(3))
async def extract(text: str, doc_id: str) -> str:
    """Extract json GraphOutput from text."""

    graph = await ask_llm(text, model="mixtral", hydrate=False, doc_ids=[doc_id])
    return graph.model_dump_json(exclude_unset=True)  # type: ignore


def process_paragraph(
    id: str, weaviate_client: weaviate.Client, connection: sqlite3.Connection
) -> None:
    """Process extraction pipeline for a paragraph."""

    paragraph = weaviate_client.data_object.get_by_id(id)

    # Unpack useful fields
    text_content = paragraph["properties"]["text_content"]
    paper_id = paragraph["properties"]["paper_id"]
    hashed_text = paragraph["properties"]["hashed_text"]

    try:
        triplets = asyncio.run(extract(text_content, paper_id))
        output = {
            "id": id,
            "hashed_text": hashed_text,
            "paper_id": paper_id,
            "triplets": triplets,
        }
    except tenacity.RetryError:
        logging.error(f"Failed to extract {id}.")
        return

    logging.info(f"Extracted paragraph {id}: {output}")
    try:
        insert_case(connection=connection, **output)
    except Exception as e:
        logging.error(f"Failed to insert entities: {id} into the database: {e}")
        return

    logging.info(f"Inserted entities for paragraph {id} into the database.")


def get_processed_count(connection: sqlite3.Connection) -> int:
    """Get the number of cases already processed."""
    cur = connection.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {DB_NAME}")
    result = cur.fetchone()
    return result[0]


def get_all_processed_ids(connection: sqlite3.Connection) -> list[str]:
    """Get all processed ids from SQLITE."""
    cur = connection.cursor()
    cur.execute(f"SELECT id FROM {DB_NAME}")
    return [row[0] for row in cur.fetchall()]


def main(job_index: int = 0, batch_size: int = 2000):
    """Main function to process paragraphs."""

    sql_connection = sqlite3.Connection(DB_PATH)
    weaviate_client = get_weaviate_client()

    # Get ids to process
    batch_start_idx = job_index * batch_size

    all_ids_pickle = "/staging/clo36/text2graph/preprocess/geoarchive_paragraph_ids.pkl"
    with open(all_ids_pickle, "rb") as f:
        all_ids = pickle.load(f)
    batch_ids = all_ids[batch_start_idx : batch_start_idx + batch_size]

    ## Remove processed
    processed = get_all_processed_ids(sql_connection)
    batch_ids = [id for id in batch_ids if id not in processed]

    for id in tqdm(batch_ids):
        try:
            process_paragraph(id, weaviate_client, sql_connection)
        except Exception as e:
            logging.error(f"Failed to process paragraph {id}: {e}")
            continue

    logging.info(f"Finished processing batch {job_index=}")
    sql_connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int)
    main(**vars(parser.parse_args()))
