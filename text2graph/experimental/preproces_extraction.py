import argparse
import asyncio
import logging
import os
import sqlite3
import threading
from pathlib import Path
from queue import Queue

import tenacity
import weaviate
from dotenv import load_dotenv

from text2graph.llm import ask_llm

logging.basicConfig(level=logging.INFO)

load_dotenv()
WEAVIATE_APIKEY = os.getenv("WEAVIATE_APIKEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

WEAVIATE_CLIENT = weaviate.Client(
    WEAVIATE_URL, weaviate.AuthApiKey(api_key=WEAVIATE_APIKEY)
)

DB_NAME = "triplets"
DB_PATH = "tmp/triplets.db"


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
    id: str,
    hashed_text: str,
    paper_id: str,
    triplets: str,
) -> None:
    """Insert entity into local DB."""

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            f"""
        INSERT INTO {DB_NAME} (id, hashed_text, paper_id, triplets)
        VALUES (?, ?, ?, ?)
        """,
            (id, hashed_text, paper_id, triplets),
        )
        conn.commit()


def in_db(id: str) -> bool:
    """Check if entity is already in the local DB."""

    logging.debug(f"Checking if {id} is in the database.")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT id FROM {DB_NAME} WHERE id=?", (id,))
        result = cur.fetchone()
    return result is not None


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(3))
async def extract(text: str) -> str:
    """Extract json GraphOutput from text."""

    graph = await ask_llm(text, model="mixtral")
    return graph.model_dump_json(exclude_unset=True)  # type: ignore


def get_batch(
    class_properties: list[str],
    class_name: str = "Paragraph",
    topic: str = "geoarchive",
    batch_size: int = 8,
    offset: int | None = None,
):
    """Get paragraphs from Weaviate."""
    if "topic_list" not in class_properties:
        class_properties.append("topic_list")

    query = (
        WEAVIATE_CLIENT.query.get(class_name, class_properties)
        .with_additional(["id"])
        .with_where(
            {
                "path": "topic_list",
                "operator": "ContainsAny",
                "valueText": [topic],
            }
        )
    )

    if offset is not None:
        query = query.with_offset(offset)

    return query.with_limit(batch_size).do()


def process_paragraph(paragraph: dict) -> None:
    """Process extraction pipeline for a paragraph."""

    # Check if the paragraph has already been processed
    if in_db(paragraph["_additional"]["id"]):
        logging.info(f"Paragraph {paragraph['hashed_text']} already processed.")
        return

    data = {
        "id": paragraph["_additional"]["id"],
        "paper_id": paragraph["paper_id"],
        "hashed_text": paragraph["hashed_text"],
    }

    try:
        data["triplets"] = asyncio.run(extract(paragraph["text_content"]))
    except tenacity.RetryError:
        logging.error(f"Failed to extract {paragraph['hashed_text']}.")
        return

    logging.info(f"Extracted paragraph {paragraph['hashed_text']}: {data}")
    try:
        insert_case(**data)
    except Exception as e:
        logging.error(
            f"Failed to insert entities for paragraph {paragraph['hashed_text']} into the database: {e}"
        )
        return
    logging.info(
        f"Inserted entities for paragraph {paragraph['hashed_text']} into the database."
    )


def worker(queue: Queue) -> None:
    while True:
        paragraph = queue.get()
        if paragraph is None:
            break
        process_paragraph(paragraph)
        queue.task_done()


def get_processed_count() -> int:
    """Get the number of cases already processed."""
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {DB_NAME}")
        result = cur.fetchone()
    return result[0]


def main(batch_size: int = 300, n_workers: int = 30):
    # Create DB if it doesn't exist
    create_db()

    # Resume from the no. cases already processed
    try:
        n = get_processed_count()
    except Exception as e:
        logging.error(f"Failed to get the number of cases already processed: {e}")
        n = 0

    queue = Queue(maxsize=batch_size)
    threads = [threading.Thread(target=worker, args=(queue,)) for _ in range(n_workers)]

    for thread in threads:
        thread.start()

    while True:
        batch = get_batch(
            ["hashed_text", "text_content", "paper_id"], batch_size=batch_size, offset=n
        )
        paragraphs = batch["data"]["Get"]["Paragraph"]
        if not paragraphs:
            break

        for paragraph in paragraphs:
            queue.put(paragraph)

        n += batch_size

    for _ in range(n_workers):
        queue.put(None)

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--n_workers", type=int, default=30)
    main(**vars(parser.parse_args()))
