import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import tenacity
import weaviate
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)

load_dotenv()
CHTC_LLM_HOST = os.getenv("CHTC_LLM_HOST", "")
CHTC_LLM_PORT = os.getenv("CHTC_LLM_PORT", "")
CHTC_LLM_API_KEY = os.getenv("CHTC_LLM_API_KEY", "")
CHTC_LLM_API_URL = f"http://{CHTC_LLM_HOST}:{CHTC_LLM_PORT}"
WEAVIATE_APIKEY = os.getenv("WEAVIATE_APIKEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")

WEAVIATE_CLIENT = weaviate.Client(
    WEAVIATE_URL, weaviate.AuthApiKey(api_key=WEAVIATE_APIKEY)
)


def create_db(local_db: str = "entities.db") -> None:
    """Create local DB if it doesn't exist."""
    if Path(local_db).exists():
        return

    with sqlite3.connect(local_db) as conn:
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            hashed_text CHAR(64) PRIMARY KEY,
            paper_id CHAR(32),
            locations TEXT,
            stratigraphic_names TEXT,
            lithologies TEXT
        )
        """)
        conn.commit()


def insert_case(
    hashed_text: str,
    paper_id: str,
    locations: str,
    stratigraphic_names: str,
    lithologies: str,
) -> None:
    """Insert entity into local DB."""
    with sqlite3.connect("entities.db") as conn:
        cur = conn.cursor()
        cur.execute(
            """
        INSERT INTO entities (hashed_text, paper_id, locations, stratigraphic_names, lithologies)
        VALUES (?, ?, ?, ?, ?)
        """,
            (hashed_text, paper_id, locations, stratigraphic_names, lithologies),
        )
        conn.commit()


def in_db(hashed_text: str) -> bool:
    """Check if entity is already in the local DB."""
    with sqlite3.connect("entities.db") as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT hashed_text FROM entities WHERE hashed_text=?", (hashed_text,)
        )
        result = cur.fetchone()
    return result is not None


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(5))
def extract(text: str) -> dict:
    """Extract entities from text."""

    auth_headers = {"Api-Key": CHTC_LLM_API_KEY}
    payload = {
        "model": "my_model",
        "messages": [
            {
                "role": "system",
                "content": 'Try to extract all locations, stratigraphic names, lithologies from the text provided. Reply in JSON format with the following structure: {"locations:": "", "stratigraphic_names": "", "lithologies": ""}. If you can\'t find any of the requested information, leave the corresponding field empty.Do not include any other information in the response.',
            },
            {"role": "user", "content": text},
        ],
    }
    response = requests.post(
        f"{CHTC_LLM_API_URL}/v1/chat/completions", headers=auth_headers, json=payload
    )
    response.raise_for_status()
    content = json.loads(response.json()["_content"])
    return json.loads(content["choices"][0]["message"]["content"])


def get_batch(
    class_properties: list[str],
    class_name: str = "Paragraph",
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
                "valueText": ["criticalmaas"],
            }
        )
    )

    if offset is not None:
        query = query.with_offset(offset)

    return query.with_limit(batch_size).do()


def process_case(paragraph: dict) -> None:
    """Process extraction pipeline for a paragraph."""

    # Check if the paragraph has already been processed
    if in_db(paragraph["hashed_text"]):
        logging.info(f"Paragraph {paragraph['hashed_text']} already processed.")
        return

    try:
        entities = extract(paragraph["text_content"])
    except tenacity.RetryError:
        logging.error(
            f"Failed to extract entities for paragraph {paragraph['hashed_text']}."
        )
        return
    entities["hashed_text"] = paragraph["hashed_text"]
    entities["paper_id"] = paragraph["paper_id"]

    logging.info(
        f"Extracted entities for paragraph {paragraph['hashed_text']}: {entities}"
    )
    try:
        insert_case(**entities)
    except Exception as e:
        logging.error(
            f"Failed to insert entities for paragraph {paragraph['hashed_text']} into the database: {e}"
        )
        return
    logging.info(
        f"Inserted entities for paragraph {paragraph['hashed_text']} into the database."
    )


def main(batch_size: int = 8):
    create_db()
    n = 0
    while True:
        batch = get_batch(
            ["hashed_text", "text_content", "paper_id"], batch_size=batch_size, offset=n
        )
        if len(batch["data"]["Get"]["Paragraph"]) == 0:
            break

        paragraphs = batch["data"]["Get"]["Paragraph"]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            list(executor.map(process_case, paragraphs))

        n += batch_size


if __name__ == "__main__":
    main()
