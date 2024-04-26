import argparse
import asyncio
import logging
import os
import pickle

import pandas as pd
import tenacity
import weaviate
from dotenv import load_dotenv
from sqlalchemy import (
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from tqdm import tqdm

from text2graph.askxdd import get_weaviate_client
from text2graph.llm import ask_llm

logging.basicConfig(level=logging.ERROR)


load_dotenv()
TURSO_DB_URL = os.getenv("TURSO_DB_URL")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
ENGINE = create_engine(
    f"sqlite+{TURSO_DB_URL}/?authToken={TURSO_AUTH_TOKEN}&secure=true",
    connect_args={"check_same_thread": False},
    echo=False,
)


class Base(DeclarativeBase):
    pass


class Triplets(Base):
    """Triplets table ORM definition."""

    __tablename__ = "triplets"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    hashed_text: Mapped[str] = mapped_column(String(64))
    paper_id: Mapped[str] = mapped_column(String(32))
    triplets: Mapped[str] = mapped_column(Text)


def hard_reset() -> None:
    """Wipe the database and re-create."""
    Base.metadata.drop_all(ENGINE)
    Base.metadata.create_all(ENGINE)


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(3))
def push(objects: list[Triplets]) -> None:
    """Push data to Turso."""

    with Session(ENGINE).no_autoflush as session:
        for i, commit in enumerate(objects):
            logging.info(f"Pushing {commit}")
            session.merge(commit)
            if (i + 1) % 100 == 0:  # commit every 100 items
                session.flush()
                session.commit()
        session.flush()
        session.commit()  # commit remaining items


def get_all_processed_ids() -> list[str]:
    """Get all processed ids from SQLITE."""

    batch_size = 500
    with ENGINE.connect() as conn:
        total_rows = conn.execute(text("SELECT COUNT(*) FROM triplets")).fetchone()
        if total_rows is None:
            return []
        total_rows = total_rows[0]

        processed_ids = []
        for i in range(0, total_rows, batch_size):
            print(f"Fetching rows {i} to {i+batch_size}")
            cur = conn.execute(
                text(f"SELECT id FROM triplets LIMIT {batch_size} OFFSET {i}")
            )
            processed_ids.extend([row[0] for row in cur.fetchall()])
    return processed_ids


def export(table: str) -> pd.DataFrame | None:
    """Export a table from Turso to a DataFrame."""

    batch_size = 500
    with ENGINE.connect() as conn:
        total_rows = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
        if total_rows is None:
            return None
        total_rows = total_rows[0]

        dfs = []
        for i in range(0, total_rows, batch_size):
            print(f"Fetching rows {i} to {i+batch_size}")
            df = pd.read_sql(f"SELECT * FROM repo LIMIT {batch_size} OFFSET {i}", conn)
            dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)


##### MAIN JOB #####


@tenacity.retry(wait=tenacity.wait_fixed(5), stop=tenacity.stop_after_attempt(3))
async def extract(text: str, doc_id: str) -> str:
    """Extract json GraphOutput from text."""

    graph = await ask_llm(text, model="mixtral", hydrate=False, doc_ids=[doc_id])
    return graph.model_dump_json(exclude_unset=True)  # type: ignore


def process_paragraph(
    id: str, weaviate_client: weaviate.Client
) -> dict[str, str] | None:
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
    return output


def main(job_index: int = 0, batch_size: int = 2000):
    """Main function to process paragraphs."""

    weaviate_client = get_weaviate_client()

    # Get ids to process
    batch_start_idx = job_index * batch_size

    all_ids_pickle = "/staging/clo36/text2graph/preprocess/geoarchive_paragraph_ids.pkl"
    with open(all_ids_pickle, "rb") as f:
        all_ids = pickle.load(f)
    batch_ids = all_ids[batch_start_idx : batch_start_idx + batch_size]

    ## Remove processed
    processed = get_all_processed_ids()
    batch_ids = [id for id in batch_ids if id not in processed]

    for id in tqdm(batch_ids):
        try:
            out = process_paragraph(id, weaviate_client)
            if out is None:
                continue
            push([Triplets(**out)])

        except Exception as e:
            logging.error(f"Failed to process paragraph {id}: {e}")
            continue

    logging.info(f"Finished processing batch {job_index=}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_index", type=int)
    main(**vars(parser.parse_args()))
