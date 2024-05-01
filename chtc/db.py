import logging
import os

import pandas as pd
import tenacity
from dotenv import load_dotenv
from sqlalchemy import (
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

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
    job_id: Mapped[int] = mapped_column(Integer, index=True)
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


def get_all_processed_ids(job_index: int, max_size: int = 2000) -> list[str]:
    """Get all processed ids from SQLITE."""

    query = text(
        f"SELECT id FROM triplets WHERE job_id = {job_index} LIMIT {max_size};"
    )
    with Session(ENGINE) as session:
        responses = session.execute(query).fetchall()
    return [r[0] for r in responses]


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
            df = pd.read_sql(
                f"SELECT * FROM {table} LIMIT {batch_size} OFFSET {i}", conn
            )
            dfs.append(df)

    return pd.concat(dfs).reset_index(drop=True)
