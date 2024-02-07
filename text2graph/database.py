import sqlite3
from pathlib import Path
import os
import json
from dotenv import load_dotenv

load_dotenv()
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH")
if not SQLITE_DB_PATH:
    raise KeyError(
        f"SQLite database is not specified in .env file. Please add it as SQLITE_DB_PATH=..."
    )
SQLITE_DB_PATH = Path(SQLITE_DB_PATH)
SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SQLITE_CONNECTION = sqlite3.connect(SQLITE_DB_PATH)


def create_table(name: str) -> None:
    """Create a table to store eval results."""

    cursor = SQLITE_CONNECTION.cursor()
    cursor.execute(
        f"""CREATE TABLE IF NOT EXISTS {name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meta TEXT,
            input TEXT,
            output TEXT
        )"""
    )


def insert_record(table_name: str, meta: dict, input: list[dict], output: dict) -> None:
    """Insert a record into the table."""
    cursor = SQLITE_CONNECTION.cursor()

    data = (json.dumps(meta), json.dumps(input), json.dumps(output))
    cursor.execute(
        f"""INSERT INTO {table_name} (meta, input, output) VALUES (?, ?, ?)""", data
    )
    SQLITE_CONNECTION.commit()
