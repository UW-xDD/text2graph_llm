import sqlite3
from typing import Any


DB_ROW_TUPLES_COLUMN_LOOKUP = {
    "index": 0,
    "table_pk": 1,
    "int_id": 2,
    "hashed_text": 3,
    "doc_id": 4,
    "triplet_json": 5,
}


def load_all_rows_from_sqlite_db(
    db_path: str,
) -> list[tuple[Any, Any, Any, Any, Any, Any]]:
    """
    Load all rows from a sqlite db.
    :param db_path: Path to the sqlite db having a table named 'triplets' with six columns.
    :return: List of tuples where each tuple is a row from the sqlite db.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triplets")
    rows = cursor.fetchall()
    conn.close()
    return rows
