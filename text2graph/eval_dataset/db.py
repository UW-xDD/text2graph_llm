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
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM triplets")
    rows = cursor.fetchall()
    conn.close()
    return rows
