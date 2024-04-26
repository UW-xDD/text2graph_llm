import sqlite3

from preprocess_extraction_direct import DB_PATH, enable_wal_mode

if __name__ == "__main__":
    with sqlite3.connect(DB_PATH) as connection:
        enable_wal_mode(connection)
