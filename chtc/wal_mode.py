import sqlite3

if __name__ == "__main__":
    db_path = "/staging/clo36/text2graph/preprocess/triplets.db"
    with sqlite3.connect(db_path) as connection:
        with connection.cursor() as cur:
            cur.execute("PRAGMA journal_mode=WAL;")
            connection.commit()
