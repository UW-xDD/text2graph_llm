import sqlite3

if __name__ == "__main__":
    db_path = "/staging/clo36/text2graph/preprocess/triplets.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")

        # Check the journal mode
        mode = connection.execute("PRAGMA journal_mode;").fetchone()
        print("Journal Mode:", mode[0])
