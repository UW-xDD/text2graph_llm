import pandas as pd
import logging
import argparse
import requests
import sqlite3
import multiprocessing
import tenacity
from text2graph.database import create_table, insert_record
from text2graph.prompt import SYSTEM_PROMPT, get_user_prompt

logging.basicConfig(level=logging.INFO)
OLLAMA_PORTS = [12000, 12001, 12002, 12003, 12004, 12005, 12007]
WORKERS = len(OLLAMA_PORTS)
MIXTRAL_TIMEOUT = 30
# OLLAMA_PORTS = [12005, 12007]
CONN = sqlite3.connect("data/eval.db")


def get_ids_from_db(run_name: str) -> list[str]:
    """Get all the ids from the database."""
    cursor = CONN.cursor()
    response = cursor.execute(f"SELECT id FROM {run_name}").fetchall()
    return [r[0] for r in response]


def split_dataframe(df: pd.DataFrame, splits: int) -> list[pd.DataFrame]:
    n = len(df)
    return [df[i * n // splits : (i + 1) * n // splits] for i in range(splits)]


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=3, max=60),
    stop=tenacity.stop_after_attempt(6),
)
def ask_mixtral(
    messages: list[dict], json: bool = False, port: int = 11434, timeout: int = 120
) -> str:
    """Ask mixtral with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]

    """
    url = f"http://localhost:{port}/api/chat"

    data = {
        "model": "mixtral",
        "messages": messages,
        "stream": False,  # set to True to get a stream of responses token-by-token
        "options": {
            "temperature": 0.0,
        },
    }

    if json:
        data["format"] = "json"

    response = requests.post(url, json=data, timeout=timeout)
    response.raise_for_status()
    return response.json()


def run_batch(
    run_name: str, batch_df: pd.DataFrame, port: int, id_row: str = "original_index"
) -> None:
    """Run a batch of inference."""

    for _, row in batch_df.iterrows():

        logging.info(f"Processing: {row[id_row]}")
        user_prompt = get_user_prompt(row["paragraph"])
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logging.debug(f"Messages: {messages}")
        try:
            response = ask_mixtral(messages, port=port, timeout=MIXTRAL_TIMEOUT)
        except Exception as e:
            logging.error(f"Error: {e}")
            continue

        # Push to database
        insert_record(
            table_name=run_name,
            id=row[id_row],
            input=messages,
            output=response,
        )


def run_olvi(run_name: str, input_file: str) -> None:
    """Run a batch of inference (for local split on OLVI)."""

    create_table(run_name)

    # Input file
    df = pd.read_parquet(input_file)
    df["original_index"] = df.index

    # filter out in DB completed records
    completed = get_ids_from_db(run_name)
    df = df[~df["original_index"].isin(completed)].reset_index(drop=True)

    batch_dfs = split_dataframe(df, WORKERS)
    logging.info(f"Batch sizes: {[len(chunk) for chunk in batch_dfs]}")

    # Run the inference
    with multiprocessing.Pool(WORKERS) as pool:
        pool.starmap(
            run_batch,
            [
                (run_name, chunk, OLLAMA_PORTS[i], "original_index")
                for i, chunk in enumerate(batch_dfs)
            ],
        )


def run(run_name:str ,input_file: str) -> None:
    """Run the inference on single CHTC node."""

    create_table(run_name)

    # Input file
    df = pd.read_parquet(input_file)
    df["original_index"] = df.index

    # filter out in DB completed records
    completed = get_ids_from_db(run_name)
    df = df[~df["original_index"].isin(completed)].reset_index(drop=True)

    run_batch(run_name=run_name, port=11434, id_row="original_index")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run llm inference.")
    parser.add_argument("--run_name", type=str, help="Name of the run")
    parser.add_argument("--input_file", type=str, help="Input file")

    settings = vars(parser.parse_args())
    logging.info(f"Settings: {settings}")
    run(**settings)
