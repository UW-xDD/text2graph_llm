import pandas as pd
import logging
import argparse
import hashlib
import requests
import os
from tqdm import tqdm
from text2graph.database import create_table, insert_record
from text2graph.prompt import SYSTEM_PROMPT, get_user_prompt

logging.basicConfig(level=logging.DEBUG)


def ask_mixtral(messages: list[dict], json: bool = False) -> str:
    """Ask mixtral with a data package.

    Example input: [{"role": "user", "content": "Hello world example in python."}]

    """
    url = "http://localhost:11434/api/chat"

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

    # Non-streaming mode
    response = requests.post(url, json=data)
    response.raise_for_status()
    # return response.json()["message"]["content"]
    return response.json()


def run(run_name: str, batch_size: int, start_index: int) -> None:
    """Run a batch of inference."""

    INPUT_FILE = os.getenv("INPUT_FILE")

    # Calculate input file md5
    with open(INPUT_FILE, "rb") as f:
        df_md5 = hashlib.md5(f.read()).hexdigest()

    # Load batch
    df = pd.read_parquet(INPUT_FILE)
    df = df.iloc[start_index : start_index + batch_size]

    # Create table if not exists
    create_table(run_name)

    for index, row in tqdm(df.iterrows()):
        meta = {
            "df_md5": df_md5,
            "index": index,
            "formation_name": row["formation_name"],
            "paper_id": row["paper_id"],
        }
        logging.info(f"Processing: {meta}")

        user_prompt = get_user_prompt(row["paragraph"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        logging.info(f"Messages: {messages}")

        response = ask_mixtral(messages)

        # Push to database
        insert_record(table_name=run_name, meta=meta, input=messages, output=response)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run llm inference.")
    parser.add_argument("--run_name", type=str, help="Name of the run")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--start_index", type=int, help="Start index")

    settings = vars(parser.parse_args())
    logging.info(f"Settings: {settings}")
    run(**settings)
