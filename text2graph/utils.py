import sqlite3
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np


def get_output_info(output: str, route: list[str]) -> str:
    """Get the information from the output."""
    response = json.loads(output)
    for r in route:
        response = response[r]
    return response


def get_eta(eval_db: Path, test_set: Path, run_name: str, n_workers: int) -> str:
    """Get the ETA from the evaluation database."""

    df = pd.read_parquet(test_set)
    n = len(df)

    conn = sqlite3.connect(eval_db)
    cursor = conn.cursor()

    completed = cursor.execute(f"SELECT COUNT(id) FROM {run_name}").fetchall()[0][0]

    samples = cursor.execute(
        f"SELECT output FROM {run_name} ORDER BY RANDOM() LIMIT 100"
    ).fetchall()
    eval_durations = []
    for r in samples:
        output = r[0]
        eval_durations.append(get_output_info(output, route=["eval_duration"]))

    hours_left = (n - completed) * (np.mean(eval_durations) / 1e9 / 3600 / n_workers)
    now = datetime.datetime.now()
    expected_finish_time = now + datetime.timedelta(hours=hours_left)
    return f"{now.strftime('%H:%M:%S')}: {completed}/{n} ({completed/n*100:.2f}%) Expected finish: {expected_finish_time.strftime('%Y-%m-%d %H:%M:%S')}"
