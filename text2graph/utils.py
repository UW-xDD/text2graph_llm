import datetime
import functools
import json
import sqlite3
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


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


def deprecated(replacement: str | None = None) -> Callable:
    """Decorator to mark functions as deprecated and suggest a replacement function.

    Args:
        replacement (str, optional): The name of the function to use instead.
    """

    def wrapper(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            message = f"Call to deprecated function {func.__name__}."
            if replacement:
                message += f" Use {replacement} instead."
            warnings.simplefilter("always", DeprecationWarning)  # turn off filter
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter("default", DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return new_func

    return wrapper
