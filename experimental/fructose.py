from dataclasses import asdict, dataclass
from typing import Callable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from evals.graph_dataset.formation_extraction import eval as get_eval


@dataclass
class Triplets:
    subject: str
    object: str
    predicate: str
    confidence: float


def get_stratigraphic_entities(graph: list[Triplets]) -> list[str]:
    """Given a list of triplets, returns all the stratigraphic entities."""

    return list(set([triplet.subject for triplet in graph if triplet.subject]))


def run_eval(
    model: str,
    extract_fn: Callable,
    post_process_fn: Callable,
    n: int,
    save: bool = False,
) -> pd.DataFrame:
    """Run evaluation."""
    eval = get_eval(
        subset=n, data="/workspaces/text2graph_llm/tmp/formation_sample.parquet.gzip"
    )
    prompt_version = extract_fn.__name__

    raw_outputs = []
    eval_outputs = []
    for text in tqdm(eval.x.tolist()):
        graph = extract_fn(text)
        stratigraphic_entities = get_stratigraphic_entities(graph)
        print(stratigraphic_entities)

        eval_outputs.append(stratigraphic_entities)
        raw_outputs.append([asdict(triplet) for triplet in graph])

    eval_outputs_flat = [", ".join(x) for x in eval_outputs]
    eval_summary = eval.from_predictions(eval_outputs_flat)

    # Item level metrics
    df = pd.DataFrame(
        [
            get_metrics([y_true], y_pred)
            for y_true, y_pred in zip(eval.y_true, eval_outputs)
        ]
    )
    df["text"] = eval.x.tolist()
    df["label"] = eval.y_true.tolist()
    df["raw_output"] = raw_outputs
    df["processed_output"] = eval_outputs
    df["model"] = model
    df["prompt_version"] = prompt_version
    df["post_process_fn"] = post_process_fn.__name__
    df["macro_f1"] = eval_summary.metrics["macro_f1"]
    df["macro_exact_match"] = eval_summary.metrics["exact_match"]

    # Prevent saving error
    str_cols = ["text", "label", "raw_output", "processed_output"]
    cat_cols = ["model", "prompt_version", "post_process_fn"]
    df = df.astype({col: str for col in str_cols})
    df = df.astype({col: "category" for col in cat_cols})

    if save:
        df.to_parquet(f"evals/{model}_{prompt_version}_{n}.parquet.gzip")
    return df


# Eval helpers


def fuzzy_match(a: str, b: str) -> bool:
    """Return True if a and b are similar."""

    intersect_score = a.lower() in b.lower() or b.lower() in a.lower()
    # Remove formation
    a = a.replace("Formation", "").strip()
    b = b.replace("Formation", "").strip()

    loose_intersect_score = a.lower() in b.lower() or b.lower() in a.lower()
    return intersect_score or loose_intersect_score


def get_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Calculate the metrics for the output."""

    print(f"{y_true=}", f"{y_pred=}")
    tp = len(set(y_true) & set(y_pred))
    fp = len(set(y_pred) - set(y_true))
    fn = len(set(y_true) - set(y_pred))
    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)

    fuzzy_tp = sum(fuzzy_match(a, b) for a in y_true for b in y_pred)
    return {"precision": precision, "recall": recall, "f1": f1, "fuzzy_tp": fuzzy_tp}


def get_leaderboard(
    eval_dir: Path = Path("/workspaces/text2graph_llm/evals"),
) -> pd.DataFrame:
    """Get the leaderboard."""

    df = pd.concat([pd.read_parquet(x) for x in eval_dir.glob("*.parquet.gzip")])
    return (
        df.groupby(["model", "prompt_version"])
        .agg(
            {
                "macro_f1": "mean",
                "macro_exact_match": "mean",
                "f1": "mean",
                "precision": "mean",
                "recall": "mean",
                "fuzzy_tp": "mean",
            }
        )
        .reset_index()
        .sort_values("macro_f1", ascending=False)
    )
