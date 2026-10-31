"""Results table and score files. Models are saved in each notebook:
joblib for sklearn, torch.save for torch."""
import json

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_PATH = RESULTS_DIR / "results.csv"
SCORES_DIR = RESULTS_DIR / "scores"

RESULT_KEY = ["task", "model", "seed", "preprocessing", "params", "threshold_policy"]
RESULT_COLUMNS = RESULT_KEY + ["micro_f1", "macro_f1", "accuracy", "fit_time_s"]


def log_results(rows: list[dict], path=RESULTS_PATH) -> pd.DataFrame:
    """Add rows to the results table. Same RESULT_KEY as an old row: the new row replaces it."""
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    table = table.drop_duplicates(subset=RESULT_KEY, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def save_scores(run_name, split_name, scores, row_ids, classes, meta=None):
    """Save a model's outputs, so Part F can compare models without retraining.

    row_ids: one array per id, e.g. {"sentence_id": ids} or {"sentence_id": ids, "category": categories}
    meta: anything that identifies the run (config, seed, thresholds)
    """
    arrays = {f"id_{name}": np.asarray(values) for name, values in row_ids.items()}
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        SCORES_DIR / f"{run_name}__{split_name}.npz",
        scores=np.asarray(scores),
        classes=np.array(list(classes), dtype=str),
        meta=np.array(json.dumps(meta or {})),
        **arrays,
    )


def load_scores(run_name, split_name, expected_sentence_ids=None):
    with np.load(SCORES_DIR / f"{run_name}__{split_name}.npz", allow_pickle=False) as saved:
        result = {
            "scores": saved["scores"],
            "classes": [str(name) for name in saved["classes"]],
            "row_ids": {name[3:]: saved[name] for name in saved.files if name.startswith("id_")},
            "meta": json.loads(str(saved["meta"])),
        }
    if expected_sentence_ids is not None:
        if not np.array_equal(result["row_ids"]["sentence_id"], np.asarray(expected_sentence_ids)):
            raise ValueError("Saved rows do not match the expected sentence ids")
    return result
