"""Results table, score files and saved runs."""
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.data import MODELS_DIR, OUTPUT_DIR

RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "results.csv"
SCORES_DIR = RESULTS_DIR / "scores"
RUNS_DIR = MODELS_DIR / "runs"

RESULT_KEY = ["task", "model", "seed", "preprocessing", "params", "threshold_policy"]
RESULT_COLUMNS = RESULT_KEY + ["micro_f1", "macro_f1", "accuracy", "fit_time_s"]


def cached_run(run_name, compute, retrain=False):
    """Run compute() once and keep what it returns. Running the cell again reloads the saved
    output, so a crash or a lost session never costs a finished run. retrain=True runs it again."""
    path = RUNS_DIR / f"{run_name}.joblib"
    if path.exists() and not retrain:
        print(f"{run_name}: loaded from disk")
        return joblib.load(path)

    output = compute()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    unfinished_path = path.with_suffix(".unfinished")
    joblib.dump(output, unfinished_path)
    unfinished_path.replace(path)
    print(f"{run_name}: saved")
    return output


def latency_ms(predict_one, texts):
    start = time.perf_counter()
    for text in texts:
        predict_one(text)
    return 1000 * (time.perf_counter() - start) / len(texts)


def result_row(task, model, preprocessing="none", params="none", threshold_policy="none", fit_time_s=None, seed=42, **scores):
    return {
        "task": task, "model": model, "seed": seed, "preprocessing": preprocessing, "params": params,
        "threshold_policy": threshold_policy, "fit_time_s": fit_time_s, **scores,
    }


def log_results(rows: list[dict], path=RESULTS_PATH) -> pd.DataFrame:
    """Add rows to the results table. Same RESULT_KEY as an old row: the new row replaces it."""
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    if path.exists():
        table = pd.concat([pd.read_csv(path), table], ignore_index=True)
    table = table.drop_duplicates(subset=RESULT_KEY, keep="last").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def best_of_each_model(results, task, metric):
    """Best run of each model, averaged over its seeds, ready for plots.plot_model_comparison.
    Leaves out rules that read the gold labels, and per-category thresholds scored on the
    sentences that tuned them."""
    candidates = results[(results.task == task) & (results.threshold_policy != "per_category")]
    candidates = candidates[~candidates.model.str.contains("gold")]
    means = candidates.groupby(["model", "preprocessing", "params"], as_index=False)[metric].mean()
    return means.sort_values(metric).groupby("model").tail(1).rename(columns={"model": "name"})


def save_scores(run_name, split_name, scores, row_ids, classes, meta=None):
    """Save a model's outputs, so Part F can compare models without retraining.

    row_ids: one array per id, e.g. {"sentence_id": ids} or {"sentence_id": ids, "category": categories}
    meta: anything that identifies the run (config, seed, thresholds)
    """
    arrays = {f"id_{name}": np.asarray(values, dtype=str if name == "category" else None) for name, values in row_ids.items()}
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
