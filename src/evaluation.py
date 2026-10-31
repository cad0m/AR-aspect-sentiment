"""Shared evaluation. Every model gives one of these two things:
  category model -> scores (n_sentences, n_classes)
  polarity model -> one label per (sentence, category), a name from data.POLARITIES
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data import POLARITIES

THRESHOLD_GRID = np.arange(0.02, 1.0, 0.02)

CORRECT = "correct"
WRONG_POLARITY = "wrong polarity"
MISSED = "missed category"
FALSE_ALARM = "false alarm"


def _ratio(top, bottom):
    top = np.asarray(top, dtype=float)
    bottom = np.asarray(bottom, dtype=float)
    return np.divide(top, bottom, out=np.zeros_like(bottom), where=bottom > 0)


def f1_from_counts(n_correct, n_predicted, n_gold):
    return _ratio(2 * np.asarray(n_correct), np.asarray(n_predicted) + np.asarray(n_gold))


def category_scores(true_labels, predicted_labels):
    """Micro and macro F1. Macro averages the categories present in true_labels."""
    n_correct = (true_labels * predicted_labels).sum(axis=0)
    n_predicted = predicted_labels.sum(axis=0)
    n_gold = true_labels.sum(axis=0)
    f1_per_category = f1_from_counts(n_correct, n_predicted, n_gold)
    return {
        "micro_f1": float(f1_from_counts(n_correct.sum(), n_predicted.sum(), n_gold.sum())),
        "macro_f1": float(f1_per_category[n_gold > 0].mean()),
        "precision": float(_ratio(n_correct.sum(), n_predicted.sum())),
        "recall": float(_ratio(n_correct.sum(), n_gold.sum())),
    }


def category_report(true_labels, predicted_labels, classes):
    """One row per category, most frequent first."""
    n_correct = (true_labels * predicted_labels).sum(axis=0)
    n_predicted = predicted_labels.sum(axis=0)
    n_gold = true_labels.sum(axis=0)
    report = pd.DataFrame({
        "category": classes,
        "support": n_gold,
        "predicted": n_predicted,
        "correct": n_correct,
        "precision": _ratio(n_correct, n_predicted),
        "recall": _ratio(n_correct, n_gold),
        "f1": f1_from_counts(n_correct, n_predicted, n_gold),
    })
    return report.sort_values("support", ascending=False).reset_index(drop=True)


def _middle_of_best(grid, f1_values):
    """Several thresholds can tie on F1: take the middle one, not the lowest."""
    tied = [threshold for threshold, f1 in zip(grid, f1_values) if f1 == max(f1_values)]
    return float(tied[len(tied) // 2])


def best_global_threshold(scores, true_labels, grid=THRESHOLD_GRID):
    """One threshold for all categories. Returns (threshold, micro_f1 at each grid value)."""
    curve = [category_scores(true_labels, (scores >= threshold).astype(int))["micro_f1"] for threshold in grid]
    return _middle_of_best(grid, curve), np.array(curve)


def best_category_thresholds(scores, true_labels, global_threshold, grid=THRESHOLD_GRID, min_support=5):
    """One threshold per category. A category with too few positives, or an F1 of 0
    at every threshold, keeps the global threshold."""
    thresholds = np.full(true_labels.shape[1], global_threshold, dtype=float)
    for category in range(true_labels.shape[1]):
        gold = true_labels[:, category]
        if gold.sum() < min_support:
            continue
        f1_values = []
        for threshold in grid:
            predicted = scores[:, category] >= threshold
            n_correct = (predicted & (gold == 1)).sum()
            f1_values.append(float(f1_from_counts(n_correct, predicted.sum(), gold.sum())))
        if max(f1_values) > 0:
            thresholds[category] = _middle_of_best(grid, f1_values)
    return thresholds


def predict_labels(scores, thresholds):
    """thresholds: one number, or one value per category."""
    return (scores >= np.asarray(thresholds, dtype=float)).astype(int)


def pairs_from_matrix(sentence_table, predicted_labels, classes):
    """0/1 matrix -> one row per predicted (sentence, category). This is the polarity model's input."""
    rows = []
    for row, sentence in enumerate(sentence_table.itertuples()):
        for column in np.flatnonzero(predicted_labels[row]):
            rows.append({"sentence_id": sentence.sentence_id, "text": sentence.text, "category": classes[column]})
    return pd.DataFrame(rows, columns=["sentence_id", "text", "category"])


def polarity_scores(units, predicted_polarities):
    """Polarity quality when the category is given. Ambiguous pairs have no single
    right answer, so they are left out."""
    keep = ~units.ambiguous.to_numpy(dtype=bool)
    true_polarities = units.polarity.to_numpy(dtype=object)[keep]
    predicted = np.asarray(predicted_polarities, dtype=object)[keep]
    return {
        "accuracy": float(accuracy_score(true_polarities, predicted)),
        "macro_f1": float(f1_score(true_polarities, predicted, labels=POLARITIES, average="macro", zero_division=0)),
    }


def _polarities_by_pair(pair_table):
    return pair_table.groupby(["sentence_id", "category"]).polarity.agg(set).to_dict()


def pipeline_outcomes(gold_pairs, predicted_pairs):
    """One row per (sentence, category), gold or predicted, with what happened to it.
    An ambiguous gold pair accepts either polarity."""
    gold = _polarities_by_pair(gold_pairs)
    predicted = _polarities_by_pair(predicted_pairs)

    rows = []
    for pair in sorted(gold.keys() | predicted.keys()):
        if pair not in predicted:
            status = MISSED
        elif pair not in gold:
            status = FALSE_ALARM
        elif predicted[pair] & gold[pair]:
            status = CORRECT
        else:
            status = WRONG_POLARITY
        rows.append({
            "sentence_id": pair[0],
            "category": pair[1],
            "gold": "/".join(sorted(gold.get(pair, []))),
            "predicted": "/".join(sorted(predicted.get(pair, []))),
            "status": status,
        })
    return pd.DataFrame(rows)


def pipeline_scores(outcomes):
    """category_f1 ignores the polarity; pipeline_f1 needs category and polarity both right."""
    counts = outcomes.status.value_counts()
    correct = counts.get(CORRECT, 0)
    wrong_polarity = counts.get(WRONG_POLARITY, 0)
    n_gold = correct + wrong_polarity + counts.get(MISSED, 0)
    n_predicted = correct + wrong_polarity + counts.get(FALSE_ALARM, 0)
    return {
        "category_f1": float(f1_from_counts(correct + wrong_polarity, n_predicted, n_gold)),
        "pipeline_f1": float(f1_from_counts(correct, n_predicted, n_gold)),
    }
