"""Shared evaluation. Every model gives one of these two things:
  category model -> scores (n_sentences, n_classes)
  polarity model -> one label per (sentence, category), a name from data.POLARITIES
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
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


def compare_threshold_policies(scores, true_labels, review_ids, n_splits=200, seed=42):
    """Tune thresholds on half of the val reviews and score them on the other half, n_splits times.
    Tuning and scoring on the same sentences would flatter the per-category thresholds."""
    random_generator = np.random.default_rng(seed)
    review_ids = np.asarray(review_ids)
    unique_reviews = np.unique(review_ids)

    rows = []
    for _ in range(n_splits):
        tuning_reviews = random_generator.choice(unique_reviews, len(unique_reviews) // 2, replace=False)
        tuning_part = np.isin(review_ids, tuning_reviews)
        scoring_part = ~tuning_part

        global_threshold, _ = best_global_threshold(scores[tuning_part], true_labels[tuning_part])
        category_thresholds = best_category_thresholds(scores[tuning_part], true_labels[tuning_part], global_threshold)

        with_global = category_scores(true_labels[scoring_part], predict_labels(scores[scoring_part], global_threshold))
        with_per_category = category_scores(true_labels[scoring_part], predict_labels(scores[scoring_part], category_thresholds))
        rows.append({
            "micro_f1_global": with_global["micro_f1"],
            "micro_f1_per_category": with_per_category["micro_f1"],
            "macro_f1_global": with_global["macro_f1"],
            "macro_f1_per_category": with_per_category["macro_f1"],
        })
    return pd.DataFrame(rows)


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


def accuracy_by_mixed_sentence(units, predicted_polarities):
    """Polarity accuracy on sentences with both a positive and a negative aspect, and on the others."""
    keep = ~units.ambiguous.to_numpy(dtype=bool)
    clear_units = units[keep]
    predicted = np.asarray(predicted_polarities, dtype=object)[keep]
    check = clear_units.assign(
        correct=predicted == clear_units.polarity.to_numpy(dtype=object),
        mixed_sentence=lambda table: table.groupby("sentence_id").polarity.transform(
            lambda polarities: {"positive", "negative"} <= set(polarities)).astype(bool),
    )
    return check.groupby("mixed_sentence").correct.agg(accuracy="mean", aspects="size").round(3)


def polarity_f1_by_class(units, predicted_polarities):
    """F1 of each polarity, for the pairs with one clear polarity."""
    keep = ~units.ambiguous.to_numpy(dtype=bool)
    true_polarities = units.polarity.to_numpy(dtype=object)[keep]
    predicted = np.asarray(predicted_polarities, dtype=object)[keep]
    values = f1_score(true_polarities, predicted, labels=POLARITIES, average=None, zero_division=0)
    return dict(zip(POLARITIES, values.tolist()))


def bootstrap_interval(metric, groups, n_resamples=1000, seed=42, confidence=0.95):
    """Confidence interval of metric(chosen_rows). Whole groups (reviews) are drawn with replacement,
    because sentences of the same review are not independent."""
    random_generator = np.random.default_rng(seed)
    groups = np.asarray(groups)
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    rows_by_group = [np.flatnonzero(group_index == index) for index in range(len(unique_groups))]

    values = []
    for _ in range(n_resamples):
        drawn = random_generator.integers(len(unique_groups), size=len(unique_groups))
        values.append(metric(np.concatenate([rows_by_group[index] for index in drawn])))

    tail = (1 - confidence) / 2 * 100
    return tuple(float(value) for value in np.percentile(values, [tail, 100 - tail]))


def mcnemar_test(first_correct, second_correct):
    """Exact McNemar test on two models scored on the same decisions."""
    first_correct = np.asarray(first_correct, dtype=bool)
    second_correct = np.asarray(second_correct, dtype=bool)
    only_first = int((first_correct & ~second_correct).sum())
    only_second = int((~first_correct & second_correct).sum())
    n_differences = only_first + only_second
    p_value = binomtest(only_first, n_differences, 0.5).pvalue if n_differences else 1.0
    return {"only_first_correct": only_first, "only_second_correct": only_second, "p_value": float(p_value)}


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
