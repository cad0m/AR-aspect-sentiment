"""Plots shared by every notebook, so the methods can be compared by eye."""
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import POLARITIES

BAR_COLOR = "#4C78A8"


def plot_f1_by_threshold(thresholds, f1_values, best_threshold, title="F1 by threshold"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1_values, color=BAR_COLOR, linewidth=2)
    ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1, label=f"chosen: {best_threshold:.2f}")
    ax.set_xlabel("threshold")
    ax.set_ylabel("micro-F1")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_category_f1(report, title="F1 per category"):
    """report: output of evaluation.category_report."""
    labels = [f"{row.category}  (n={row.support})" for row in report.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 0.3 * len(report) + 1.5))
    ax.barh(labels, report.f1, color=BAR_COLOR, height=0.7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_polarity_confusion(true_labels, predicted_labels, title="Polarity confusion (share of each true class)"):
    display = ConfusionMatrixDisplay.from_predictions(
        true_labels, predicted_labels, labels=POLARITIES, normalize="true", cmap="Blues", values_format=".2f",
    )
    display.ax_.grid(False)
    display.ax_.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_pipeline_outcomes(outcomes, title="What happens to each (sentence, category)"):
    """outcomes: output of evaluation.pipeline_outcomes."""
    counts = outcomes.status.value_counts()
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.barh(counts.index, counts.values, color=BAR_COLOR, height=0.7)
    ax.bar_label(bars, padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("pairs")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(results, metric, title=None):
    """results: one row per model, with columns `name` and `metric`."""
    ordered = results.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1.5))
    bars = ax.barh(ordered["name"], ordered[metric], color=BAR_COLOR, height=0.7)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(title or metric)
    plt.tight_layout()
    plt.show()
