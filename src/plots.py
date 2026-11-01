"""Plots shared by every notebook, so the methods can be compared by eye."""
from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib.ticker import NullFormatter, ScalarFormatter
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import POLARITIES

BAR_COLOR = "#4C78A8"
SERIES_COLORS = ["#2E6DA4", "#D9730D", "#6E3FA3", "#1FA187", "#C2185B"]


def plot_f1_by_threshold(thresholds, f1_values, best_threshold, title="F1 by threshold", posthoc_threshold=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, f1_values, color=BAR_COLOR, linewidth=2)
    ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1, label=f"chosen: {best_threshold:.2f}")
    if posthoc_threshold is not None:
        ax.axvline(posthoc_threshold, color=SERIES_COLORS[1], linestyle=":", linewidth=2, label=f"best afterwards: {posthoc_threshold:.2f}")
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


def plot_category_gain(first_report, second_report, first_name, second_name):
    """F1 of two models per category: the long side of each line shows who wins, and by how much."""
    first = first_report.set_index("category")
    second = second_report.set_index("category").loc[first.index]
    labels = [f"{category}  (n={support})" for category, support in first.support.items()]

    fig, ax = plt.subplots(figsize=(9, 0.3 * len(first) + 1.5))
    ax.hlines(labels, first.f1, second.f1, color="#BBBBBB", linewidth=2)
    ax.scatter(first.f1, labels, color=SERIES_COLORS[0], s=35, zorder=3, label=first_name)
    ax.scatter(second.f1, labels, color=SERIES_COLORS[1], s=35, zorder=3, marker="D", label=second_name)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("F1")
    ax.set_title(f"F1 per category: {first_name} and {second_name}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_threshold_policies(holdout, title="Thresholds tuned and scored on different val reviews"):
    """holdout: output of evaluation.compare_threshold_policies."""
    columns = {
        "micro-F1, global": "micro_f1_global",
        "micro-F1, per category": "micro_f1_per_category",
        "macro-F1, global": "macro_f1_global",
        "macro-F1, per category": "macro_f1_per_category",
    }
    fig, ax = plt.subplots(figsize=(8, 3.5))
    boxes = ax.boxplot([holdout[column] for column in columns.values()], vert=False, tick_labels=list(columns),
                       patch_artist=True, widths=0.6, medianprops={"color": "black"})
    for box, color in zip(boxes["boxes"], [SERIES_COLORS[0], SERIES_COLORS[1]] * 2):
        box.set_facecolor(color)
    ax.invert_yaxis()
    ax.set_xlabel("score on the held-out reviews")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def plot_training_curves(histories, title="Training curves", score_name="val score"):
    """histories: {run name: rows returned by training.fit}. A loss that keeps falling while
    the val score stalls is overfitting."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (name, history), color in zip(histories.items(), SERIES_COLORS):
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], color=color, marker="o", linewidth=2, label=name)
        axes[1].plot(epochs, [row["val_score"] for row in history], color=color, marker="o", linewidth=2, label=name)
    axes[0].set_title("Train loss")
    axes[1].set_title(score_name)
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_grouped_bars(table, value_name, title):
    """One group of bars per row of table, one bar colour per column, between 0 and 1."""
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(table)), 4))
    width = 0.8 / len(table.columns)
    positions = np.arange(len(table))
    for index, (column, color) in enumerate(zip(table.columns, SERIES_COLORS)):
        bars = ax.bar(positions + index * width, table[column], width * 0.92, color=color, label=column)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(positions + width * (len(table.columns) - 1) / 2, table.index)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(value_name)
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=len(table.columns), frameon=False)
    plt.tight_layout()
    plt.show()


def plot_polarity_f1(f1_by_model, title="F1 of each polarity"):
    """f1_by_model: {model name: output of evaluation.polarity_f1_by_class}."""
    plot_grouped_bars(pd.DataFrame(f1_by_model).T[POLARITIES], "F1", title)


def plot_mixed_accuracy(tables, title="Polarity accuracy on one-polarity and mixed sentences"):
    """tables: {model name: output of evaluation.accuracy_by_mixed_sentence}."""
    table = pd.DataFrame({name: result.accuracy for name, result in tables.items()}).T
    table.columns = ["mixed" if is_mixed else "one polarity" for is_mixed in table.columns]
    plot_grouped_bars(table, "accuracy", title)


def plot_quality_against_cost(table, cost, quality, title=None):
    """table: one row per model, with columns `name`, cost and quality. Best models sit at the top left."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(table[cost], table[quality], color=BAR_COLOR, s=70, edgecolor="white")
    for row in table.itertuples():
        ax.annotate(row.name, (getattr(row, cost), getattr(row, quality)), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.margins(x=0.2, y=0.15)
    ax.set_xlabel(f"{cost} (log scale)")
    ax.set_ylabel(quality)
    ax.set_title(title or f"{quality} against {cost}")
    plt.tight_layout()
    plt.show()


def plot_polarity_confusion(true_labels, predicted_labels, title="Polarity confusion (share of each true class)"):
    display_matrix = ConfusionMatrixDisplay.from_predictions(
        true_labels, predicted_labels, labels=POLARITIES, normalize="true", cmap="Blues", values_format=".2f",
    )
    display_matrix.ax_.grid(False)
    display_matrix.ax_.set_title(title)
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


def plot_model_comparison(results, metric, title=None, error=None):
    """results: one row per model, with columns `name` and `metric`. error: column of the error bars."""
    ordered = results.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1.5))
    bars = ax.barh(ordered["name"], ordered[metric], color=BAR_COLOR, height=0.7,
                   xerr=None if error is None else ordered[error], error_kw={"ecolor": "black", "capsize": 3})
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric)
    ax.set_title(title or metric)
    plt.tight_layout()
    plt.show()


def show_attention(token_lists, weight_lists, titles):
    """Sentences as HTML, darker where the attention is higher. Browsers join and order
    Arabic letters correctly, matplotlib does not."""
    for tokens, weights, title in zip(token_lists, weight_lists, titles):
        strongest = max(weights)
        words = [
            f'<span style="background: rgba(76, 120, 168, {weight / strongest:.2f}); padding: 2px">{escape(token)}</span>'
            for token, weight in zip(tokens, weights)
        ]
        display(HTML(f'<p><b>{escape(title)}</b></p><div dir="rtl" style="font-size: 18px; line-height: 2">{" ".join(words)}</div>'))
