"""Figures for the report.

Chart rules: the title states the finding, not the metric; the reader's units
are used; one colour carries meaning and the rest recede.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from src import config, metrics

PALETTE = config.PALETTE
MODEL_COLOURS = {
    "recency_baseline": PALETTE["muted"],
    "logistic_regression": "#7a9cc6",
    "random_forest": PALETTE["accent"],
    "xgboost": PALETTE["primary"],
}


def _style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150,
        "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.labelsize": 10, "axes.edgecolor": "#4a4a4a", "font.size": 10,
    })


def _despine(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(config.PROJECT_ROOT))


# --------------------------------------------------------------------------- #
def plot_temporal_design(seasonality: pd.DataFrame) -> str:
    """The base rate moves with the season — the reason for a temporal split."""
    _style()
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    colours = [PALETTE["accent"] if s == "holdout" else PALETTE["primary"]
               for s in seasonality["split"]]
    ax.bar(seasonality["cutoff"], seasonality["base_rate"], color=colours)
    for x, y in zip(seasonality["cutoff"], seasonality["base_rate"]):
        ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", fontsize=9)

    ax.axhline(seasonality["base_rate"].mean(), color=PALETTE["bad"], ls="--", lw=1.2)
    ax.text(0.02, seasonality["base_rate"].mean() + 1.2,
            f"mean {seasonality['base_rate'].mean():.1f}%",
            color=PALETTE["bad"], fontsize=9, transform=ax.get_yaxis_transform())
    ax.set_title("The target moves with the season: 24% of customers return in August, 35% in November")
    ax.set_ylabel("Customers returning\nwithin 30 days (%)")
    ax.set_xlabel("Snapshot cutoff  (blue = development, orange = holdout)")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0, seasonality["base_rate"].max() * 1.25)
    _despine(ax)
    return _save(fig, "01_temporal_design")


def plot_cv_results(cv: pd.DataFrame) -> str:
    """Per-fold PR-AUC: the spread matters as much as the mean."""
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    for name, group in cv.groupby("model"):
        ax1.plot(group["valid_cutoff"], group["pr_auc"], marker="o", ms=5, lw=2,
                 label=name, color=MODEL_COLOURS.get(name))
    ax1.set_title("Every model beats the rule; none clearly beats the others")
    ax1.set_ylabel("PR-AUC on the validation fold")
    ax1.set_xlabel("Validation cutoff")
    ax1.tick_params(axis="x", rotation=30)
    ax1.legend(frameon=False, fontsize=8.5)
    _despine(ax1)

    summary = cv.groupby("model")[["accuracy", "majority_accuracy"]].mean().sort_values("accuracy")
    y = np.arange(len(summary))
    ax2.barh(y, summary["accuracy"], color=[MODEL_COLOURS.get(m) for m in summary.index])
    ax2.axvline(summary["majority_accuracy"].iloc[0], color=PALETTE["bad"], ls="--", lw=1.6)
    ax2.text(summary["majority_accuracy"].iloc[0] + 0.004, 0.15,
             "predict\n'nobody returns'", color=PALETTE["bad"], fontsize=9)
    ax2.set_yticks(y, summary.index)
    ax2.set_xlim(0.65, 0.81)
    ax2.set_title("...and accuracy cannot tell you that")
    ax2.set_xlabel("Accuracy at threshold 0.5")
    _despine(ax2)
    return _save(fig, "02_cv_results")


def plot_pr_and_roc(y_true, scores: dict) -> str:
    """PR and ROC side by side, to show why PR is the honest one here."""
    from sklearn.metrics import precision_recall_curve, roc_curve
    _style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    base = float(np.mean(y_true))

    for name, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_true, score)
        ax1.plot(recall, precision, lw=2, label=f"{name} ({metrics.evaluate(y_true, score)['pr_auc']:.3f})",
                 color=MODEL_COLOURS.get(name))
        fpr, tpr, _ = roc_curve(y_true, score)
        ax2.plot(fpr, tpr, lw=2, color=MODEL_COLOURS.get(name), label=name)

    ax1.axhline(base, color=PALETTE["bad"], ls=":", lw=1.4)
    ax1.text(0.55, base + 0.02, f"random targeting = {base:.0%}", color=PALETTE["bad"], fontsize=9)
    ax1.set_title("Precision–recall (PR-AUC in the legend)")
    ax1.set_xlabel("Recall"); ax1.set_ylabel("Precision")
    ax1.legend(frameon=False, fontsize=8.5, loc="upper right")
    _despine(ax1)

    ax2.plot([0, 1], [0, 1], color=PALETTE["muted"], ls=":", lw=1.2)
    ax2.set_title("ROC — flattering, because the easy negatives dominate")
    ax2.set_xlabel("False positive rate"); ax2.set_ylabel("True positive rate")
    ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
    _despine(ax2)
    fig.suptitle("On an imbalanced ranking task the two curves tell different stories",
                 fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "03_pr_roc_curves")


def plot_gains(y_true, y_score) -> str:
    """The decile chart a marketing team reads, plus cumulative capture."""
    _style()
    gains = metrics.gains_table(y_true, y_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    base = float(np.mean(y_true))

    colours = [PALETTE["primary"] if d <= 2 else PALETTE["muted"] for d in gains.index]
    ax1.bar(gains.index, 100 * gains["precision"], color=colours)
    ax1.axhline(100 * base, color=PALETTE["bad"], ls="--", lw=1.4)
    ax1.text(6.2, 100 * base + 1.5, f"random = {100*base:.0f}%", color=PALETTE["bad"], fontsize=9)
    ax1.set_title("The top decile returns at 3x the base rate")
    ax1.set_xlabel("Score decile (1 = highest)"); ax1.set_ylabel("Returned within 30 days (%)")
    _despine(ax1)

    x = np.arange(1, 11) * 10
    ax2.plot(x, 100 * gains["cumulative_recall"], marker="o", ms=5, lw=2.2,
             color=PALETTE["primary"], label="model")
    ax2.plot([0, 100], [0, 100], color=PALETTE["muted"], ls=":", lw=1.2, label="random")
    capacity = int(config.CAMPAIGN_CAPACITY * 100)
    captured = 100 * gains["cumulative_recall"].iloc[1]
    ax2.axvline(capacity, color=PALETTE["accent"], ls="--", lw=1.4)
    ax2.annotate(f"campaign budget: top {capacity}%\ncaptures {captured:.0f}% of returners",
                 xy=(capacity, captured), xytext=(capacity + 12, captured - 22),
                 fontsize=9, color=PALETTE["accent"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["accent"]))
    ax2.set_title("Cumulative capture against the contact budget")
    ax2.set_xlabel("Share of customers contacted (%)")
    ax2.set_ylabel("Share of returners reached (%)")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    _despine(ax2)
    return _save(fig, "04_gains")


def plot_calibration(y_true, scores: dict) -> str:
    """Reliability curves — does a 0.3 score mean a 30% chance?"""
    from sklearn.calibration import calibration_curve
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], ls=":", lw=1.4, label="perfect")

    for name, score in scores.items():
        observed, predicted = calibration_curve(y_true, score, n_bins=10, strategy="quantile")
        brier = metrics.evaluate(y_true, score)["brier"]
        ax.plot(predicted, observed, marker="o", ms=5, lw=2,
                color=MODEL_COLOURS.get(name), label=f"{name} (Brier {brier:.3f})")

    ax.set_title("Well calibrated at the top, under-confident at the bottom")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _despine(ax)
    return _save(fig, "05_calibration")


def plot_importance(importance: pd.DataFrame, top_n: int = 15) -> str:
    """Permutation importance on the holdout, scored by PR-AUC."""
    _style()
    data = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    colours = [PALETTE["accent"] if f == "recency_vs_gap" else PALETTE["primary"]
               for f in data["feature"]]
    ax.barh(data["feature"], data["pr_auc_drop"], xerr=data["std"],
            color=colours, error_kw={"ecolor": "#666", "lw": 1})
    ax.set_title("The engineered cadence feature matters most")
    ax.set_xlabel("Drop in holdout PR-AUC when the feature is shuffled")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.3f}"))
    _despine(ax)
    return _save(fig, "06_permutation_importance")


def plot_capacity(sweep: pd.DataFrame) -> str:
    """Efficiency versus total value: the decision the model cannot make."""
    _style()
    fig, ax1 = plt.subplots(figsize=(8.5, 4.6))
    x = 100 * sweep["k"]

    ax1.plot(x, sweep["net"], marker="o", ms=6, lw=2.4, color=PALETTE["primary"],
             label="total net value")
    ax1.set_xlabel("Share of customers contacted (%)")
    ax1.set_ylabel("Total net value (£)", color=PALETTE["primary"])
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v/1000:.0f}K"))
    _despine(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, sweep["net_per_contact"], marker="s", ms=6, lw=2.4, ls="--",
             color=PALETTE["accent"], label="value per contact")
    ax2.set_ylabel("Net value per contact (£)", color=PALETTE["accent"])
    ax2.grid(False)
    for side in ("top",):
        ax2.spines[side].set_visible(False)

    ax1.set_title("Contact more people to earn more; contact fewer to earn more each")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=9, loc="center right")
    return _save(fig, "07_capacity_tradeoff")


def build_all(seasonality, cv, y_true, scores: dict, importance, sweep) -> list:
    best = max(scores, key=lambda n: metrics.evaluate(y_true, scores[n])["pr_auc"])
    return [
        plot_temporal_design(seasonality),
        plot_cv_results(cv),
        plot_pr_and_roc(y_true, scores),
        plot_gains(y_true, scores[best]),
        plot_calibration(y_true, scores),
        plot_importance(importance),
        plot_capacity(sweep),
    ]
