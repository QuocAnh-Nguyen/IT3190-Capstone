"""Multi-class evaluation framework for single-label genre classification.

Provides the evaluation functions needed for the improved pipeline (Step 2 of
improve_plan).  These replace the multi-label equivalents when operating in
single-label (multi-class) mode.

Public API
----------
compute_metrics_multiclass(y_true, y_pred, label_names=None) -> dict
    Compute standard multi-class scalar metrics and a per-class breakdown.

cross_validate_multiclass(model_fn, X, y, n_folds, label_names) -> dict
    K-Fold cross-validation returning mean +/- std of each metric.

plot_confusion_matrix(y_true, y_pred, label_names, output_path=None)
    Generate and save a confusion matrix heatmap.

plot_per_class_f1(per_class_df, output_path=None)
    Horizontal bar chart of per-class F1, sorted descending.

save_metrics_multiclass(metrics_dict, model_name, output_dir)
    Persist the metric dict to JSON and per-class DataFrame to CSV.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

from src2.config import FIGURES_DIR, REPORTS_DIR, RANDOM_SEED, N_CV_FOLDS
from src2.utils.io_utils import save_csv, save_json

logger = logging.getLogger("music_genre")

Array = np.ndarray
MetricsDict = dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Core metrics (multi-class)
# ---------------------------------------------------------------------------


def compute_metrics_multiclass(
    y_true: Array,
    y_pred: Array,
    label_names: list[str] | None = None,
) -> MetricsDict:
    """Compute comprehensive multi-class classification metrics.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        Ground-truth labels (string or int).
    y_pred : np.ndarray of shape (n_samples,)
        Predicted labels (same type as y_true).
    label_names : list[str] or None
        Ordered class names.  When None, inferred from unique values in y_true.

    Returns
    -------
    dict with keys:
        ``"summary"`` — dict of scalar metrics (accuracy, macro_f1, micro_f1,
            weighted_f1).
        ``"per_class"`` — DataFrame indexed by label with precision/recall/f1/support.
        ``"per_label_f1"`` — dict mapping label -> F1 score.
        ``"confusion_matrix"`` — np.ndarray confusion matrix.
        ``"classification_report"`` — str classification report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if label_names is None:
        label_names = sorted(set(np.unique(y_true)) | set(np.unique(y_pred)))
        label_names = [str(ln) for ln in label_names]

    # Ensure string labels for sklearn
    y_true_str = np.array([str(v) for v in y_true])
    y_pred_str = np.array([str(v) for v in y_pred])

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------
    accuracy = float(accuracy_score(y_true_str, y_pred_str))
    macro_f1 = float(f1_score(y_true_str, y_pred_str, average="macro",
                               labels=label_names, zero_division=0))
    micro_f1 = float(f1_score(y_true_str, y_pred_str, average="micro",
                               labels=label_names, zero_division=0))
    weighted_f1 = float(f1_score(y_true_str, y_pred_str, average="weighted",
                                  labels=label_names, zero_division=0))

    summary: dict[str, float] = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
    }

    logger.info(
        "Multi-class metrics — Accuracy=%.4f  Macro-F1=%.4f  Micro-F1=%.4f  Weighted-F1=%.4f",
        accuracy, macro_f1, micro_f1, weighted_f1,
    )

    # ------------------------------------------------------------------
    # Per-class metrics
    # ------------------------------------------------------------------
    per_class_precision = precision_score(
        y_true_str, y_pred_str, average=None, labels=label_names, zero_division=0,
    )
    per_class_recall = recall_score(
        y_true_str, y_pred_str, average=None, labels=label_names, zero_division=0,
    )
    per_class_f1 = f1_score(
        y_true_str, y_pred_str, average=None, labels=label_names, zero_division=0,
    )

    # Support from ground truth
    support_series = pd.Series(y_true_str).value_counts()
    support = [int(support_series.get(label, 0)) for label in label_names]

    per_class_df = pd.DataFrame({
        "precision": per_class_precision,
        "recall": per_class_recall,
        "f1": per_class_f1,
        "support": support,
    }, index=label_names)
    per_class_df.index.name = "label"

    per_label_f1: dict[str, float] = {
        label: float(per_class_f1[i]) for i, label in enumerate(label_names)
    }

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm = confusion_matrix(y_true_str, y_pred_str, labels=label_names)

    # ------------------------------------------------------------------
    # Classification report (string)
    # ------------------------------------------------------------------
    report_str = classification_report(
        y_true_str, y_pred_str, labels=label_names, zero_division=0,
    )
    logger.info("Classification Report:\n%s", report_str)

    return {
        "summary": summary,
        "per_class": per_class_df,
        "per_label_f1": per_label_f1,
        "confusion_matrix": cm,
        "classification_report": report_str,
    }


# ---------------------------------------------------------------------------
# 2. Cross-validation (multi-class)
# ---------------------------------------------------------------------------


def cross_validate_multiclass(
    model_fn: Callable[[], Any],
    X: Array,
    y: Array,
    n_folds: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
    label_names: list[str] | None = None,
    stratify: bool = True,
    smote_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """K-Fold cross-validation for multi-class classifiers.

    Parameters
    ----------
    model_fn : callable
        Zero-argument factory returning an unfitted model.
    X : np.ndarray shape (n_samples, n_features)
    y : np.ndarray shape (n_samples,)
    n_folds : int
    random_state : int
    label_names : list[str] or None
    stratify : bool
        If True, use StratifiedKFold to preserve class proportions.
    smote_config : dict or None
        If provided, SMOTE is applied **inside each CV fold** to the
        training portion only (no leakage).  Expected keys:
        - ``k_neighbors`` (int)
        - ``target_min_samples`` (int)
        - ``cap_ratio`` (float)

    Returns
    -------
    dict with keys ``fold_metrics``, ``mean``, ``std``, ``n_folds``.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if label_names is None:
        label_names = sorted(set(str(v) for v in np.unique(y)))

    if stratify:
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    fold_summaries: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y), start=1):
        X_train_fold, X_val = X[train_idx], X[val_idx]
        y_train_fold, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE *inside* the fold — only oversample training portion
        if smote_config is not None:
            from collections import Counter
            from imblearn.over_sampling import SMOTE

            counts = Counter(y_train_fold)
            sampling_strategy = {}
            for lbl, cnt in counts.items():
                if cnt < smote_config["target_min_samples"]:
                    target = min(
                        smote_config["target_min_samples"],
                        int(cnt * smote_config["cap_ratio"]),
                    )
                    if cnt >= smote_config["k_neighbors"] + 1 and target > cnt:
                        sampling_strategy[lbl] = target

            if sampling_strategy:
                try:
                    sm = SMOTE(
                        sampling_strategy=sampling_strategy,
                        k_neighbors=smote_config["k_neighbors"],
                        random_state=random_state + fold_idx,
                    )
                    X_train_fold, y_train_fold = sm.fit_resample(X_train_fold, y_train_fold)
                except Exception as exc:
                    logger.warning("SMOTE failed on fold %d: %s — continuing without.", fold_idx, exc)

        logger.info(
            "CV fold %d/%d — training on %d, validating on %d",
            fold_idx, n_folds, len(X_train_fold), len(X_val),
        )

        try:
            model = model_fn()
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val)
        except Exception as exc:
            logger.error("Fold %d failed: %s", fold_idx, exc)
            raise

        fold_result = compute_metrics_multiclass(y_val, y_pred, label_names=label_names)
        fold_summaries.append(fold_result["summary"])

        logger.info(
            "Fold %d: %s",
            fold_idx,
            {k: f"{v:.4f}" for k, v in fold_result["summary"].items()},
        )

    # Aggregate
    metric_keys = list(fold_summaries[0].keys())
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}

    for key in metric_keys:
        values = np.array([s[key] for s in fold_summaries], dtype=float)
        mean_metrics[key] = float(values.mean())
        std_metrics[key] = float(values.std())

    logger.info("CV mean: %s", {k: f"{v:.4f}" for k, v in mean_metrics.items()})
    logger.info("CV std:  %s", {k: f"{v:.4f}" for k, v in std_metrics.items()})

    return {
        "fold_metrics": fold_summaries,
        "mean": mean_metrics,
        "std": std_metrics,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# 3. Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    output_path: Path | str | None = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes).
    label_names : list[str]
        Class labels for axes.
    output_path : Path or str or None
        If provided, save figure to this path.
    title : str
        Figure title.
    normalize : bool
        If True, normalise rows so each sums to 1 (recall view).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div-by-zero
        cm_display = cm.astype(float) / row_sums
        fmt = ".2f"
        vmin, vmax = 0, 1
    else:
        cm_display = cm
        fmt = "d"
        vmin, vmax = None, None

    n_classes = len(label_names)
    fig_size = max(6, n_classes * 0.7)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"label": "Recall (row-normalized)" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved -> %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# 4. Per-class F1 bar chart
# ---------------------------------------------------------------------------


def plot_per_class_f1(
    per_class_df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Per-Class F1 Score",
) -> plt.Figure:
    """Horizontal bar chart of per-class F1, sorted descending.

    Parameters
    ----------
    per_class_df : pd.DataFrame
        Must have an ``f1`` column and label names as index.
    output_path : Path or str or None
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    if "f1" not in per_class_df.columns:
        raise ValueError("per_class_df must contain an 'f1' column.")

    sorted_df = per_class_df.sort_values("f1", ascending=True)
    n_classes = len(sorted_df)
    fig_height = max(4, n_classes * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    colours = plt.cm.RdYlGn(sorted_df["f1"].values)
    bars = ax.barh(sorted_df.index, sorted_df["f1"], color=colours,
                   edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, sorted_df["f1"]):
        ax.text(
            min(float(val) + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.3f}",
            va="center", ha="left", fontsize=8,
        )

    mean_f1 = float(sorted_df["f1"].mean())
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(x=mean_f1, color="steelblue", linestyle="--", linewidth=1.2,
               label=f"Mean F1 = {mean_f1:.3f}")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Per-class F1 chart saved -> %s", output_path)

    return fig


# ---------------------------------------------------------------------------
# 5. Save metrics (multi-class)
# ---------------------------------------------------------------------------


def save_metrics_multiclass(
    metrics_dict: MetricsDict,
    model_name: str,
    output_dir: Path | str | None = None,
) -> None:
    """Persist multi-class evaluation results.

    Writes:
    * ``{model_name}_metrics.json`` — scalar summary
    * ``{model_name}_per_class.csv`` — per-class breakdown
    * ``{model_name}_classification_report.txt`` — text report

    Parameters
    ----------
    metrics_dict : dict
        From :func:`compute_metrics_multiclass`.
    model_name : str
        Short identifier for file naming.
    output_dir : Path or str or None
        Defaults to REPORTS_DIR from config.
    """
    output_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = metrics_dict.get("summary", {})
    per_class_df = metrics_dict.get("per_class")

    # Log
    logger.info("=== Multi-class results for '%s' ===", model_name)
    for metric_name, value in summary.items():
        logger.info("  %-20s %.4f", metric_name, value)

    # JSON summary
    try:
        save_json(summary, output_dir / f"{model_name}_metrics.json")
        logger.info("Summary metrics saved -> %s", output_dir / f"{model_name}_metrics.json")
    except Exception as exc:
        logger.error("Failed to save metrics JSON: %s", exc)

    # CSV per-class
    if per_class_df is not None and not per_class_df.empty:
        try:
            save_csv(per_class_df, output_dir / f"{model_name}_per_class.csv", index=True)
            logger.info("Per-class CSV saved -> %s", output_dir / f"{model_name}_per_class.csv")
        except Exception as exc:
            logger.error("Failed to save per-class CSV: %s", exc)

    # Classification report as text
    report_str = metrics_dict.get("classification_report", "")
    if report_str:
        report_path = output_dir / f"{model_name}_classification_report.txt"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            logger.info("Classification report saved -> %s", report_path)
        except Exception as exc:
            logger.error("Failed to save classification report: %s", exc)

    # Confusion matrix plot
    cm = metrics_dict.get("confusion_matrix")
    if cm is not None and per_class_df is not None:
        label_names = per_class_df.index.tolist()
        try:
            plot_confusion_matrix(
                cm, label_names,
                output_path=output_dir / f"{model_name}_confusion_matrix.png",
                title=f"Confusion Matrix — {model_name}",
            )
        except Exception as exc:
            logger.error("Failed to plot confusion matrix: %s", exc)


# ---------------------------------------------------------------------------
# 6. Model comparison (multi-class)
# ---------------------------------------------------------------------------


def compare_models_multiclass(
    results_dict: dict[str, MetricsDict],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build and save a model-comparison table for multi-class results.

    Parameters
    ----------
    results_dict : dict[str, MetricsDict]
        Mapping model_name -> metrics_dict from compute_metrics_multiclass.
    output_dir : Path or str or None

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by macro_f1 descending.
    """
    if not results_dict:
        raise ValueError("results_dict is empty.")

    output_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results_dict.items():
        summary = metrics.get("summary", {})
        row: dict[str, Any] = {"model": model_name, **summary}
        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("model")

    if "macro_f1" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("macro_f1", ascending=False)

    logger.info("Model comparison:\n%s", comparison_df.to_string())

    # Save CSV
    csv_path = output_dir / "model_comparison.csv"
    try:
        save_csv(comparison_df, csv_path, index=True)
        logger.info("Comparison CSV saved -> %s", csv_path)
    except Exception as exc:
        logger.error("Failed to save comparison CSV: %s", exc)

    # Plot comparison
    _plot_comparison_multiclass(comparison_df, output_dir)

    return comparison_df


def _plot_comparison_multiclass(
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Grouped bar chart comparing macro_f1, weighted_f1, accuracy across models."""
    plot_cols = [c for c in comparison_df.columns
                 if c in {"accuracy", "macro_f1", "weighted_f1", "micro_f1"}]
    if not plot_cols:
        logger.warning("No plottable metric columns found.")
        return

    n_models = len(comparison_df)
    n_metrics = len(plot_cols)
    bar_width = 0.8 / n_metrics
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.6), 5))
    cmap = plt.cm.tab10

    for metric_idx, metric in enumerate(plot_cols):
        offsets = x + (metric_idx - n_metrics / 2.0 + 0.5) * bar_width
        values = comparison_df[metric].values.astype(float)
        colour = cmap(metric_idx / max(n_metrics - 1, 1))
        bars = ax.bar(offsets, values, width=bar_width * 0.92, label=metric,
                      color=colour, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Multi-Class Metrics", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    chart_path = output_dir / "model_comparison.png"
    try:
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        logger.info("Comparison chart saved -> %s", chart_path)
    except Exception as exc:
        logger.error("Failed to save comparison chart: %s", exc)
    finally:
        plt.close(fig)