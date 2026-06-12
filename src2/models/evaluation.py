"""Multi-label evaluation framework for the Multi-Modal Music Genre Classification project.

Phase 3.1 — Evaluation Framework
==================================
This module provides a complete, reusable evaluation suite for any multi-label
genre classifier trained within the project.  Every public function operates on
plain NumPy arrays so it is agnostic of the underlying model type (scikit-learn,
PyTorch, etc.).

Public API
----------
compute_metrics(Y_true, Y_pred, label_names=None) -> dict
    Compute the standard multi-label scalar metrics and a per-class breakdown.

cross_validate_multilabel(model_fn, X, Y, n_folds, random_state) -> dict
    K-Fold cross-validation returning mean +/- std of each metric.

plot_per_class_metrics(per_class_df, output_path=None)
    Horizontal bar chart of per-class F1, sorted descending.

save_results(metrics_dict, model_name, output_dir)
    Persist the metric dict to JSON and the per-class DataFrame to CSV.

compare_models(results_dict, output_dir) -> pd.DataFrame
    Build a model-comparison table, save it, and plot a grouped bar chart.

Dependencies
------------
scikit-learn, matplotlib, pandas, numpy.
All project-level paths / hyper-parameters are imported from src2.config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; safe for servers / notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold

from src2.config import (
    FIGURES_DIR,  # noqa: F401 – imported so callers can reference it
    N_CV_FOLDS,
    RANDOM_SEED,
    REPORTS_DIR,
)
from src2.utils.io_utils import save_csv, save_json

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Array = np.ndarray
MetricsDict = dict[str, Any]


# ---------------------------------------------------------------------------
# 1. Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    Y_true: Array,
    Y_pred: Array,
    label_names: list[str] | None = None,
) -> MetricsDict:
    """Compute a comprehensive set of multi-label classification metrics.

    Parameters
    ----------
    Y_true : np.ndarray of shape (n_samples, n_labels)
        Binary ground-truth label matrix.
    Y_pred : np.ndarray of shape (n_samples, n_labels)
        Binary predicted label matrix.
    label_names : list[str] or None
        Human-readable names for each label column.  When ``None`` the labels
        are named ``label_0``, ``label_1``, ... automatically.

    Returns
    -------
    dict
        A dict with two keys:

        ``"summary"`` : dict[str, float]
            Scalar aggregate metrics (macro/micro/weighted F1, Hamming loss,
            exact match ratio).

        ``"per_class"`` : pd.DataFrame
            DataFrame indexed by label name with columns
            ``precision``, ``recall``, ``f1``, ``support``.

    Raises
    ------
    ValueError
        If Y_true and Y_pred have mismatched shapes.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)

    if Y_true.shape != Y_pred.shape:
        raise ValueError(
            f"Shape mismatch: Y_true={Y_true.shape}, Y_pred={Y_pred.shape}"
        )

    n_labels = Y_true.shape[1] if Y_true.ndim == 2 else 1
    if label_names is None:
        label_names = [f"label_{i}" for i in range(n_labels)]

    # ------------------------------------------------------------------
    # Aggregate (scalar) metrics
    # ------------------------------------------------------------------
    try:
        macro_f1 = float(f1_score(Y_true, Y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(Y_true, Y_pred, average="micro", zero_division=0))
        weighted_f1 = float(
            f1_score(Y_true, Y_pred, average="weighted", zero_division=0)
        )
        h_loss = float(hamming_loss(Y_true, Y_pred))
        exact_match = float(accuracy_score(Y_true, Y_pred))
    except Exception as exc:
        logger.error("Failed to compute aggregate metrics: %s", exc)
        raise

    summary: dict[str, float] = {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "hamming_loss": h_loss,
        "exact_match_ratio": exact_match,
    }

    logger.info(
        "Metrics — macro_f1=%.4f  micro_f1=%.4f  weighted_f1=%.4f  "
        "hamming_loss=%.4f  exact_match=%.4f",
        macro_f1,
        micro_f1,
        weighted_f1,
        h_loss,
        exact_match,
    )

    # ------------------------------------------------------------------
    # Per-class precision / recall / F1 / support
    # ------------------------------------------------------------------
    try:
        per_class_precision = precision_score(
            Y_true, Y_pred, average=None, zero_division=0
        )
        per_class_recall = recall_score(
            Y_true, Y_pred, average=None, zero_division=0
        )
        per_class_f1 = f1_score(Y_true, Y_pred, average=None, zero_division=0)
        support = Y_true.sum(axis=0).astype(int)
    except Exception as exc:
        logger.error("Failed to compute per-class metrics: %s", exc)
        raise

    per_class_df = pd.DataFrame(
        {
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1": per_class_f1,
            "support": support,
        },
        index=label_names,
    )
    per_class_df.index.name = "label"

    return {"summary": summary, "per_class": per_class_df}


# ---------------------------------------------------------------------------
# 2. Cross-validation
# ---------------------------------------------------------------------------


def cross_validate_multilabel(
    model_fn: Callable[[], Any],
    X: Array,
    Y: Array,
    n_folds: int = N_CV_FOLDS,
    random_state: int = RANDOM_SEED,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """K-Fold cross-validation for any multi-label classifier.

    Parameters
    ----------
    model_fn : callable
        A zero-argument factory that returns a fresh, **unfitted** model each
        time it is called.  The model must expose ``fit(X, Y)`` and
        ``predict(X)`` compatible with the scikit-learn API.
    X : np.ndarray of shape (n_samples, n_features)
        Feature matrix.
    Y : np.ndarray of shape (n_samples, n_labels)
        Binary label matrix.
    n_folds : int
        Number of cross-validation folds.  Defaults to ``N_CV_FOLDS`` from
        ``src2.config``.
    random_state : int
        Random seed for fold splitting.  Defaults to ``RANDOM_SEED`` from
        ``src2.config``.
    label_names : list[str] or None
        Optional label names forwarded to :func:`compute_metrics`.

    Returns
    -------
    dict
        Keys:

        ``"fold_metrics"`` : list[dict]
            Raw ``summary`` dict for each fold.

        ``"mean"`` : dict[str, float]
            Mean of each scalar metric across folds.

        ``"std"`` : dict[str, float]
            Standard deviation of each scalar metric across folds.

        ``"n_folds"`` : int
            Number of folds actually used.

    Raises
    ------
    RuntimeError
        If any fold fails during fit / predict (original exception is re-raised).
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_summaries: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        logger.info(
            "CV fold %d/%d — training on %d samples, validating on %d samples",
            fold_idx,
            n_folds,
            len(train_idx),
            len(val_idx),
        )

        try:
            model = model_fn()
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_val)
        except Exception as exc:
            logger.error("Fold %d failed during fit/predict: %s", fold_idx, exc)
            raise

        fold_result = compute_metrics(Y_val, Y_pred, label_names=label_names)
        fold_summaries.append(fold_result["summary"])

        logger.info(
            "Fold %d results: %s",
            fold_idx,
            {k: f"{v:.4f}" for k, v in fold_result["summary"].items()},
        )

    # Aggregate across folds
    metric_keys = list(fold_summaries[0].keys())
    mean_metrics: dict[str, float] = {}
    std_metrics: dict[str, float] = {}

    for key in metric_keys:
        values = np.array([s[key] for s in fold_summaries], dtype=float)
        mean_metrics[key] = float(values.mean())
        std_metrics[key] = float(values.std())

    logger.info(
        "CV complete — mean metrics: %s",
        {k: f"{v:.4f}" for k, v in mean_metrics.items()},
    )
    logger.info(
        "CV complete — std  metrics: %s",
        {k: f"{v:.4f}" for k, v in std_metrics.items()},
    )

    return {
        "fold_metrics": fold_summaries,
        "mean": mean_metrics,
        "std": std_metrics,
        "n_folds": n_folds,
    }


# ---------------------------------------------------------------------------
# 3. Per-class bar chart
# ---------------------------------------------------------------------------


def plot_per_class_metrics(
    per_class_df: pd.DataFrame,
    output_path: Path | str | None = None,
    title: str = "Per-Class F1 Score",
) -> plt.Figure:
    """Plot a horizontal bar chart of per-class F1 scores, sorted descending.

    Parameters
    ----------
    per_class_df : pd.DataFrame
        DataFrame with at minimum an ``f1`` column and label names as the
        index (as returned by :func:`compute_metrics`).
    output_path : Path or str or None
        If provided, the figure is saved to this path.  The parent directory
        is created automatically.  Supported extensions: .png, .pdf, .svg.
    title : str
        Figure title shown above the chart.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (caller may display or further customise it).

    Raises
    ------
    ValueError
        If ``per_class_df`` does not contain an ``f1`` column.
    """
    if "f1" not in per_class_df.columns:
        raise ValueError("per_class_df must contain an 'f1' column.")

    # Sort ascending so the highest F1 appears at the top of a horizontal bar
    sorted_df = per_class_df.sort_values("f1", ascending=True)

    n_classes = len(sorted_df)
    fig_height = max(4, int(n_classes * 0.35))
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Colour-map: red (low F1) -> yellow -> green (high F1)
    colours = plt.cm.RdYlGn(sorted_df["f1"].values)  # type: ignore[attr-defined]
    bars = ax.barh(
        sorted_df.index,
        sorted_df["f1"],
        color=colours,
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate bar ends with the numeric F1 value
    for bar, val in zip(bars, sorted_df["f1"]):
        ax.text(
            min(float(val) + 0.01, 0.98),
            bar.get_y() + bar.get_height() / 2.0,
            f"{val:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    mean_f1 = float(sorted_df["f1"].mean())
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(
        x=mean_f1,
        color="steelblue",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean F1 = {mean_f1:.3f}",
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            logger.info("Per-class F1 chart saved -> %s", output_path)
        except Exception as exc:
            logger.error(
                "Could not save per-class chart to %s: %s", output_path, exc
            )

    return fig


# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------


def save_results(
    metrics_dict: MetricsDict,
    model_name: str,
    output_dir: Path | str | None = None,
) -> None:
    """Persist evaluation results (JSON + CSV) and log the summary.

    Writes two artefacts inside *output_dir*:

    * ``{model_name}_metrics.json``  — scalar summary metrics.
    * ``{model_name}_per_class.csv`` — per-class precision / recall / F1.

    Parameters
    ----------
    metrics_dict : dict
        Dict as returned by :func:`compute_metrics`, containing
        ``"summary"`` (scalar dict) and ``"per_class"`` (DataFrame) keys.
    model_name : str
        Short identifier used to name the output files.
    output_dir : Path or str or None
        Directory to write outputs into.  Defaults to ``REPORTS_DIR`` from
        ``src2.config``.

    Returns
    -------
    None
    """
    output_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, float] = metrics_dict.get("summary", {})
    per_class_df: pd.DataFrame | None = metrics_dict.get("per_class")

    # Log summary to console / log file
    logger.info("=== Evaluation results for model '%s' ===", model_name)
    for metric_name, value in summary.items():
        logger.info("  %-25s %.4f", metric_name, value)

    # ------------------------------------------------------------------
    # Persist scalar metrics as JSON
    # ------------------------------------------------------------------
    json_path = output_dir / f"{model_name}_metrics.json"
    try:
        save_json(summary, json_path)
        logger.info("Summary metrics saved -> %s", json_path)
    except Exception as exc:
        logger.error(
            "Failed to save metrics JSON for '%s': %s", model_name, exc
        )

    # ------------------------------------------------------------------
    # Persist per-class breakdown as CSV
    # ------------------------------------------------------------------
    if per_class_df is not None and not per_class_df.empty:
        csv_path = output_dir / f"{model_name}_per_class.csv"
        try:
            save_csv(per_class_df, csv_path, index=True)
            logger.info("Per-class metrics saved -> %s", csv_path)
        except Exception as exc:
            logger.error(
                "Failed to save per-class CSV for '%s': %s", model_name, exc
            )
    else:
        logger.warning(
            "No per-class DataFrame found in metrics_dict for '%s'.", model_name
        )


# ---------------------------------------------------------------------------
# 5. Model comparison
# ---------------------------------------------------------------------------


def compare_models(
    results_dict: dict[str, MetricsDict],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Build and save a model-comparison table, then plot a grouped bar chart.

    Parameters
    ----------
    results_dict : dict[str, MetricsDict]
        Mapping of ``model_name -> metrics_dict`` where each ``metrics_dict``
        is the output of :func:`compute_metrics` (must contain a ``"summary"``
        key with scalar metrics).
    output_dir : Path or str or None
        Directory to write outputs into.  Defaults to ``REPORTS_DIR`` from
        ``src2.config``.

    Returns
    -------
    pd.DataFrame
        Comparison table with models as rows and metrics as columns, sorted
        by ``macro_f1`` descending.

    Raises
    ------
    ValueError
        If *results_dict* is empty.
    """
    if not results_dict:
        raise ValueError("results_dict is empty — nothing to compare.")

    output_dir = Path(output_dir) if output_dir is not None else REPORTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build comparison DataFrame
    # ------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    for model_name, metrics in results_dict.items():
        summary = metrics.get("summary", {})
        row: dict[str, Any] = {"model": model_name, **summary}
        rows.append(row)

    comparison_df = pd.DataFrame(rows).set_index("model")

    if "macro_f1" in comparison_df.columns:
        comparison_df = comparison_df.sort_values("macro_f1", ascending=False)

    logger.info("Model comparison table:\n%s", comparison_df.to_string())

    # ------------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------------
    csv_path = output_dir / "model_comparison.csv"
    try:
        save_csv(comparison_df, csv_path, index=True)
        logger.info("Model comparison CSV saved -> %s", csv_path)
    except Exception as exc:
        logger.error("Failed to save comparison CSV: %s", exc)

    # ------------------------------------------------------------------
    # Render charts
    # ------------------------------------------------------------------
    _plot_model_comparison(comparison_df, output_dir)

    return comparison_df


# ---------------------------------------------------------------------------
# Internal helper — grouped bar chart
# ---------------------------------------------------------------------------


def _plot_model_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Render and save grouped bar charts comparing metrics across models.

    Two charts are produced:

    1. A grouped bar chart for bounded [0, 1] metrics
       (macro/micro/weighted F1, exact match ratio).
    2. A separate bar chart for Hamming Loss (scale may differ; lower is better).

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Models as rows, metrics as columns (as built in :func:`compare_models`).
    output_dir : Path
        Directory where the figures are saved.

    Returns
    -------
    None
    """
    # --- Chart 1: bounded metrics -------------------------------------------
    plot_cols = [
        c
        for c in comparison_df.columns
        if c in {"macro_f1", "micro_f1", "weighted_f1", "exact_match_ratio"}
    ]
    if not plot_cols:
        logger.warning(
            "No plottable metric columns found in comparison DataFrame."
        )
    else:
        n_models = len(comparison_df)
        n_metrics = len(plot_cols)
        bar_width = 0.8 / n_metrics
        x = np.arange(n_models)

        fig, ax = plt.subplots(figsize=(max(8, n_models * 1.6), 5))
        cmap = plt.cm.tab10  # type: ignore[attr-defined]

        for metric_idx, metric in enumerate(plot_cols):
            offsets = x + (metric_idx - n_metrics / 2.0 + 0.5) * bar_width
            values = comparison_df[metric].values.astype(float)
            colour = cmap(metric_idx / max(n_metrics - 1, 1))
            bars = ax.bar(
                offsets,
                values,
                width=bar_width * 0.92,
                label=metric,
                color=colour,
                alpha=0.85,
                edgecolor="white",
            )
            for bar, val in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=45,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            comparison_df.index, rotation=15, ha="right", fontsize=9
        )
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(
            "Model Comparison — Multi-Label Metrics",
            fontsize=13,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=9, framealpha=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)
        fig.tight_layout()

        chart_path = output_dir / "model_comparison.png"
        try:
            fig.savefig(chart_path, dpi=150, bbox_inches="tight")
            logger.info("Model comparison chart saved -> %s", chart_path)
        except Exception as exc:
            logger.error("Failed to save model comparison chart: %s", exc)
        finally:
            plt.close(fig)

    # --- Chart 2: Hamming Loss (lower is better) ----------------------------
    if "hamming_loss" not in comparison_df.columns:
        return

    n_models = len(comparison_df)
    fig2, ax2 = plt.subplots(figsize=(max(6, n_models * 1.4), 4))
    hl_values = comparison_df["hamming_loss"].values.astype(float)
    hl_max = float(hl_values.max()) if hl_values.max() > 0 else 1.0
    bar_colours = plt.cm.OrRd(hl_values / hl_max)  # type: ignore[attr-defined]

    b = ax2.bar(
        comparison_df.index,
        hl_values,
        color=bar_colours,
        edgecolor="white",
    )
    for bar, val in zip(b, hl_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            float(bar.get_height()) + hl_max * 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax2.set_title(
        "Model Comparison — Hamming Loss (lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_ylabel("Hamming Loss", fontsize=11)
    ax2.set_xticklabels(
        comparison_df.index, rotation=15, ha="right", fontsize=9
    )
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)
    fig2.tight_layout()

    hl_path = output_dir / "model_comparison_hamming_loss.png"
    try:
        fig2.savefig(hl_path, dpi=150, bbox_inches="tight")
        logger.info("Hamming loss comparison chart saved -> %s", hl_path)
    except Exception as exc:
        logger.error("Failed to save hamming loss chart: %s", exc)
    finally:
        plt.close(fig2)
