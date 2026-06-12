"""Multi-Modal MLP Fusion Model — Phase 3.3.

This module implements a PyTorch-based Multi-Layer Perceptron that fuses
handcrafted audio features with TF-IDF text features for multi-label music
genre classification.

Architecture overview
---------------------
* Two parallel *input branches* (one per modality) each perform:
      Linear → BatchNorm1d → ReLU → Dropout
* The branch outputs are **concatenated** and fed through a stack of
  *shared hidden layers*  (one per remaining entry in ``hidden_dims``).
* A final linear projection produces raw **logits** (no activation).
  BCEWithLogitsLoss is used during training; sigmoid + threshold is applied
  at inference time.

Phase split
-----------
* ``run_mlp_pipeline``  — FULLY EXECUTABLE (traditional hand-crafted features).
* ``run_mlp_dl_pipeline`` — CODE ONLY / DEFERRED.  The function is complete
  and correct but is intentionally **not called** from ``train_baseline.py``
  during the baseline phase.  See the prominent comment inside the function.

Milestone: Phase 3 — Model Training & Evaluation
Sub-phase: 3.3 — Neural Network Fusion
"""

from __future__ import annotations

# ============================================================
# ⚠️  DEFERRED-EXECUTION WARNING
# ============================================================
# ``run_mlp_dl_pipeline`` is implemented but NOT executed during
# the baseline phase.  It is marked with a
#   # DEFERRED: not executed during baseline phase
# comment at its definition.  Do not call it from train_baseline.py.
# ============================================================

import copy
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src2.config import (
    BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EARLY_STOPPING_PATIENCE,
    MLP_EPOCHS,
    MLP_HIDDEN_DIMS,
    MLP_LR,
    MODELS_DIR,
    RANDOM_SEED,
)
from src2.models.evaluation import compute_metrics
from src2.utils.io_utils import save_pickle

log = logging.getLogger("music_genre")

# ---------------------------------------------------------------------------
# Helper: build one input branch
# ---------------------------------------------------------------------------

def _build_branch(input_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    """Build a single input-modality processing branch.

    Parameters
    ----------
    input_dim:
        Dimensionality of the raw input features for this modality.
    out_dim:
        Output dimensionality of the branch (= first entry of hidden_dims).
    dropout:
        Dropout probability applied after ReLU.

    Returns
    -------
    nn.Sequential
        A sequential module: Linear -> BatchNorm1d -> ReLU -> Dropout.
    """
    return nn.Sequential(
        nn.Linear(input_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultiModalDataset(Dataset):
    """PyTorch Dataset for multi-modal (audio + text) feature arrays.

    Parameters
    ----------
    audio_features:
        NumPy array of shape ``(N, audio_dim)`` or ``None`` when the audio
        modality is absent.
    text_features:
        NumPy array of shape ``(N, text_dim)`` or ``None`` when the text
        modality is absent.
    labels:
        NumPy array of shape ``(N, n_classes)`` containing multi-hot label
        vectors (float or int).
    """

    def __init__(
        self,
        audio_features: Optional[np.ndarray],
        text_features: Optional[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        self.audio_features = audio_features
        self.text_features = text_features
        self.labels = labels.astype(np.float32)

        # Infer fallback zero-vector dimensions from whichever array exists.
        self._audio_dim: int = (
            audio_features.shape[1] if audio_features is not None else 0
        )
        self._text_dim: int = (
            text_features.shape[1] if text_features is not None else 0
        )

        n = len(labels)
        if audio_features is not None and len(audio_features) != n:
            raise ValueError(
                f"audio_features length {len(audio_features)} != labels length {n}"
            )
        if text_features is not None and len(text_features) != n:
            raise ValueError(
                f"text_features length {len(text_features)} != labels length {n}"
            )

    def __len__(self) -> int:  # noqa: D105
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sample at *idx*.

        Parameters
        ----------
        idx:
            Integer sample index.

        Returns
        -------
        tuple
            ``(audio_tensor, text_tensor, label_tensor)`` all as float32.
            Missing modalities are represented by zero tensors of their
            original dimension.
        """
        if self.audio_features is not None:
            audio_t = torch.from_numpy(
                self.audio_features[idx].astype(np.float32)
            )
        else:
            audio_t = torch.zeros(self._audio_dim, dtype=torch.float32)

        if self.text_features is not None:
            text_t = torch.from_numpy(
                self.text_features[idx].astype(np.float32)
            )
        else:
            text_t = torch.zeros(self._text_dim, dtype=torch.float32)

        label_t = torch.from_numpy(self.labels[idx])
        return audio_t, text_t, label_t


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiModalMLP(nn.Module):
    """Multi-modal MLP fusion model for multi-label genre classification.

    The network processes audio and text inputs through separate *branch*
    networks, concatenates their outputs, then passes the result through
    shared *hidden* layers before a linear output head.

    Parameters
    ----------
    audio_dim:
        Number of audio input features.
    text_dim:
        Number of text input features.
    hidden_dims:
        Tuple of hidden-layer widths.  The first entry ``hidden_dims[0]``
        is used as the branch output dimension; subsequent entries form the
        shared hidden stack.
    n_classes:
        Number of output classes (genres).
    dropout:
        Dropout probability used in both branch and hidden layers.
    """

    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        hidden_dims: Tuple[int, ...] = MLP_HIDDEN_DIMS,
        n_classes: int = 1,
        dropout: float = MLP_DROPOUT,
    ) -> None:
        super().__init__()

        if len(hidden_dims) < 1:
            raise ValueError("hidden_dims must contain at least one element.")

        branch_out = hidden_dims[0]

        # --- Input branches -----------------------------------------------
        self.audio_branch = _build_branch(audio_dim, branch_out, dropout)
        self.text_branch = _build_branch(text_dim, branch_out, dropout)

        # --- Shared hidden layers (dims after the first) ------------------
        # Fusion concatenates both branch outputs -> 2 * branch_out
        shared_layers: List[nn.Module] = []
        in_dim = branch_out * 2
        for h_dim in hidden_dims[1:]:
            shared_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_dim = h_dim

        self.shared = nn.Sequential(*shared_layers) if shared_layers else nn.Identity()

        # --- Output head --------------------------------------------------
        self.output_layer = nn.Linear(in_dim, n_classes)

        # Store metadata for zero-filling missing modalities at runtime.
        self._audio_dim = audio_dim
        self._text_dim = text_dim

        log.info(
            "MultiModalMLP | audio_dim=%d  text_dim=%d  hidden_dims=%s  "
            "n_classes=%d  dropout=%.2f",
            audio_dim,
            text_dim,
            hidden_dims,
            n_classes,
            dropout,
        )

    def forward(
        self,
        audio_x: Optional[torch.Tensor],
        text_x: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute raw logits for a batch.

        Parameters
        ----------
        audio_x:
            Float tensor of shape ``(B, audio_dim)`` or ``None``.
            If ``None``, a zero tensor of the correct shape is substituted.
        text_x:
            Float tensor of shape ``(B, text_dim)`` or ``None``.
            If ``None``, a zero tensor of the correct shape is substituted.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, n_classes)``.  Apply sigmoid for
            probabilities; BCEWithLogitsLoss handles this internally during
            training.
        """
        # Handle missing modalities with zero tensors.
        if audio_x is None:
            batch = text_x.size(0) if text_x is not None else 1
            device = next(self.parameters()).device
            audio_x = torch.zeros(batch, self._audio_dim, device=device)

        if text_x is None:
            batch = audio_x.size(0)
            device = next(self.parameters()).device
            text_x = torch.zeros(batch, self._text_dim, device=device)

        # Branch processing.
        audio_out = self.audio_branch(audio_x)   # (B, branch_out)
        text_out = self.text_branch(text_x)       # (B, branch_out)

        # Fusion.
        fused = torch.cat([audio_out, text_out], dim=1)  # (B, 2*branch_out)

        # Shared hidden layers.
        hidden = self.shared(fused)

        # Raw logits — no sigmoid here.
        logits = self.output_layer(hidden)
        return logits


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mlp(
    model: MultiModalMLP,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = MLP_EPOCHS,
    lr: float = MLP_LR,
    patience: int = MLP_EARLY_STOPPING_PATIENCE,
    class_weights: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Tuple[MultiModalMLP, Dict]:
    """Train a ``MultiModalMLP`` with early stopping.

    Uses BCEWithLogitsLoss with optional per-class positive weights to
    address class imbalance.  The Adam optimiser is used throughout.

    Parameters
    ----------
    model:
        The instantiated (untrained) model.
    train_loader:
        DataLoader yielding ``(audio, text, labels)`` batches for training.
    val_loader:
        DataLoader yielding ``(audio, text, labels)`` batches for validation.
    epochs:
        Maximum number of training epochs.
    lr:
        Adam learning rate.
    patience:
        Number of epochs without validation-loss improvement before stopping.
    class_weights:
        Optional 1-D tensor of shape ``(n_classes,)`` used as ``pos_weight``
        in BCEWithLogitsLoss to handle class imbalance.
    device:
        PyTorch device string, e.g. ``'cpu'`` or ``'cuda'``.

    Returns
    -------
    tuple
        ``(best_model, history)`` where ``best_model`` has the weights of
        the epoch with lowest validation loss, and ``history`` is a dict
        with keys ``train_loss``, ``val_loss``, ``train_f1``, ``val_f1``
        (each a list of per-epoch values).
    """
    from sklearn.metrics import f1_score as _f1  # lazy import to avoid top-level dep

    torch_device = torch.device(device)
    model = model.to(torch_device)

    pos_weight: Optional[torch.Tensor] = (
        class_weights.to(torch_device) if class_weights is not None else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    best_val_loss = float("inf")
    best_state: Optional[Dict] = None
    epochs_no_improve = 0

    log.info(
        "Training MultiModalMLP for up to %d epochs (patience=%d, lr=%.4e, device=%s)",
        epochs,
        patience,
        lr,
        device,
    )

    for epoch in range(1, epochs + 1):
        # ---- Training pass -----------------------------------------------
        model.train()
        train_loss_acc = 0.0
        all_train_true: List[np.ndarray] = []
        all_train_pred: List[np.ndarray] = []

        for audio_batch, text_batch, label_batch in train_loader:
            audio_batch = audio_batch.to(torch_device)
            text_batch = text_batch.to(torch_device)
            label_batch = label_batch.to(torch_device)

            optimiser.zero_grad()
            logits = model(audio_batch, text_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimiser.step()

            train_loss_acc += loss.item() * audio_batch.size(0)

            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_train_true.append(label_batch.cpu().numpy().astype(int))
            all_train_pred.append(preds)

        n_train = len(train_loader.dataset)  # type: ignore[arg-type]
        avg_train_loss = train_loss_acc / max(n_train, 1)
        Y_true_tr = np.vstack(all_train_true)
        Y_pred_tr = np.vstack(all_train_pred)
        try:
            train_f1 = float(_f1(Y_true_tr, Y_pred_tr, average="macro", zero_division=0))
        except Exception:
            train_f1 = 0.0

        # ---- Validation pass ---------------------------------------------
        model.eval()
        val_loss_acc = 0.0
        all_val_true: List[np.ndarray] = []
        all_val_pred: List[np.ndarray] = []

        with torch.no_grad():
            for audio_batch, text_batch, label_batch in val_loader:
                audio_batch = audio_batch.to(torch_device)
                text_batch = text_batch.to(torch_device)
                label_batch = label_batch.to(torch_device)

                logits = model(audio_batch, text_batch)
                loss = criterion(logits, label_batch)
                val_loss_acc += loss.item() * audio_batch.size(0)

                preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
                all_val_true.append(label_batch.cpu().numpy().astype(int))
                all_val_pred.append(preds)

        n_val = len(val_loader.dataset)  # type: ignore[arg-type]
        avg_val_loss = val_loss_acc / max(n_val, 1)
        Y_true_val = np.vstack(all_val_true)
        Y_pred_val = np.vstack(all_val_pred)
        try:
            val_f1 = float(_f1(Y_true_val, Y_pred_val, average="macro", zero_division=0))
        except Exception:
            val_f1 = 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        log.info(
            "Epoch %3d/%d | train_loss=%.4f  val_loss=%.4f  "
            "train_f1=%.4f  val_f1=%.4f",
            epoch,
            epochs,
            avg_train_loss,
            avg_val_loss,
            train_f1,
            val_f1,
        )

        # ---- Early stopping check ----------------------------------------
        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info(
                    "Early stopping triggered at epoch %d (patience=%d).",
                    epoch,
                    patience,
                )
                break

    # Restore best weights.
    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(
            "Restored model weights from best checkpoint (val_loss=%.4f).",
            best_val_loss,
        )

    return model, history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_mlp(
    model: MultiModalMLP,
    data_loader: DataLoader,
    threshold: float = 0.5,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return ground-truth / prediction arrays.

    Parameters
    ----------
    model:
        A trained ``MultiModalMLP`` instance.
    data_loader:
        DataLoader yielding ``(audio, text, labels)`` batches.
    threshold:
        Sigmoid probability threshold for converting to binary predictions.
    device:
        PyTorch device string.

    Returns
    -------
    tuple
        ``(Y_true, Y_pred)`` — both int NumPy arrays of shape
        ``(N, n_classes)``.
    """
    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    with torch.no_grad():
        for audio_batch, text_batch, label_batch in data_loader:
            audio_batch = audio_batch.to(torch_device)
            text_batch = text_batch.to(torch_device)

            logits = model(audio_batch, text_batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= threshold).astype(int)

            all_true.append(label_batch.numpy().astype(int))
            all_pred.append(preds)

    Y_true = np.vstack(all_true)
    Y_pred = np.vstack(all_pred)
    return Y_true, Y_pred


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _build_pos_weight(Y_train: np.ndarray) -> torch.Tensor:
    """Compute per-class positive weights from training labels.

    Uses the ratio ``n_negative / n_positive`` per class, clipped to
    ``[0.1, 50.0]`` to avoid extreme values.

    Parameters
    ----------
    Y_train:
        Binary label matrix of shape ``(N, n_classes)``.

    Returns
    -------
    torch.Tensor
        1-D float tensor of shape ``(n_classes,)``.
    """
    n = Y_train.shape[0]
    pos = Y_train.sum(axis=0).astype(float)
    neg = n - pos
    # Avoid division by zero for labels with no positive examples.
    pos = np.where(pos == 0, 1.0, pos)
    weights = np.clip(neg / pos, 0.1, 50.0)
    return torch.tensor(weights, dtype=torch.float32)


def _resolve_device() -> str:
    """Return ``'cuda'`` if a GPU is available, otherwise ``'cpu'``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Traditional-feature pipeline (FULLY EXECUTABLE)
# ---------------------------------------------------------------------------

def run_mlp_pipeline(
    X_audio_train: np.ndarray,
    X_text_train: np.ndarray,
    Y_train: np.ndarray,
    X_audio_val: np.ndarray,
    X_text_val: np.ndarray,
    Y_val: np.ndarray,
    X_audio_test: np.ndarray,
    X_text_test: np.ndarray,
    Y_test: np.ndarray,
    label_names: List[str],
    output_dir: Path = MODELS_DIR,
    config: Optional[Dict] = None,
) -> Dict:
    """End-to-end MLP training pipeline for traditional hand-crafted features.

    This is the primary entry point called by ``train_baseline.py`` for
    Phase 3.3.  It uses pre-computed audio (e.g. MFCC statistics) and
    text (TF-IDF) feature matrices.

    Parameters
    ----------
    X_audio_train:
        Audio feature matrix for training, shape ``(N_train, audio_dim)``.
    X_text_train:
        Text feature matrix for training, shape ``(N_train, text_dim)``.
    Y_train:
        Multi-hot label matrix for training, shape ``(N_train, n_classes)``.
    X_audio_val:
        Audio feature matrix for validation, shape ``(N_val, audio_dim)``.
    X_text_val:
        Text feature matrix for validation, shape ``(N_val, text_dim)``.
    Y_val:
        Multi-hot label matrix for validation, shape ``(N_val, n_classes)``.
    X_audio_test:
        Audio feature matrix for test, shape ``(N_test, audio_dim)``.
    X_text_test:
        Text feature matrix for test, shape ``(N_test, text_dim)``.
    Y_test:
        Multi-hot label matrix for test, shape ``(N_test, n_classes)``.
    label_names:
        Ordered list of genre label strings.
    output_dir:
        Directory where the trained model and artefacts will be saved.
    config:
        Optional dict of hyperparameter overrides.  Supported keys:
        ``hidden_dims``, ``dropout``, ``lr``, ``epochs``, ``patience``,
        ``batch_size``.

    Returns
    -------
    dict
        Metrics dictionary as returned by ``compute_metrics``, augmented with
        a ``"history"`` key containing per-epoch train/val loss and F1 lists.
    """
    cfg = config or {}
    hidden_dims: Tuple[int, ...] = tuple(cfg.get("hidden_dims", MLP_HIDDEN_DIMS))
    dropout: float = float(cfg.get("dropout", MLP_DROPOUT))
    lr: float = float(cfg.get("lr", MLP_LR))
    epochs: int = int(cfg.get("epochs", MLP_EPOCHS))
    patience: int = int(cfg.get("patience", MLP_EARLY_STOPPING_PATIENCE))
    batch_size: int = int(cfg.get("batch_size", BATCH_SIZE))

    device = _resolve_device()
    log.info(
        "run_mlp_pipeline | audio_dim=%d  text_dim=%d  n_classes=%d  device=%s",
        X_audio_train.shape[1],
        X_text_train.shape[1],
        Y_train.shape[1],
        device,
    )

    # Reproducibility.
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Datasets & loaders -----------------------------------------------
    train_ds = MultiModalDataset(X_audio_train, X_text_train, Y_train)
    val_ds = MultiModalDataset(X_audio_val, X_text_val, Y_val)
    test_ds = MultiModalDataset(X_audio_test, X_text_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # --- Build model -------------------------------------------------------
    model = MultiModalMLP(
        audio_dim=X_audio_train.shape[1],
        text_dim=X_text_train.shape[1],
        hidden_dims=hidden_dims,
        n_classes=Y_train.shape[1],
        dropout=dropout,
    )

    # --- Class-imbalance weights ------------------------------------------
    try:
        pos_weight = _build_pos_weight(Y_train)
        log.info("Computed pos_weight for BCEWithLogitsLoss.")
    except Exception as exc:
        log.warning("Could not compute pos_weight: %s — using None.", exc)
        pos_weight = None

    # --- Train -------------------------------------------------------------
    try:
        model, history = train_mlp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            patience=patience,
            class_weights=pos_weight,
            device=device,
        )
    except Exception as exc:
        log.error("Training failed: %s", exc, exc_info=True)
        raise

    # --- Evaluate on test set ---------------------------------------------
    try:
        Y_true, Y_pred = evaluate_mlp(model, test_loader, threshold=0.5, device=device)
        metrics = compute_metrics(Y_true, Y_pred, label_names)
        metrics["history"] = history
        log.info(
            "Test results | macro_f1=%.4f  micro_f1=%.4f  hamming=%.4f",
            metrics.get("macro_f1", float("nan")),
            metrics.get("micro_f1", float("nan")),
            metrics.get("hamming_loss", float("nan")),
        )
    except Exception as exc:
        log.error("Evaluation failed: %s", exc, exc_info=True)
        raise

    # --- Persist model & artefacts ----------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "mlp_traditional.pt"
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "audio_dim": X_audio_train.shape[1],
                "text_dim": X_text_train.shape[1],
                "n_classes": Y_train.shape[1],
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "label_names": label_names,
            },
            model_path,
        )
        log.info("Model saved -> %s", model_path)
    except Exception as exc:
        log.error("Failed to save model: %s", exc, exc_info=True)

    history_path = output_dir / "mlp_traditional_history.pkl"
    try:
        save_pickle(history, history_path)
        log.info("Training history saved -> %s", history_path)
    except Exception as exc:
        log.warning("Could not save history: %s", exc)

    metrics_path = output_dir / "mlp_traditional_metrics.pkl"
    try:
        save_pickle(metrics, metrics_path)
        log.info("Metrics saved -> %s", metrics_path)
    except Exception as exc:
        log.warning("Could not save metrics: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# DL-embedding pipeline — CODE ONLY, NOT EXECUTED DURING BASELINE PHASE
# ---------------------------------------------------------------------------

# DEFERRED: not executed during baseline phase
def run_mlp_dl_pipeline(
    X_audio_train: np.ndarray,
    X_text_train: np.ndarray,
    Y_train: np.ndarray,
    X_audio_val: np.ndarray,
    X_text_val: np.ndarray,
    Y_val: np.ndarray,
    X_audio_test: np.ndarray,
    X_text_test: np.ndarray,
    Y_test: np.ndarray,
    label_names: List[str],
    output_dir: Path = MODELS_DIR,
    config: Optional[Dict] = None,
) -> Dict:
    """MLP training pipeline for deep-learning embedding features.

    .. warning::
        **DEFERRED — NOT CALLED DURING BASELINE PHASE.**

        This function is complete and correct but is intentionally excluded
        from ``train_baseline.py``.  It is reserved for the DL feature
        extraction phase where audio embeddings (e.g. from a CNN or
        transformer) and text embeddings (e.g. from a sentence encoder) are
        available instead of hand-crafted features.

        The API is identical to :func:`run_mlp_pipeline`; only the saved
        artefact filenames differ so both pipelines can coexist in the same
        ``output_dir``.

    Parameters
    ----------
    X_audio_train:
        DL audio embedding matrix for training, shape ``(N_train, audio_emb_dim)``.
    X_text_train:
        DL text embedding matrix for training, shape ``(N_train, text_emb_dim)``.
    Y_train:
        Multi-hot label matrix for training, shape ``(N_train, n_classes)``.
    X_audio_val:
        DL audio embedding matrix for validation, shape ``(N_val, audio_emb_dim)``.
    X_text_val:
        DL text embedding matrix for validation, shape ``(N_val, text_emb_dim)``.
    Y_val:
        Multi-hot label matrix for validation, shape ``(N_val, n_classes)``.
    X_audio_test:
        DL audio embedding matrix for test, shape ``(N_test, audio_emb_dim)``.
    X_text_test:
        DL text embedding matrix for test, shape ``(N_test, text_emb_dim)``.
    Y_test:
        Multi-hot label matrix for test, shape ``(N_test, n_classes)``.
    label_names:
        Ordered list of genre label strings.
    output_dir:
        Directory where the trained model and artefacts will be saved.
    config:
        Optional dict of hyperparameter overrides (same keys as
        :func:`run_mlp_pipeline`).

    Returns
    -------
    dict
        Metrics dictionary augmented with a ``"history"`` key.
    """
    # DEFERRED: not executed during baseline phase

    cfg = config or {}
    hidden_dims: Tuple[int, ...] = tuple(cfg.get("hidden_dims", MLP_HIDDEN_DIMS))
    dropout: float = float(cfg.get("dropout", MLP_DROPOUT))
    lr: float = float(cfg.get("lr", MLP_LR))
    epochs: int = int(cfg.get("epochs", MLP_EPOCHS))
    patience: int = int(cfg.get("patience", MLP_EARLY_STOPPING_PATIENCE))
    batch_size: int = int(cfg.get("batch_size", BATCH_SIZE))

    device = _resolve_device()
    log.info(
        "run_mlp_dl_pipeline | audio_emb_dim=%d  text_emb_dim=%d  "
        "n_classes=%d  device=%s",
        X_audio_train.shape[1],
        X_text_train.shape[1],
        Y_train.shape[1],
        device,
    )

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Datasets & loaders -----------------------------------------------
    train_ds = MultiModalDataset(X_audio_train, X_text_train, Y_train)
    val_ds = MultiModalDataset(X_audio_val, X_text_val, Y_val)
    test_ds = MultiModalDataset(X_audio_test, X_text_test, Y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # --- Build model -------------------------------------------------------
    model = MultiModalMLP(
        audio_dim=X_audio_train.shape[1],
        text_dim=X_text_train.shape[1],
        hidden_dims=hidden_dims,
        n_classes=Y_train.shape[1],
        dropout=dropout,
    )

    # --- Class-imbalance weights ------------------------------------------
    try:
        pos_weight = _build_pos_weight(Y_train)
    except Exception as exc:
        log.warning("Could not compute pos_weight: %s — using None.", exc)
        pos_weight = None

    # --- Train -------------------------------------------------------------
    try:
        model, history = train_mlp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            patience=patience,
            class_weights=pos_weight,
            device=device,
        )
    except Exception as exc:
        log.error("DL pipeline training failed: %s", exc, exc_info=True)
        raise

    # --- Evaluate on test set ---------------------------------------------
    try:
        Y_true, Y_pred = evaluate_mlp(model, test_loader, threshold=0.5, device=device)
        metrics = compute_metrics(Y_true, Y_pred, label_names)
        metrics["history"] = history
        log.info(
            "DL test results | macro_f1=%.4f  micro_f1=%.4f  hamming=%.4f",
            metrics.get("macro_f1", float("nan")),
            metrics.get("micro_f1", float("nan")),
            metrics.get("hamming_loss", float("nan")),
        )
    except Exception as exc:
        log.error("DL evaluation failed: %s", exc, exc_info=True)
        raise

    # --- Persist model & artefacts ----------------------------------------
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "mlp_dl_embeddings.pt"
    try:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "audio_dim": X_audio_train.shape[1],
                "text_dim": X_text_train.shape[1],
                "n_classes": Y_train.shape[1],
                "hidden_dims": hidden_dims,
                "dropout": dropout,
                "label_names": label_names,
            },
            model_path,
        )
        log.info("DL model saved -> %s", model_path)
    except Exception as exc:
        log.error("Failed to save DL model: %s", exc, exc_info=True)

    history_path = output_dir / "mlp_dl_embeddings_history.pkl"
    try:
        save_pickle(history, history_path)
        log.info("DL training history saved -> %s", history_path)
    except Exception as exc:
        log.warning("Could not save DL history: %s", exc)

    metrics_path = output_dir / "mlp_dl_embeddings_metrics.pkl"
    try:
        save_pickle(metrics, metrics_path)
        log.info("DL metrics saved -> %s", metrics_path)
    except Exception as exc:
        log.warning("Could not save DL metrics: %s", exc)

    return metrics
