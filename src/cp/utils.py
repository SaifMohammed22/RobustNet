"""Conformal prediction utilities for classification."""
import torch
import numpy as np


def conformal_scores(probas, labels):
    """Nonconformity scores: 1 - softmax prob of the true class.

    Args:
        probas: (N, C) softmax probabilities.
        labels: (N,) true class indices.

    Returns:
        np.ndarray of shape (N,) with scores in [0, 1].
    """
    if not isinstance(probas, np.ndarray):
        probas = probas.detach().cpu().numpy()
    if probas.ndim != 2:
        raise ValueError("probas must be 2D (N, C)")
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    true_probs = probas[np.arange(len(labels)), labels]
    return 1.0 - true_probs


def compute_quantile(scores, alpha):
    """Empirical quantile for split conformal prediction (finite-sample).

    Args:
        scores: (N,) nonconformity scores on the calibration set.
        alpha: target error rate (coverage = 1 - alpha).

    Returns:
        float quantile threshold q such that C(x) = {k : p_k >= 1 - q}.
    """
    n = len(scores)
    if n == 0:
        raise ValueError("Calibration set is empty; cannot compute quantile")
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def prediction_sets(probas, quantile, n_classes=None):
    """Build label-conditional prediction sets from a quantile threshold.

    C(x) = {k : softmax_k(x) >= 1 - quantile}

    Args:
        probas: (N, C) softmax probabilities or raw logits.
        quantile: conformal quantile threshold q.
        n_classes: (optional) number of classes, defaults to probas.shape[1].

    Returns:
        A list of sets.
    """
    probas_np = probas.detach().cpu().numpy() if isinstance(probas, torch.Tensor) else np.asarray(probas)
    if probas_np.ndim != 2:
        raise ValueError("probas must be 2D (N, C)")
    if n_classes is None:
        n_classes = probas_np.shape[1]

    mask = probas_np >= (1.0 - quantile)
    sets = [set(np.where(row)[0].tolist()) for row in mask]
    return sets
