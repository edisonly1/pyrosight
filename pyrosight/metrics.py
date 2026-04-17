"""Evaluation metrics for prediction quality, calibration, and uncertainty."""

import numpy as np
from sklearn.metrics import average_precision_score


def compute_f1(preds: np.ndarray, targets: np.ndarray,
               valid_mask: np.ndarray, prev_fire_mask: np.ndarray = None) -> dict:
    """Compute precision, recall, F1 with honest evaluation.

    Excludes invalid pixels (label == -1) and optionally PrevFireMask == 1
    pixels to avoid the 44% F1 inflation from counting already-burning pixels.

    Args:
        preds: Binary predictions, shape (N, H, W).
        targets: Ground truth {0, 1}, shape (N, H, W).
        valid_mask: Boolean mask, shape (N, H, W).
        prev_fire_mask: Optional PrevFireMask, shape (N, H, W). If provided,
            pixels where prev_fire == 1 are excluded from metrics.

    Returns:
        Dict with precision, recall, f1.
    """
    mask = valid_mask.astype(bool)
    if prev_fire_mask is not None:
        mask = mask & (prev_fire_mask < 0.5)

    p = preds[mask]
    t = targets[mask]

    tp = ((p == 1) & (t == 1)).sum()
    fp = ((p == 1) & (t == 0)).sum()
    fn = ((p == 0) & (t == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def compute_auc_pr(probs: np.ndarray, targets: np.ndarray,
                   valid_mask: np.ndarray) -> float:
    """Area Under Precision-Recall curve. Baseline = class prevalence (~0.03)."""
    mask = valid_mask.astype(bool)
    return float(average_precision_score(targets[mask], probs[mask]))


def compute_ece(probs: np.ndarray, targets: np.ndarray,
                valid_mask: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error with equal-width bins."""
    mask = valid_mask.astype(bool)
    p = probs[mask]
    t = targets[mask]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
        if i == 0:
            in_bin = (p >= bin_edges[i]) & (p <= bin_edges[i + 1])

        count = in_bin.sum()
        if count == 0:
            continue

        avg_conf = p[in_bin].mean()
        avg_acc = t[in_bin].mean()
        ece += (count / len(p)) * abs(avg_acc - avg_conf)

    return float(ece)


def compute_brier(probs: np.ndarray, targets: np.ndarray,
                  valid_mask: np.ndarray) -> float:
    """Brier Score: mean squared error of probabilistic predictions."""
    mask = valid_mask.astype(bool)
    return float(((probs[mask] - targets[mask]) ** 2).mean())


def compute_ause(errors: np.ndarray, uncertainties: np.ndarray,
                 valid_mask: np.ndarray, n_points: int = 100) -> float:
    """Area Under Sparsification Error — measures uncertainty-error alignment.

    AUSE = area between model sparsification curve and oracle curve.
    AUSE = 0 means perfect alignment (model knows exactly what it doesn't know).

    Args:
        errors: Per-pixel absolute errors, shape (N,).
        uncertainties: Per-pixel uncertainty values, shape (N,).
        valid_mask: Boolean mask, shape (N,).
        n_points: Number of sparsification fractions to evaluate.

    Returns:
        AUSE score (lower is better).
    """
    mask = valid_mask.astype(bool)
    err = errors[mask]
    unc = uncertainties[mask]
    n = len(err)
    if n == 0:
        return 0.0

    fractions = np.linspace(0, 1, n_points + 1)[:-1]  # exclude 1.0

    # Sort errors by uncertainty (descending) — highest-uncertainty removed first
    model_order = np.argsort(-unc)
    err_by_unc = err[model_order]

    # Sort errors by actual error (descending) — oracle ordering
    oracle_order = np.argsort(-err)
    err_by_err = err[oracle_order]

    # Use cumulative sums for O(n) computation instead of O(n²) np.delete
    cumsum_model = np.cumsum(err_by_unc)
    cumsum_oracle = np.cumsum(err_by_err)
    total_err = cumsum_model[-1]

    model_errors = []
    oracle_errors = []

    for frac in fractions:
        n_remove = int(frac * n)
        n_keep = n - n_remove
        if n_keep == 0:
            break

        # Mean error of remaining pixels = (total - removed) / n_keep
        removed_model = cumsum_model[n_remove - 1] if n_remove > 0 else 0.0
        removed_oracle = cumsum_oracle[n_remove - 1] if n_remove > 0 else 0.0

        model_errors.append((total_err - removed_model) / n_keep)
        oracle_errors.append((total_err - removed_oracle) / n_keep)

    model_errors = np.array(model_errors)
    oracle_errors = np.array(oracle_errors)

    # AUSE = area between curves (trapezoidal)
    diff = model_errors - oracle_errors
    ause = np.trapz(diff, dx=1.0 / len(diff))

    return float(ause)


def compute_risk_coverage(errors: np.ndarray, uncertainties: np.ndarray,
                          valid_mask: np.ndarray,
                          n_thresholds: int = 100) -> dict:
    """Risk-coverage curve for selective prediction.

    Returns:
        Dict with 'coverages' and 'risks' arrays, plus 'aurc'.
    """
    mask = valid_mask.astype(bool)
    err = errors[mask]
    unc = uncertainties[mask]

    thresholds = np.percentile(unc, np.linspace(0, 100, n_thresholds))
    coverages = []
    risks = []

    for tau in thresholds:
        retained = unc <= tau
        coverage = retained.sum() / len(unc)
        if coverage == 0:
            continue
        risk = err[retained].mean()
        coverages.append(float(coverage))
        risks.append(float(risk))

    coverages = np.array(coverages)
    risks = np.array(risks)

    # AURC via trapezoidal integration
    sort_idx = np.argsort(coverages)
    aurc = float(np.trapz(risks[sort_idx], coverages[sort_idx]))

    return {"coverages": coverages, "risks": risks, "aurc": aurc}
