"""
Evaluation metrics for misclassification detection.

Four complementary metrics are used:
    - AUROC:  Area Under the ROC Curve (threshold-independent)
    - AUPR:   Area Under the Precision-Recall Curve (sensitive to class imbalance)
    - FPR@95: False Positive Rate at 95% True Positive Rate (threshold-dependent)
    - AURC:   Area Under the Risk-Coverage Curve (selective prediction)
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def compute_auroc(errors, uncertainty):
    """Area Under the ROC Curve.

    Args:
        errors: np.ndarray (N,), binary (1=misclassified, 0=correct).
        uncertainty: np.ndarray (N,), uncertainty scores (higher = more uncertain).

    Returns:
        float or None if computation fails.
    """
    if len(np.unique(errors)) < 2:
        return None
    try:
        return float(roc_auc_score(errors, uncertainty))
    except Exception:
        return None


def compute_aupr(errors, uncertainty):
    """Area Under the Precision-Recall Curve.

    Args:
        errors: np.ndarray (N,), binary error labels.
        uncertainty: np.ndarray (N,), uncertainty scores.

    Returns:
        float or None.
    """
    if len(np.unique(errors)) < 2:
        return None
    try:
        return float(average_precision_score(errors, uncertainty))
    except Exception:
        return None


def compute_fpr_at_tpr(errors, uncertainty, target_tpr=0.95):
    """False Positive Rate at a target True Positive Rate.

    Args:
        errors: np.ndarray (N,), binary error labels.
        uncertainty: np.ndarray (N,), uncertainty scores.
        target_tpr: float, target TPR (default 0.95).

    Returns:
        float or None.
    """
    if len(np.unique(errors)) < 2:
        return None
    try:
        fpr, tpr, _ = roc_curve(errors, uncertainty)
        idx = np.where(tpr >= target_tpr)[0]
        if len(idx) == 0:
            return 1.0
        return float(fpr[idx[0]])
    except Exception:
        return None


def compute_aurc(errors, uncertainty):
    """Area Under the Risk-Coverage Curve.

    Measures selective prediction performance: samples are sorted by
    ascending uncertainty, and the cumulative error rate is integrated
    over the coverage fraction.

    Args:
        errors: np.ndarray (N,), binary error labels.
        uncertainty: np.ndarray (N,), uncertainty scores.

    Returns:
        float or None.
    """
    try:
        n = len(errors)
        sorted_idx = np.argsort(uncertainty)
        sorted_errors = errors[sorted_idx]
        cumsum = np.cumsum(sorted_errors)
        counts = np.arange(1, n + 1)
        risks = cumsum / counts
        coverages = counts / n
        return float(np.trapz(risks, coverages))
    except Exception:
        return None


def evaluate_all_metrics(uncertainty, predictions, labels):
    """Compute all four metrics for a given uncertainty score.

    Args:
        uncertainty: np.ndarray (N,), uncertainty scores.
        predictions: np.ndarray (N,), predicted labels.
        labels: np.ndarray (N,), ground truth labels.

    Returns:
        Dict with keys: auroc, aupr, fpr95, aurc, n_errors, error_rate.
    """
    errors = (predictions != labels).astype(int)
    n_errors = int(errors.sum())

    results = {
        'n_total': len(errors),
        'n_errors': n_errors,
        'error_rate': float(n_errors / len(errors)) if len(errors) > 0 else 0.0,
    }

    if n_errors < 10 or (len(errors) - n_errors) < 10:
        results.update({'auroc': None, 'aupr': None, 'fpr95': None, 'aurc': None})
        return results

    results['auroc'] = compute_auroc(errors, uncertainty)
    results['aupr'] = compute_aupr(errors, uncertainty)
    results['fpr95'] = compute_fpr_at_tpr(errors, uncertainty)
    results['aurc'] = compute_aurc(errors, uncertainty)

    return results
