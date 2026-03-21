"""
Unsupervised baseline methods for misclassification detection.

Methods:
    - MSP:           Maximum Softmax Probability (Hendrycks & Gimpel, ICLR 2017)
    - Entropy:       Predictive entropy
    - MC-Dropout:    Monte Carlo Dropout variance (Gal & Ghahramani, ICML 2016)
    - MC-Entropy:    Predictive entropy over MC samples
    - AuxDisagree:   Auxiliary head disagreement
    - GradNorm:      Gradient norm (Huang et al., NeurIPS 2021)
    - ReAct:         Rectified activations (Sun et al., NeurIPS 2021)
    - DICE:          Directed sparsification (Sun & Li, ECCV 2022)
    - ASH:           Activation shaping (Djurisic et al., ICLR 2023)

Note: Under binary sigmoid classification (multi-label medical imaging),
Energy Score, MaxLogit, and DOCTOR reduce to monotonic transformations
of a single logit and produce identical AUROC to MSP.
"""

import numpy as np
import torch
import torch.nn.functional as F


def _binary_entropy(p):
    """Binary entropy H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def _confidence(probs):
    """Confidence score: |2p - 1|."""
    return np.abs(probs - 0.5) * 2


# ============================================================================
# Output-based methods
# ============================================================================

def unc_msp(probs):
    """MSP uncertainty: 1 - confidence (Hendrycks & Gimpel, ICLR 2017)."""
    return 1 - _confidence(probs)


def unc_entropy(probs):
    """Binary entropy as uncertainty."""
    return _binary_entropy(probs)


# ============================================================================
# Sampling-based methods
# ============================================================================

def unc_mc_dropout(mc_probs):
    """MC Dropout variance (Gal & Ghahramani, ICML 2016).

    Args:
        mc_probs: np.ndarray (T, N), T stochastic forward passes.

    Returns:
        np.ndarray (N,), predictive variance.
    """
    return mc_probs.var(axis=0)


def unc_mc_entropy(mc_probs):
    """Predictive entropy over MC Dropout samples.

    Args:
        mc_probs: np.ndarray (T, N), T stochastic forward passes.

    Returns:
        np.ndarray (N,), entropy of mean prediction.
    """
    return _binary_entropy(mc_probs.mean(axis=0))


# ============================================================================
# Multi-exit method
# ============================================================================

def unc_aux_disagreement(main_probs, aux_probs):
    """Auxiliary head disagreement (ensemble-style variance).

    Args:
        main_probs: np.ndarray (N,).
        aux_probs: List of 4 np.ndarrays (N,).

    Returns:
        np.ndarray (N,), variance across all depths.
    """
    all_probs = np.stack(aux_probs + [main_probs], axis=0)
    return all_probs.var(axis=0)


# ============================================================================
# Gradient-based method
# ============================================================================

def unc_gradnorm(probs_all, penultimate_features):
    """GradNorm (Huang et al., NeurIPS 2021).

    Efficient analytical approximation: product of feature norm
    and deviation of predictions from uniform.

    Args:
        probs_all: np.ndarray (N, C), all class probabilities.
        penultimate_features: np.ndarray (N, D), penultimate layer features.

    Returns:
        np.ndarray (N,), negative gradient norm (higher = more uncertain).
    """
    h_norms = np.linalg.norm(penultimate_features, axis=1)
    deviations = np.sqrt(np.sum((probs_all - 0.5) ** 2, axis=1))
    C = probs_all.shape[1]
    gradnorm = h_norms * deviations / C
    return -gradnorm


# ============================================================================
# Activation-based methods
# ============================================================================

class ReAct:
    """ReAct: Rectified Activations (Sun et al., NeurIPS 2021).

    Clips penultimate-layer activations above a threshold, then
    recomputes predictions to obtain uncertainty scores.

    Args:
        percentile: Activation clipping percentile (default: 90).
    """

    def __init__(self, percentile=90):
        self.percentile = percentile
        self.threshold = None

    def fit(self, train_features):
        """Compute activation threshold from training features."""
        self.threshold = np.percentile(train_features.flatten(), self.percentile)
        return self

    def get_uncertainty(self, model, features, device, class_idx=None):
        """Compute uncertainty with clipped activations.

        Args:
            model: Backbone model with forward_from_features method.
            features: np.ndarray (N, D), penultimate features.
            device: torch device.
            class_idx: If provided, return uncertainty for this class only.

        Returns:
            np.ndarray, uncertainty scores.
        """
        clipped = np.clip(features, None, self.threshold)
        model.eval()
        with torch.no_grad():
            logits = model.forward_from_features(
                torch.FloatTensor(clipped).to(device)
            ).cpu().numpy()
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        unc = 1 - np.abs(probs - 0.5) * 2
        if class_idx is not None:
            return unc[:, class_idx]
        return unc


class DICE:
    """DICE: Directed Sparsification (Sun & Li, ECCV 2022).

    Masks weight contributions based on magnitude, then recomputes
    predictions for uncertainty estimation.

    Args:
        sparsity: Fraction of contributions to mask (default: 0.9).
    """

    def __init__(self, sparsity=0.9):
        self.sparsity = sparsity
        self.weight_mask = None

    def fit(self, train_features, classifier_weights):
        """Compute contribution mask from training features and classifier weights.

        Args:
            train_features: np.ndarray (N, D).
            classifier_weights: np.ndarray, last layer weights.
        """
        if len(classifier_weights.shape) > 1:
            weights = classifier_weights.mean(axis=0)
        else:
            weights = classifier_weights
        mean_features = np.abs(train_features).mean(axis=0)
        contributions = mean_features * np.abs(weights)
        threshold = np.percentile(contributions, self.sparsity * 100)
        self.weight_mask = (contributions >= threshold).astype(float)
        return self

    def get_uncertainty(self, model, features, device, class_idx=None):
        """Compute uncertainty with masked features.

        Args:
            model: Backbone model.
            features: np.ndarray (N, D), penultimate features.
            device: torch device.
            class_idx: If provided, return uncertainty for this class only.

        Returns:
            np.ndarray, uncertainty scores.
        """
        masked = features * self.weight_mask
        model.eval()
        with torch.no_grad():
            logits = model.forward_from_features(
                torch.FloatTensor(masked).to(device)
            ).cpu().numpy()
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        unc = 1 - np.abs(probs - 0.5) * 2
        if class_idx is not None:
            return unc[:, class_idx]
        return unc


class ASH:
    """ASH: Activation Shaping (Djurisic et al., ICLR 2023).

    Zeros out activations below a per-sample percentile threshold,
    then recomputes predictions for uncertainty.

    Args:
        percentile: Per-sample activation threshold percentile (default: 90).
    """

    def __init__(self, percentile=90):
        self.percentile = percentile

    def get_uncertainty(self, model, features, device, class_idx=None):
        """Compute uncertainty with shaped activations.

        Args:
            model: Backbone model.
            features: np.ndarray (N, D).
            device: torch device.
            class_idx: If provided, return uncertainty for this class only.

        Returns:
            np.ndarray, uncertainty scores.
        """
        thresholds = np.percentile(
            np.abs(features), self.percentile, axis=1, keepdims=True
        )
        shaped = features.copy()
        shaped[np.abs(features) < thresholds] = 0

        model.eval()
        with torch.no_grad():
            logits = model.forward_from_features(
                torch.FloatTensor(shaped).to(device)
            ).cpu().numpy()
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        unc = 1 - np.abs(probs - 0.5) * 2
        if class_idx is not None:
            return unc[:, class_idx]
        return unc
