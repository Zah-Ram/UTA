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
Energy Score, MaxLogit, and DOCTOR reduce to monotone transformations of
the ABSOLUTE logit |z|, i.e. the distance from the decision boundary, and
therefore produce identical AUROC to MSP. (A score monotone in the signed
logit z would not tie with MSP, since MSP is symmetric about z = 0.)
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
    """Binary entropy as uncertainty.

    Note: binary entropy is a strictly decreasing function of |p - 0.5|, so
    this score is rank-identical to unc_msp by construction and will report
    the same AUROC. This is expected, not a bug.
    """
    return _binary_entropy(probs)


# ============================================================================
# Sampling-based methods
# ============================================================================

def unc_mc_dropout(mc_probs):
    """MC Dropout variance (Gal & Ghahramani, ICML 2016).

    The paper's predictive variance is the sample variance of T stochastic
    forward passes plus the inverse model precision; the additive constant
    does not affect ranking, so the sample variance is used directly.

    Caller requirement: the T passes must be stochastic through DROPOUT ONLY.
    Calling model.train() before the MC loop also switches BatchNorm to batch
    statistics (and updates its running estimates), which is not part of the
    method. Keep the model in eval() and re-enable only nn.Dropout modules.

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

    Note: this is a multi-label adaptation, not Deep Ensembles
    (Lakshminarayanan et al., 2017), which averages predictions across
    independently initialised networks. Independent sigmoids admit no shared
    categorical distribution to average, and the exits share one backbone.

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

    Analytical closed form for the last linear layer (paper Eq. 9):
    S(x) = (1 / CT) * U * V, where U is the L1-norm of the feature vector and
    V is the L1 deviation of the predictions from the uniform target. The
    paper's norm ablation finds L1 best on all four datasets and L-infinity
    worst, so L1 is used for both terms. T = 1.

    Note: GradNorm produces ONE score per image, not one per class. Under a
    per-disease evaluation the same vector is scored against every label, so
    it should be reported as an image-level bound rather than a per-class
    detector.

    Args:
        probs_all: np.ndarray (N, C), all class probabilities.
        penultimate_features: np.ndarray (N, D), penultimate layer features.

    Returns:
        np.ndarray (N,), negative gradient norm (higher = more uncertain).
    """
    h_norms = np.sum(np.abs(penultimate_features), axis=1)
    deviations = np.sum(np.abs(probs_all - 0.5), axis=1)
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

    The threshold is the p-th percentile of activations estimated on the ID
    training data (paper's default p = 90). The MSP-style readout below is
    sanctioned by the paper, which reports "Softmax score + ReAct" alongside
    the energy variant.

    Caller requirement: `features` must be the PENULTIMATE representation,
    i.e. the direct input to the final linear layer, since the paper's
    placement ablation finds that layer most effective. For a two-layer head
    Linear(1024, 512) -> ReLU -> Linear(512, C), that is the 512-d activation,
    not the 1024-d backbone output.

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

    Builds the contribution matrix V of shape (C, D) with each row
    v_c = E_x[w_c * h(x)] (paper Eq. 1), keeps the k largest elements of V,
    and computes the sparsified output (M * W) h(x) + b (paper Eq. 2).
    Contributions are signed, and the mask is per-class: this class direction
    is what the paper's "directed" refers to.

    The top-k threshold is taken globally over all C*D entries of V, matching
    the paper's sparsity parameter p = 1 - k / (m*C).

    The MSP-style readout is the paper's acknowledged alternative to the
    energy score (its Appendix F), which it notes is less competitive.

    Args:
        sparsity: Fraction of contributions to mask (default: 0.9).
    """

    def __init__(self, sparsity=0.9):
        self.sparsity = sparsity
        self.weight_mask = None
        self.weights = None
        self.bias = None

    def fit(self, train_features, classifier_weights, classifier_bias=None):
        """Compute the per-class contribution mask.

        Args:
            train_features: np.ndarray (N, D), penultimate features.
            classifier_weights: np.ndarray (C, D), final linear layer weights
                (nn.Linear.weight, i.e. out_features x in_features).
            classifier_bias: np.ndarray (C,), final linear layer bias. If None,
                a zero bias is used; pass the real bias to match the paper.
        """
        W = np.asarray(classifier_weights, dtype=float)
        if W.ndim == 1:
            W = W[None, :]
        self.weights = W

        C = W.shape[0]
        if classifier_bias is None:
            self.bias = np.zeros(C, dtype=float)
        else:
            self.bias = np.asarray(classifier_bias, dtype=float).reshape(C)

        mean_features = train_features.mean(axis=0)          # (D,), signed
        contributions = W * mean_features[None, :]           # (C, D), signed
        threshold = np.percentile(contributions, self.sparsity * 100)
        self.weight_mask = (contributions >= threshold).astype(float)
        return self

    def get_uncertainty(self, model, features, device, class_idx=None):
        """Compute uncertainty with the sparsified weight matrix.

        Args:
            model: Unused; retained for interface compatibility. The
                sparsified logits are computed directly from the masked
                weights, since the mask is per-class and cannot be applied
                by scaling the shared feature vector.
            features: np.ndarray (N, D), penultimate features.
            device: Unused; retained for interface compatibility.
            class_idx: If provided, return uncertainty for this class only.

        Returns:
            np.ndarray, uncertainty scores.
        """
        masked_W = self.weight_mask * self.weights          # (C, D)
        logits = features @ masked_W.T + self.bias[None, :]  # (N, C)
        probs = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        unc = 1 - np.abs(probs - 0.5) * 2
        if class_idx is not None:
            return unc[:, class_idx]
        return unc


class ASH:
    """ASH: Activation Shaping (Djurisic et al., ICLR 2023).

    Prunes activations below a per-sample percentile threshold, then applies
    one of the paper's treatments to the survivors. The threshold is computed
    per-sample ("local"), which the paper's ablation finds always better than
    a global threshold.

    Variants (paper Algorithms 1 and 3):
        'S': prune, then multiply survivors by exp(s1 / s2), where s1 and s2
             are the activation sums before and after pruning. This is the
             paper's headline variant.
        'P': prune only. The paper describes ASH-P as a baseline included to
             highlight the gains of the other treatments, so results from this
             variant should be labelled ASH-P, not ASH.

    Args:
        percentile: Per-sample pruning percentile (default: 90).
        variant: 'S' (default) or 'P'.
    """

    def __init__(self, percentile=90, variant='S'):
        if variant not in ('S', 'P'):
            raise ValueError("variant must be 'S' or 'P'")
        self.percentile = percentile
        self.variant = variant

    def get_uncertainty(self, model, features, device, class_idx=None):
        """Compute uncertainty with shaped activations.

        Args:
            model: Backbone model.
            features: np.ndarray (N, D), post-ReLU activations.
            device: torch device.
            class_idx: If provided, return uncertainty for this class only.

        Returns:
            np.ndarray, uncertainty scores.
        """
        thresholds = np.percentile(
            features, self.percentile, axis=1, keepdims=True
        )
        s1 = features.sum(axis=1, keepdims=True)
        shaped = features.copy()
        shaped[features < thresholds] = 0

        if self.variant == 'S':
            s2 = shaped.sum(axis=1, keepdims=True)
            scale = np.exp(np.divide(
                s1, s2, out=np.zeros_like(s1), where=s2 != 0
            ))
            shaped = shaped * scale

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
