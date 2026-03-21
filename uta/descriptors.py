"""
Trajectory Descriptors for Uncertainty Trajectory Analysis (UTA).

Encodes the nonlinear dynamics of prediction trajectories across network depth
into 23 interpretable descriptors organized into six conceptual families:

    1. Uncertainty Evolution (7):  confidence, entropy, U1, U2, U3, U4, traj_slope
    2. Trajectory Shape (2):       late_spike, U_range
    3. Cross-Depth Agreement (3):  agreement, early_disagree, late_disagree
    4. Logit & Probability (4):    logit_mean, logit_range, prob_mean, prob_std
    5. Velocity & Acceleration (4): velocity_mean, velocity_std, accel_mean, accel_std
    6. Depth-Specific Confidence (3): early_conf, mid_conf, late_conf

All descriptors operate on per-class prediction trajectories extracted from
auxiliary classifiers at four intermediate depths plus the main classifier.
"""

import numpy as np
from collections import OrderedDict


def _binary_entropy(p):
    """Compute binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return -p * np.log(p) - (1 - p) * np.log(1 - p)


def _confidence(probs):
    """Compute confidence as distance from decision boundary: |2p - 1|."""
    return np.abs(probs - 0.5) * 2


def compute_trajectory_descriptors(main_probs, aux_probs, main_logits, aux_logits):
    """Compute 23 trajectory descriptors for a single disease class.

    Args:
        main_probs:  np.ndarray of shape (N,), sigmoid probabilities from main classifier.
        aux_probs:   List of 4 np.ndarrays, each shape (N,), from auxiliary classifiers.
        main_logits: np.ndarray of shape (N,), raw logits from main classifier.
        aux_logits:  List of 4 np.ndarrays, each shape (N,), from auxiliary classifiers.

    Returns:
        OrderedDict mapping descriptor names to np.ndarrays of shape (N,).
        The 23 descriptors are returned in a fixed order for reproducibility.
    """
    # Sanitize inputs
    main_probs = np.nan_to_num(main_probs, nan=0.5, posinf=1.0, neginf=0.0)
    main_logits = np.nan_to_num(main_logits, nan=0.0, posinf=10.0, neginf=-10.0)
    aux_probs = [np.nan_to_num(ap, nan=0.5, posinf=1.0, neginf=0.0) for ap in aux_probs]
    aux_logits = [np.nan_to_num(al, nan=0.0, posinf=10.0, neginf=-10.0) for al in aux_logits]

    # Pre-compute per-depth uncertainties
    U_main = _binary_entropy(main_probs)
    U_aux = [_binary_entropy(ap) for ap in aux_probs]

    descriptors = OrderedDict()

    # =========================================================================
    # Family 1: Uncertainty Evolution (7 descriptors)
    # =========================================================================
    descriptors['confidence'] = _confidence(main_probs)
    descriptors['entropy'] = U_main
    descriptors['U1'] = U_aux[0]
    descriptors['U2'] = U_aux[1]
    descriptors['U3'] = U_aux[2]
    descriptors['U4'] = U_aux[3]
    descriptors['traj_slope'] = U_aux[0] - U_main  # Overall uncertainty reduction

    # =========================================================================
    # Family 2: Trajectory Shape (2 descriptors)
    # =========================================================================
    descriptors['late_spike'] = (U_main > U_aux[3]).astype(float)

    all_U = np.stack(U_aux + [U_main], axis=0)
    descriptors['U_range'] = all_U.max(axis=0) - all_U.min(axis=0)

    # =========================================================================
    # Family 3: Cross-Depth Agreement (3 descriptors)
    # =========================================================================
    main_pred = (main_probs > 0.5).astype(float)
    aux_preds = [(ap > 0.5).astype(float) for ap in aux_probs]

    descriptors['agreement'] = sum(
        (ap == main_pred).astype(float) for ap in aux_preds
    ) / len(aux_preds)
    descriptors['early_disagree'] = (aux_preds[0] != main_pred).astype(float)
    descriptors['late_disagree'] = (aux_preds[3] != main_pred).astype(float)

    # =========================================================================
    # Family 4: Logit & Probability Statistics (4 descriptors)
    # =========================================================================
    all_logits = np.stack(aux_logits + [main_logits], axis=0)
    descriptors['logit_mean'] = all_logits.mean(axis=0)
    descriptors['logit_range'] = all_logits.max(axis=0) - all_logits.min(axis=0)

    all_probs = np.stack(aux_probs + [main_probs], axis=0)
    descriptors['prob_mean'] = all_probs.mean(axis=0)
    descriptors['prob_std'] = all_probs.std(axis=0)

    # =========================================================================
    # Family 5: Velocity & Acceleration Dynamics (4 descriptors)
    # =========================================================================
    velocity = np.diff(all_probs, axis=0)  # Shape: (4, N)
    descriptors['velocity_mean'] = velocity.mean(axis=0)
    descriptors['velocity_std'] = velocity.std(axis=0)

    accel = np.diff(velocity, axis=0)  # Shape: (3, N)
    descriptors['accel_mean'] = accel.mean(axis=0)
    descriptors['accel_std'] = accel.std(axis=0)

    # =========================================================================
    # Family 6: Depth-Specific Confidence Progression (3 descriptors)
    # =========================================================================
    descriptors['early_conf'] = _confidence(aux_probs[0])
    descriptors['mid_conf'] = _confidence(aux_probs[1])
    descriptors['late_conf'] = _confidence(aux_probs[3])

    # Final sanitization
    for key in descriptors:
        descriptors[key] = np.nan_to_num(
            descriptors[key], nan=0.0, posinf=1e6, neginf=-1e6
        )

    assert len(descriptors) == 23, f"Expected 23 descriptors, got {len(descriptors)}"

    return descriptors


def descriptors_to_matrix(descriptors):
    """Convert descriptor dictionary to feature matrix.

    Args:
        descriptors: OrderedDict from compute_trajectory_descriptors.

    Returns:
        X: np.ndarray of shape (N, 23), feature matrix for meta-learner.
        names: List of 23 descriptor names.
    """
    names = list(descriptors.keys())
    X = np.column_stack([descriptors[k] for k in names])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, names


DESCRIPTOR_FAMILIES = {
    'Uncertainty Evolution': [
        'confidence', 'entropy', 'U1', 'U2', 'U3', 'U4', 'traj_slope'
    ],
    'Trajectory Shape': [
        'late_spike', 'U_range'
    ],
    'Cross-Depth Agreement': [
        'agreement', 'early_disagree', 'late_disagree'
    ],
    'Logit & Probability Statistics': [
        'logit_mean', 'logit_range', 'prob_mean', 'prob_std'
    ],
    'Velocity & Acceleration': [
        'velocity_mean', 'velocity_std', 'accel_mean', 'accel_std'
    ],
    'Depth-Specific Confidence': [
        'early_conf', 'mid_conf', 'late_conf'
    ],
}
