"""
Supervised baseline methods for misclassification detection.

Methods:
    - ConfidNet: Auxiliary MLP predicting True Class Probability
                 (Corbiere et al., NeurIPS 2019)
    - Learned:   Logistic regression on simple uncertainty features
                 (fair supervised baseline using same error labels as UTA)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from .unsupervised import unc_msp, unc_entropy, unc_aux_disagreement


class ConfidNet(nn.Module):
    """ConfidNet: Confidence estimation network (Corbiere et al., NeurIPS 2019).

    A succession of dense layers with a final sigmoid activation, built on the
    penultimate features of the frozen base classifier, trained to regress the
    True Class Probability (TCP).

    Args:
        input_dim: Dimensionality of penultimate features (e.g., 1024 for DenseNet-121).
        hidden_dims: List of hidden layer dimensions. Default gives 5 dense
            layers in total, following the paper's architecture.
    """

    def __init__(self, input_dim, hidden_dims=None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [400, 400, 400, 400]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def compute_tcp_targets(probs, labels):
    """True Class Probability targets (Corbiere et al., Eq. 2).

    TCP is the probability the model assigns to the *true* class. Under
    independent sigmoids, the two-class distribution for a given label is
    [1 - p, p], so the true-class probability is p when the label is positive
    and 1 - p when it is negative.

    Note this is NOT binary correctness: a confidently wrong prediction gets a
    TCP near 0, a borderline correct one gets a TCP near 0.5. That graded
    target is the paper's contribution over a correct/incorrect target.

    Args:
        probs: np.ndarray (N,), predicted positive-class probabilities.
        labels: np.ndarray (N,), ground-truth binary labels.

    Returns:
        np.ndarray (N,), TCP targets in [0, 1].
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return np.where(labels == 1, probs, 1.0 - probs)


def train_confidnet(features, tcp_targets, device, epochs=30, patience=5,
                    batch_size=128):
    """Train ConfidNet on penultimate features.

    Trained with the l2 loss of Corbiere et al. (Eq. 4), regressing the TCP
    target. The paper reports that a binary cross-entropy target performs
    worse, so BCE is a different (weaker) method, not this one.

    Args:
        features: np.ndarray (N, D), penultimate features.
        tcp_targets: np.ndarray (N,), True Class Probability targets
            (use compute_tcp_targets).
        device: torch device.
        epochs: Maximum training epochs.
        patience: Early stopping patience.
        batch_size: Mini-batch size for training.

    Returns:
        Trained ConfidNet model.
    """
    n_samples = len(features)
    val_split = int(0.8 * n_samples)
    idx = np.random.permutation(n_samples)
    train_idx, val_idx = idx[:val_split], idx[val_split:]

    X_train = torch.FloatTensor(features[train_idx]).to(device)
    y_train = torch.FloatTensor(tcp_targets[train_idx]).unsqueeze(1).to(device)
    X_val = torch.FloatTensor(features[val_idx]).to(device)
    y_val = torch.FloatTensor(tcp_targets[val_idx]).unsqueeze(1).to(device)

    model = ConfidNet(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    criterion = nn.MSELoss()

    n_train = len(train_idx)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            batch = perm[start:start + batch_size]
            optimizer.zero_grad()
            pred = model(X_train[batch])
            loss = criterion(pred, y_train[batch])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def learned_baseline(train_probs, train_aux_probs, train_mc_var, train_errors,
                     test_probs, test_aux_probs, test_mc_var, seed=42):
    """Learned baseline: logistic regression on simple uncertainty features.

    Aggregates MSP, Entropy, MC-Dropout variance, and auxiliary disagreement
    into a logistic regression model. Uses the same error labels as UTA
    for fair comparison.

    Args:
        train_probs: np.ndarray (N_train,), training probabilities.
        train_aux_probs: List of 4 np.ndarrays (N_train,).
        train_mc_var: np.ndarray (N_train,), MC-Dropout variance.
        train_errors: np.ndarray (N_train,), binary error labels.
        test_probs: np.ndarray (N_test,), test probabilities.
        test_aux_probs: List of 4 np.ndarrays (N_test,).
        test_mc_var: np.ndarray (N_test,), MC-Dropout variance.
        seed: Random seed.

    Returns:
        np.ndarray (N_test,), predicted error probabilities.
    """
    X_train = np.column_stack([
        unc_msp(train_probs),
        unc_entropy(train_probs),
        train_mc_var,
        unc_aux_disagreement(train_probs, train_aux_probs),
    ])
    X_test = np.column_stack([
        unc_msp(test_probs),
        unc_entropy(test_probs),
        test_mc_var,
        unc_aux_disagreement(test_probs, test_aux_probs),
    ])

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    lr = LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=seed
    )
    lr.fit(X_train, train_errors)

    return lr.predict_proba(X_test)[:, 1]
