#!/usr/bin/env python3
"""
Reproduce NIH ChestX-ray14 misclassification detection results.

Usage:
    python evaluate_nih.py \
        --data_root ./data/nih \
        --model_path ./pretrained/nih_densenet121.pth \
        --output_dir ./results/nih
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from uta.models import TrajectoryDenseNet121
from uta.descriptors import compute_trajectory_descriptors, descriptors_to_matrix
from uta.meta_learner import PerClassMetaLearner
from uta.metrics import evaluate_all_metrics

from baselines.unsupervised import (
    unc_msp, unc_entropy, unc_mc_dropout, unc_mc_entropy,
    unc_aux_disagreement, unc_gradnorm, ReAct, DICE, ASH
)
from baselines.supervised import ConfidNet, train_confidnet, learned_baseline


# ============================================================================
# Configuration
# ============================================================================

DISEASE_NAMES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]
NUM_CLASSES = 14
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]


# ============================================================================
# Dataset
# ============================================================================

class NIHChestXrayDataset(Dataset):
    """NIH ChestX-ray14 test dataset."""

    _image_cache = {}

    @classmethod
    def build_image_cache(cls, root):
        root_str = str(root)
        if root_str in cls._image_cache:
            return cls._image_cache[root_str]
        paths = {}
        root_path = Path(root)
        for pattern in ['images_*/images/*.png', 'images_*/*.png']:
            for p in root_path.glob(pattern):
                paths[p.name] = str(p)
        cls._image_cache[root_str] = paths
        return paths

    def __init__(self, image_names, data_entry_df, root, transform):
        self.transform = transform
        self.data_entry = data_entry_df.set_index('Image Index')
        self.image_paths = self.build_image_cache(root)
        available = set(self.image_paths.keys())
        self.image_names = [n for n in image_names if n in available]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        path = self.image_paths[name]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), 'gray')
        if self.transform:
            image = self.transform(image)
        finding_labels = self.data_entry.loc[name, 'Finding Labels']
        labels = np.zeros(NUM_CLASSES, dtype=np.float32)
        if pd.notna(finding_labels) and finding_labels != 'No Finding':
            for finding in finding_labels.split('|'):
                finding = finding.strip()
                if finding in DISEASE_NAMES:
                    labels[DISEASE_NAMES.index(finding)] = 1.0
        return image, labels


def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


# ============================================================================
# Inference
# ============================================================================

def run_inference(model, dataloader, device, mc_samples=30):
    """Run standard + MC Dropout inference."""
    data = {
        'logits': [], 'probs': [], 'labels': [],
        'features': [],
        'aux_logits': [[] for _ in range(4)],
        'aux_probs': [[] for _ in range(4)],
    }

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference"):
            images = images.to(device)
            logits, aux_outs, feats = model.forward_with_features(images)
            data['logits'].append(logits.cpu().numpy())
            data['probs'].append(torch.sigmoid(logits).cpu().numpy())
            data['labels'].append(labels.numpy())
            data['features'].append(feats.cpu().numpy())
            for i, aux in enumerate(aux_outs):
                data['aux_logits'][i].append(aux.cpu().numpy())
                data['aux_probs'][i].append(torch.sigmoid(aux).cpu().numpy())

    # MC Dropout
    mc_predictions = []
    if mc_samples > 0:
        print(f"Running MC Dropout ({mc_samples} samples)...")
        model.train()
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="MC Dropout"):
                images = images.to(device)
                batch_preds = []
                for _ in range(mc_samples):
                    feats = model.forward_with_features(images)[2]
                    feats = F.dropout(feats, p=0.3, training=True)
                    logits = model.forward_from_features(feats)
                    batch_preds.append(torch.sigmoid(logits).cpu().numpy())
                mc_predictions.append(np.stack(batch_preds, axis=0))
        mc_predictions = np.concatenate(mc_predictions, axis=1)

    results = {
        'logits': np.concatenate(data['logits']),
        'probs': np.concatenate(data['probs']),
        'labels': np.concatenate(data['labels']),
        'features': np.concatenate(data['features']),
        'aux_logits': [np.concatenate(al) for al in data['aux_logits']],
        'aux_probs': [np.concatenate(ap) for ap in data['aux_probs']],
        'mc_probs': mc_predictions if mc_samples > 0 else None,
    }
    return results


# ============================================================================
# Per-disease cross-validated evaluation
# ============================================================================

def evaluate_disease(model, results, disease_idx, device, n_folds=5, seed=42):
    """Run cross-validated evaluation for a single disease."""

    probs = results['probs'][:, disease_idx]
    logits = results['logits'][:, disease_idx]
    labels = results['labels'][:, disease_idx].astype(int)
    features = results['features']
    aux_probs = [ap[:, disease_idx] for ap in results['aux_probs']]
    aux_logits = [al[:, disease_idx] for al in results['aux_logits']]
    mc_probs = results['mc_probs'][:, :, disease_idx] if results['mc_probs'] is not None else None

    preds = (probs > 0.5).astype(int)
    errors = (preds != labels).astype(int)

    if errors.sum() < 50:
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    cv_results = defaultdict(lambda: defaultdict(list))

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(errors)), errors)):
        train_errors = errors[train_idx]
        test_errors = errors[test_idx]

        if train_errors.sum() < 10 or test_errors.sum() < 5:
            continue

        test_preds = preds[test_idx]
        test_labels = labels[test_idx]

        def store(method, unc):
            unc = np.nan_to_num(unc, nan=0.5)
            res = evaluate_all_metrics(unc, test_preds, test_labels)
            for m, v in res.items():
                if v is not None and not isinstance(v, dict):
                    cv_results[method][m].append(v)

        # Unsupervised baselines
        store('MSP', unc_msp(probs[test_idx]))
        store('Entropy', unc_entropy(probs[test_idx]))

        if mc_probs is not None:
            store('MC_Dropout', unc_mc_dropout(mc_probs[:, test_idx]))
            store('MC_Entropy', unc_mc_entropy(mc_probs[:, test_idx]))

        store('AuxDisagree', unc_aux_disagreement(
            probs[test_idx], [ap[test_idx] for ap in aux_probs]
        ))

        store('GradNorm', unc_gradnorm(
            results['probs'][test_idx], features[test_idx]
        ))

        # ReAct
        try:
            react = ReAct(percentile=90)
            react.fit(features[train_idx])
            store('ReAct', react.get_uncertainty(
                model, features[test_idx], device, class_idx=disease_idx
            ))
        except Exception:
            pass

        # DICE
        try:
            weights = model.classifier[-1].weight.data.cpu().numpy()
            dice = DICE(sparsity=0.9)
            dice.fit(features[train_idx], weights)
            store('DICE', dice.get_uncertainty(
                model, features[test_idx], device, class_idx=disease_idx
            ))
        except Exception:
            pass

        # ASH
        try:
            ash = ASH(percentile=90)
            store('ASH', ash.get_uncertainty(
                model, features[test_idx], device, class_idx=disease_idx
            ))
        except Exception:
            pass

        # ConfidNet (supervised)
        try:
            tcp = probs[train_idx] * labels[train_idx] + \
                  (1 - probs[train_idx]) * (1 - labels[train_idx])
            confidnet = train_confidnet(features[train_idx], tcp, device)
            confidnet.eval()
            with torch.no_grad():
                scores = confidnet(
                    torch.FloatTensor(features[test_idx]).to(device)
                ).cpu().numpy().flatten()
            store('ConfidNet', 1 - scores)
        except Exception:
            pass

        # Learned baseline (supervised)
        try:
            mc_var_train = mc_probs[:, train_idx].var(axis=0) if mc_probs is not None else np.zeros(len(train_idx))
            mc_var_test = mc_probs[:, test_idx].var(axis=0) if mc_probs is not None else np.zeros(len(test_idx))
            store('Learned', learned_baseline(
                probs[train_idx], [ap[train_idx] for ap in aux_probs],
                mc_var_train, train_errors,
                probs[test_idx], [ap[test_idx] for ap in aux_probs],
                mc_var_test, seed=seed
            ))
        except Exception:
            pass

        # UTA (Ours)
        try:
            train_desc = compute_trajectory_descriptors(
                probs[train_idx], [ap[train_idx] for ap in aux_probs],
                logits[train_idx], [al[train_idx] for al in aux_logits]
            )
            test_desc = compute_trajectory_descriptors(
                probs[test_idx], [ap[test_idx] for ap in aux_probs],
                logits[test_idx], [al[test_idx] for al in aux_logits]
            )
            X_train, _ = descriptors_to_matrix(train_desc)
            X_test, _ = descriptors_to_matrix(test_desc)

            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(
                max_iter=500, class_weight='balanced', random_state=seed
            )
            lr.fit(X_train, train_errors)
            store('UTA', lr.predict_proba(X_test)[:, 1])
        except Exception:
            pass

    # Aggregate across folds
    aggregated = {}
    for method, metrics in cv_results.items():
        aggregated[method] = {}
        for metric, values in metrics.items():
            if values:
                aggregated[method][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                }
    return aggregated


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate UTA on NIH ChestX-ray14')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./results/nih')
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--mc_samples', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("NIH ChestX-ray14 — UTA Evaluation")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Folds: {args.n_folds}, MC samples: {args.mc_samples}")

    # Load data
    data_entry = pd.read_csv(os.path.join(args.data_root, 'Data_Entry_2017.csv'))
    with open(os.path.join(args.data_root, 'test_list.txt')) as f:
        test_names = [l.strip() for l in f if l.strip()]

    dataset = NIHChestXrayDataset(test_names, data_entry, args.data_root, get_transforms())
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Load model
    model = TrajectoryDenseNet121(num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print(f"Model loaded: {args.model_path}")

    # Inference (once)
    torch.manual_seed(args.seeds[0])
    np.random.seed(args.seeds[0])
    results = run_inference(model, dataloader, device, mc_samples=args.mc_samples)
    print(f"Inference complete: {len(results['probs'])} samples")

    # Multi-seed evaluation
    all_seed_results = {}

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)
        seed_data = {}

        for d_idx, disease in enumerate(DISEASE_NAMES):
            result = evaluate_disease(
                model, results, d_idx, device,
                n_folds=args.n_folds, seed=seed
            )
            if result:
                seed_data[disease] = result
                uta_auroc = result.get('UTA', {}).get('auroc', {}).get('mean', 'N/A')
                print(f"  {disease}: UTA AUROC = {uta_auroc}")

        all_seed_results[str(seed)] = seed_data

    # Save results
    output_path = os.path.join(args.output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(all_seed_results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY (mean across seeds)")
    print("=" * 70)

    method_aurocs = defaultdict(list)
    for seed_data in all_seed_results.values():
        seed_means = defaultdict(list)
        for disease_data in seed_data.values():
            for method, metrics in disease_data.items():
                if 'auroc' in metrics:
                    seed_means[method].append(metrics['auroc']['mean'])
        for method, values in seed_means.items():
            method_aurocs[method].append(np.mean(values))

    for method in sorted(method_aurocs, key=lambda m: np.mean(method_aurocs[m]), reverse=True):
        vals = method_aurocs[method]
        print(f"  {method:<15}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")


if __name__ == '__main__':
    main()
