# Uncertainty Trajectory Analysis (UTA)

**Uncertainty Trajectory Analysis for Misclassification Detection in Deep Neural Networks**

> Accepted at *Knowledge-Based Systems* (Elsevier), 2025.

## Overview

UTA is a post-hoc misclassification detection framework that monitors the evolution of predictions across hierarchical network depths rather than relying solely on terminal outputs. By equipping a backbone network with lightweight auxiliary classifiers, UTA extracts the prediction trajectory (probability, logit, and entropy) across multiple intermediate depths in a single forward pass, then encodes these dynamics into 23 interpretable descriptors for error detection.

![UTA Framework](assets/uta_framework.png)

## Key Results

| Dataset | Backbone | AUROC | AUPR | FPR@95 | AURC |
|---|---|---|---|---|---|
| NIH ChestX-ray14 | DenseNet-121 | **0.927** | **0.768** | **0.306** | **0.062** |
| CheXpert | DenseNet-121 | **0.851** | **0.490** | **0.582** | **0.037** |
| CIFAR-100 | ResNet-50 | **0.889** | **0.646** | 0.400 | **0.045** |

All results averaged over 5 random seeds × 5-fold cross-validation.

## Repository Structure

```
UTA/
├── README.md
├── requirements.txt
├── pretrained/                # Pretrained model weights (download below)
│   ├── nih_densenet121.pth
│   ├── chexpert_densenet121.pth
│   └── cifar100_resnet50.pth
├── uta/
│   ├── __init__.py
│   ├── models.py              # Backbone architectures with auxiliary classifiers
│   ├── descriptors.py         # 23 trajectory descriptors (6 families)
│   ├── meta_learner.py        # Per-class logistic regression meta-learner
│   └── metrics.py             # AUROC, AUPR, FPR@95, AURC evaluation
├── baselines/
│   ├── __init__.py
│   ├── unsupervised.py        # MSP, Entropy, MC-Dropout, ReAct, DICE, ASH, GradNorm
│   └── supervised.py          # ConfidNet, Learned baseline
├── evaluate_nih.py            # Reproduce NIH ChestX-ray14 results
├── evaluate_chexpert.py       # Reproduce CheXpert results
└── evaluate_cifar100.py       # Reproduce CIFAR-100 results
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/UTA.git
cd UTA
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- torchvision >= 0.13
- scikit-learn >= 1.0
- numpy, pandas, scipy, tqdm, Pillow

## Dataset Preparation

### NIH ChestX-ray14

1. Download from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
2. Extract all `images_*.tar.gz` into a single directory
3. Place `Data_Entry_2017.csv` and `test_list.txt` in the dataset root

```
data/nih/
├── images_001/images/*.png
├── images_002/images/*.png
├── ...
├── Data_Entry_2017.csv
└── test_list.txt
```

### CheXpert

1. Download CheXpert-v1.0-small from [Stanford ML Group](https://stanfordmlgroup.github.io/competitions/chexpert/)
2. Extract into the dataset directory

```
data/chexpert/
├── train.csv
├── train/
│   └── patient*/study*/*.jpg
└── valid/
```

### CIFAR-100

Automatically downloaded by torchvision:
```python
torchvision.datasets.CIFAR100(root='./data', download=True)
```

## Pretrained Models

Download pretrained weights from [Google Drive / Zenodo link]:

| Model | Dataset | Architecture | Val AUC | Download |
|---|---|---|---|---|
| `nih_densenet121.pth` | NIH ChestX-ray14 | DenseNet-121 + 4 aux heads | 0.843 | [link] |
| `chexpert_densenet121.pth` | CheXpert | DenseNet-121 + 4 aux heads | 0.878 | [link] |
| `cifar100_resnet50.pth` | CIFAR-100 | ResNet-50 + 4 aux heads | 0.807 | [link] |

Place downloaded weights in the `pretrained/` directory.

## Reproducing Results

### NIH ChestX-ray14

```bash
python evaluate_nih.py \
    --data_root ./data/nih \
    --model_path ./pretrained/nih_densenet121.pth \
    --output_dir ./results/nih \
    --seeds 42 123 456 789 1024 \
    --n_folds 5 \
    --mc_samples 30
```

### CheXpert

```bash
python evaluate_chexpert.py \
    --data_root ./data/chexpert \
    --model_path ./pretrained/chexpert_densenet121.pth \
    --output_dir ./results/chexpert \
    --seeds 42 123 456 789 1024 \
    --n_folds 5 \
    --mc_samples 30
```

### CIFAR-100

```bash
python evaluate_cifar100.py \
    --data_root ./data/cifar100 \
    --model_path ./pretrained/cifar100_resnet50.pth \
    --output_dir ./results/cifar100 \
    --seeds 42 123 456 789 1024 \
    --n_folds 5 \
    --mc_samples 30
```

## UTA Descriptor Families

The 23 trajectory descriptors are organized into six interpretable families:

| Family | Descriptors | Description |
|---|---|---|
| **Uncertainty Evolution** | U1, U2, U3, U4, entropy, confidence | Per-depth uncertainty values |
| **Trajectory Shape** | traj_slope, late_spike, U_range | Overall trajectory geometry |
| **Cross-Depth Agreement** | agreement, early_disagree, late_disagree | Consistency of predictions across depths |
| **Logit & Probability Statistics** | logit_mean, logit_range, prob_mean, prob_std | Distributional properties |
| **Velocity & Acceleration** | velocity_mean, velocity_std, accel_mean, accel_std | Rate of change dynamics |
| **Depth-Specific Confidence** | early_conf, mid_conf, late_conf | Confidence at specific network stages |

## Method Comparison

UTA is compared against 11 baselines spanning three categories:

**Unsupervised (9):** MSP, Entropy, MC-Dropout, MC-Entropy, AuxDisagree, GradNorm, ReAct, DICE, ASH

**Supervised (2):** ConfidNet, Learned (logistic regression on simple uncertainty features)

Note: Under binary sigmoid classification (NIH, CheXpert), Energy Score, MaxLogit, and DOCTOR reduce to monotonic transformations of a single logit, producing identical AUROC to MSP.

## Citation

```bibtex
@article{rahman2025uta,
  title={Uncertainty Trajectory Analysis for Misclassification Detection in Deep Neural Networks},
  author={Rahman, Zahid Ur and Kim, Jin-Young},
  journal={Knowledge-Based Systems},
  year={2025},
  publisher={Elsevier}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work was supported by the Korea Health Industry Development Institute (KHIDI), the Institute of Information & Communications Technology Planning & Evaluation (IITP), and the Ministry of Science and ICT (MSIT), Republic of Korea.
