"""
Backbone architectures with auxiliary classifiers for Uncertainty Trajectory Analysis.

DenseNet-121 is used for medical imaging (NIH ChestX-ray14, CheXpert).
ResNet-50 is used for natural image classification (CIFAR-100).

Both architectures attach lightweight auxiliary classifiers at four intermediate
depths, producing prediction trajectories across the network hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class TrajectoryDenseNet121(nn.Module):
    """DenseNet-121 with auxiliary classifiers at each dense block.

    Architecture:
        - Backbone: DenseNet-121 pretrained on ImageNet
        - Auxiliary classifiers: 4 lightweight heads after each dense block
        - Main classifier: 2-layer MLP on final features

    Depths and channels:
        - Depth 1 (after denseblock1): 256 channels
        - Depth 2 (after denseblock2): 512 channels
        - Depth 3 (after denseblock3): 1024 channels
        - Depth 4 (after denseblock4): 1024 channels
        - Depth 5 (main classifier):   1024 channels
    """

    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()

        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)
        self.features = densenet.features

        self.aux_channels = [256, 512, 1024, 1024]
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(ch, num_classes)
            ) for ch in self.aux_channels
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, return_trajectory=False):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            return_trajectory: If True, return (main_logits, aux_logits_list).
                             If False, return main_logits only.

        Returns:
            main_logits: Shape (B, num_classes).
            aux_logits_list (optional): List of 4 tensors, each (B, num_classes).
        """
        aux_outputs = []

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        aux_outputs.append(self.aux_classifiers[0](x))
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        aux_outputs.append(self.aux_classifiers[1](x))
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        aux_outputs.append(self.aux_classifiers[2](x))
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)
        aux_outputs.append(self.aux_classifiers[3](x))

        x = self.features.norm5(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        main_logits = self.classifier(features)

        if return_trajectory:
            return main_logits, aux_outputs
        return main_logits

    def forward_with_features(self, x):
        """Forward pass returning logits, auxiliary outputs, and penultimate features.

        Used by baseline methods (ReAct, DICE, ASH, ConfidNet, GradNorm) that
        require access to internal representations.

        Returns:
            main_logits: Shape (B, num_classes).
            aux_logits_list: List of 4 tensors.
            features: Penultimate features, shape (B, 1024).
        """
        aux_outputs = []

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)

        x = self.features.denseblock1(x)
        aux_outputs.append(self.aux_classifiers[0](x))
        x = self.features.transition1(x)

        x = self.features.denseblock2(x)
        aux_outputs.append(self.aux_classifiers[1](x))
        x = self.features.transition2(x)

        x = self.features.denseblock3(x)
        aux_outputs.append(self.aux_classifiers[2](x))
        x = self.features.transition3(x)

        x = self.features.denseblock4(x)
        aux_outputs.append(self.aux_classifiers[3](x))

        x = self.features.norm5(x)
        x = F.relu(x, inplace=True)
        x = self.avgpool(x)
        features = x.view(x.size(0), -1)
        main_logits = self.classifier(features)

        return main_logits, aux_outputs, features

    def forward_from_features(self, features):
        """Compute logits from pre-extracted features. Used by ReAct, DICE, ASH."""
        return self.classifier(features)


class TrajectoryResNet50(nn.Module):
    """ResNet-50 with auxiliary classifiers at each residual stage.

    Architecture:
        - Backbone: ResNet-50 pretrained on ImageNet
        - Auxiliary classifiers: 4 lightweight heads after each residual stage
        - Main classifier: single linear layer

    Depths and channels:
        - Depth 1 (after layer1): 256 channels
        - Depth 2 (after layer2): 512 channels
        - Depth 3 (after layer3): 1024 channels
        - Depth 4 (after layer4): 2048 channels
        - Depth 5 (main classifier): 2048 channels
    """

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()

        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.aux_channels = [256, 512, 1024, 2048]
        self.aux_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(ch, num_classes)
            ) for ch in self.aux_channels
        ])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x, return_trajectory=False):
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).
            return_trajectory: If True, return (main_logits, aux_logits_list).

        Returns:
            main_logits: Shape (B, num_classes).
            aux_logits_list (optional): List of 4 tensors.
        """
        aux_outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        aux_outputs.append(self.aux_classifiers[0](x))

        x = self.layer2(x)
        aux_outputs.append(self.aux_classifiers[1](x))

        x = self.layer3(x)
        aux_outputs.append(self.aux_classifiers[2](x))

        x = self.layer4(x)
        aux_outputs.append(self.aux_classifiers[3](x))

        features = self.avgpool(x).view(x.size(0), -1)
        main_logits = self.classifier(features)

        if return_trajectory:
            return main_logits, aux_outputs
        return main_logits

    def forward_with_features(self, x):
        """Forward pass returning logits, auxiliary outputs, and features."""
        aux_outputs = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        aux_outputs.append(self.aux_classifiers[0](x))

        x = self.layer2(x)
        aux_outputs.append(self.aux_classifiers[1](x))

        x = self.layer3(x)
        aux_outputs.append(self.aux_classifiers[2](x))

        x = self.layer4(x)
        aux_outputs.append(self.aux_classifiers[3](x))

        features = self.avgpool(x).view(x.size(0), -1)
        main_logits = self.classifier(features)

        return main_logits, aux_outputs, features

    def forward_from_features(self, features):
        """Compute logits from pre-extracted features."""
        return self.classifier(features)
