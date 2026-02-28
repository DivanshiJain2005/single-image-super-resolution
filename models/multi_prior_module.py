import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Utility Blocks
# ------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, 3,
            padding=dilation,
            dilation=dilation
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, 3,
            padding=dilation,
            dilation=dilation
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return identity + out


# ------------------------------------------------------------
# 1️⃣ Edge Prior (Small Receptive Field)
# Focus: High-frequency edges
# ------------------------------------------------------------

class EdgePrior(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        feat = self.model(x)
        return feat


# ------------------------------------------------------------
# 2️⃣ Texture Prior (Medium Receptive Field)
# Uses dilated convolution to capture texture patterns
# ------------------------------------------------------------

class TexturePrior(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, x):
        feat = self.model(x)
        return feat


# ------------------------------------------------------------
# 3️⃣ Structure Prior (Large Receptive Field)
# Uses residual blocks + dilation for global consistency
# ------------------------------------------------------------

class StructurePrior(nn.Module):
    def __init__(self, channels, num_blocks=3):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualBlock(channels, dilation=2))

        self.blocks = nn.Sequential(*blocks)

        self.final = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        feat = self.blocks(x)
        feat = self.final(feat)
        return feat


# ------------------------------------------------------------
# Multi-Prior Wrapper (Optional Convenience Module)
# Combines priors using adaptive weights
# ------------------------------------------------------------

class MultiPriorFusion(nn.Module):
    """
    This module fuses edge, texture and structure priors
    using externally provided weights (from router).
    """

    def __init__(self, channels):
        super().__init__()

        self.edge_prior = EdgePrior(channels)
        self.texture_prior = TexturePrior(channels)
        self.structure_prior = StructurePrior(channels)

    def forward(self, x, weights=None, return_features=False):
        """
        weights: tensor of shape (B, 3)
        """

        edge_feat = self.edge_prior(x)
        texture_feat = self.texture_prior(x)
        structure_feat = self.structure_prior(x)

        if weights is not None:
            # reshape weights
            lambda1 = weights[:, 0].view(-1, 1, 1, 1)
            lambda2 = weights[:, 1].view(-1, 1, 1, 1)
            lambda3 = weights[:, 2].view(-1, 1, 1, 1)

            fused = (
                lambda1 * edge_feat +
                lambda2 * texture_feat +
                lambda3 * structure_feat
            )
        else:
            fused = edge_feat + texture_feat + structure_feat

        if return_features:
            return fused, edge_feat, texture_feat, structure_feat

        return fused


# ------------------------------------------------------------
# Orthogonality / Diversity Loss Utility
# ------------------------------------------------------------

def orthogonality_loss(f1, f2):
    """
    Encourages feature decorrelation between priors.
    """
    b = f1.size(0)
    f1 = f1.view(b, -1)
    f2 = f2.view(b, -1)

    dot = torch.sum(f1 * f2, dim=1)
    return torch.mean(dot ** 2)


def multi_prior_diversity_loss(edge_feat, texture_feat, structure_feat):
    """
    Total diversity constraint.
    """
    loss = (
        orthogonality_loss(edge_feat, texture_feat) +
        orthogonality_loss(edge_feat, structure_feat) +
        orthogonality_loss(texture_feat, structure_feat)
    )
    return loss