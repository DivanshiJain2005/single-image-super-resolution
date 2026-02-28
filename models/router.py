import torch
import torch.nn as nn


class PriorRouter(nn.Module):
    """
    Adaptive Routing Network for dynamic prior weighting.

    Input:
        x : Feature map (B, C, H, W)

    Output:
        weights : (B, 3) tensor
                  Adaptive weights for:
                  [edge_prior, texture_prior, structure_prior]
    """

    def __init__(self, channels, reduction=16):
        super(PriorRouter, self).__init__()

        hidden_dim = max(channels // reduction, 32)

        # Global context extraction
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected routing head
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x : (B, C, H, W)

        Returns:
            weights : (B, 3)
        """
        b, c, _, _ = x.size()

        # Global pooling → (B, C, 1, 1)
        pooled = self.pool(x)

        # Flatten → (B, C)
        pooled = pooled.view(b, c)

        # Routing weights → (B, 3)
        weights = self.fc(pooled)

        return weights