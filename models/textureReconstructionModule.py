import torch.nn as nn


class ConvUp(nn.Module):
    """Upsampling via PixelShuffle — avoids checkerboard artifacts from ConvTranspose2d."""

    def __init__(self, ch_in, up_factor, n_feats=128):
        super(ConvUp, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(ch_in, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if up_factor == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, ch_in * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        elif up_factor == 3:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, ch_in * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3),
            )
        elif up_factor == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(n_feats, ch_in * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )

    def forward(self, input):
        return self.tail(self.body(input))


class ConvDown(nn.Module):

    def __init__(self, ch_in, up_factor, n_feats=128):
        super(ConvDown, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(ch_in, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if up_factor == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(n_feats, ch_in, kernel_size=3, padding=1)
            )
        elif up_factor == 3:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=up_factor),
                nn.Conv2d(n_feats, ch_in, kernel_size=3, padding=1)
            )
        elif up_factor == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, stride=up_factor),
                nn.Conv2d(n_feats, ch_in, kernel_size=3, padding=1)
            )

    def forward(self, input):
        return self.tail(self.body(input))
