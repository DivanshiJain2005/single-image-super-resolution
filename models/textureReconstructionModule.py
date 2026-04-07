import torch.nn as nn


class ConvUp(nn.Module):
    """Upsampling via PixelShuffle — avoids checkerboard artifacts from ConvTranspose2d."""

    def __init__(self, ch_in, up_factor):
        super(ConvUp, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        if up_factor == 2:
            self.tail = nn.Sequential(
                nn.Conv2d(64, ch_in * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )
        elif up_factor == 3:
            self.tail = nn.Sequential(
                nn.Conv2d(64, ch_in * 9, kernel_size=3, padding=1),
                nn.PixelShuffle(3),
            )
        elif up_factor == 4:
            self.tail = nn.Sequential(
                nn.Conv2d(64, 64 * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(64, ch_in * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
            )

    def forward(self, input):
        return self.tail(self.body(input))


class ConvDown(nn.Module):

    def __init__(self, ch_in, up_factor):

        super(ConvDown, self).__init__()

        body = [
            nn.Conv2d(ch_in, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=3 // 2),
            nn.ReLU(),
        ]

        if up_factor == 4:
            modules_tail = [
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=2),
                nn.Conv2d(64, ch_in, kernel_size=3, padding=3 // 2, bias=True)
            ]
        elif up_factor == 3:
            modules_tail = [
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=up_factor),
                nn.Conv2d(64, ch_in, kernel_size=3, padding=3 // 2, bias=True)
            ]
        elif up_factor == 2:
            modules_tail = [
                nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=up_factor),
                nn.Conv2d(64, ch_in, kernel_size=3, padding=3 // 2, bias=True)
            ]

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, input):

        out = self.body(input)
        out = self.tail(out)
        return out
