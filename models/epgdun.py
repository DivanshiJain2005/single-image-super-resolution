import torch
import torch.nn as nn
import torch.nn.functional as F

from models.denoisingModule import EncodingBlock, EncodingBlockEnd, DecodingBlock, DecodingBlockEnd
from models.residualprojectionModule import UCNet
from models.textureReconstructionModule import ConvDown, ConvUp
from models.edgemap import EdgeMap
from models.edgefeatureextractionModule import EAFM
from models.intermediatevariableupdateModule import EGIM
from models.variableguidereconstructionModule import IGRM


def make_model(args, parent=False):
    return EPGDUN(args)


# ==========================================================
# ---------------- Multi Prior Module ----------------
# ==========================================================
class MultiPriorModule(nn.Module):
    """
    Produces multiple priors from input feature.
    Here we use 3 experts:
        1. Local convolution prior
        2. Dilated convolution prior
        3. Edge-enhanced prior
    """
    def __init__(self, in_channels=3, num_priors=3):
        super(MultiPriorModule, self).__init__()

        self.num_priors = num_priors

        # Local prior
        self.prior1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

        # Dilated prior
        self.prior2 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

        # Edge-aware prior
        self.prior3 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, 3, padding=1)
        )

    def forward(self, x):
        p1 = self.prior1(x)
        p2 = self.prior2(x)
        p3 = self.prior3(x)
        return [p1, p2, p3]


# ==========================================================
# ---------------- Router Module ----------------
# ==========================================================
class Router(nn.Module):
    """
    Produces adaptive weights for each prior.
    Output shape: [B, num_priors]
    """
    def __init__(self, in_channels=3, num_priors=3):
        super(Router, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_priors)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        pooled = self.pool(x).view(b, c)
        weights = self.fc(pooled)
        return weights


# ==========================================================
# ---------------- EPGDUN Main Model ----------------
# ==========================================================
class EPGDUN(nn.Module):
    def __init__(self, args):
        super(EPGDUN, self).__init__()

        self.channel0 = args.n_colors
        self.up_factor = args.scale[0]
        self.patch_size = args.patch_size
        self.batch_size = int(args.batch_size / args.n_GPUs)

        # Unfolding iterations
        self.T = 4

        # ---------------- Texture Reconstruction ----------------
        self.eta = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.T)])
        self.mu = nn.ParameterList([nn.Parameter(torch.tensor(0.1)) for _ in range(self.T)])
        self.delta_3 = nn.ParameterList([nn.Parameter(torch.tensor(0.5)) for _ in range(self.T)])

        self.conv_up = ConvUp(3, self.up_factor)
        self.conv_down = ConvDown(3, self.up_factor)

        # ---------------- Edge-Guided ----------------
        self.edgemap = EdgeMap()
        self.EAFM = EAFM()
        self.EGIM = EGIM()
        self.IGRM = IGRM()

        # ---------------- Multi-Prior + Router ----------------
        self.mpm = MultiPriorModule(in_channels=3, num_priors=3)
        self.router = Router(in_channels=3, num_priors=3)

    # ==========================================================
    # Forward
    # ==========================================================
    def forward(self, y, idx_scale=None):

        x_init = [F.interpolate(y, scale_factor=self.up_factor,
                                mode='bilinear', align_corners=False)]
        v_init = [x_init[0]]

        for i in range(self.T):

            x_curr = x_init[i]
            input_target = x_curr.shape[2:]

            # Downsample
            x_down = F.interpolate(x_curr, scale_factor=1/self.up_factor,
                                   mode='bilinear', align_corners=False)

            v_down = F.interpolate(v_init[i], size=x_down.shape[2:],
                                   mode='bilinear', align_corners=False)

            y_down_sampled = F.interpolate(y, size=x_down.shape[2:],
                                           mode='bilinear', align_corners=False)

            # Conv difference
            conv_down_out = self.conv_down(x_down)
            y_down_resized = F.interpolate(y_down_sampled,
                                           size=conv_down_out.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)

            conv_diff = self.conv_up(conv_down_out - y_down_resized)

            v_down_resized = F.interpolate(v_down,
                                           size=conv_diff.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)

            x_down_resized = F.interpolate(x_down,
                                           size=conv_diff.shape[2:],
                                           mode='bilinear',
                                           align_corners=False)

            # ======================================================
            # Multi-Prior + Router Fusion
            # ======================================================
            priors = self.mpm(x_down_resized)
            weights = self.router(x_down_resized)
            weights = F.softmax(weights, dim=1)

            fused_prior = 0
            for j in range(len(priors)):
                fused_prior += weights[:, j:j+1, None, None] * priors[j]

            # ======================================================
            # Residual Update
            # ======================================================
            residual = fused_prior - self.delta_3[i] * (
                conv_diff + self.mu[i] * (v_down_resized - fused_prior)
            )

            # Edge-guided refinement
            f_curr = self.EAFM(self.edgemap(fused_prior))
            v_next = self.EGIM(v_down_resized, f_curr)
            x_next = self.IGRM(residual, v_next)

            # Upsample back
            x_next = F.interpolate(x_next,
                                   size=input_target,
                                   mode='bilinear',
                                   align_corners=False)

            v_next = F.interpolate(v_next,
                                   size=input_target,
                                   mode='bilinear',
                                   align_corners=False)

            x_init.append(x_next)
            v_init.append(v_next)

        return x_next