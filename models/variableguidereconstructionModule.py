import torch
import torch.nn as nn
import torch.nn.functional as F

from models.intermediatevariableupdateModule import NonLocalBlock


class PFM_1(nn.Module):  # 并行交叉融合模块
    def __init__(self):
        super(PFM_1, self).__init__()

        self.convh = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.convr = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.convh_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.convr_1 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

    def forward(self, r, h):
        h = F.relu(self.convh(h))
        r = F.relu(self.convr(r))
        h_s = self.sigmoid(h)
        r_s = self.sigmoid(r)
        r_m = r * h_s
        h_m = h * r_s
        h_p = h_m + h
        r_p = r_m + r
        h = self.convh_1(h_p)
        r = self.convr_1(r_p)
        h = self.sigmoid(h)
        r = self.tanh(r)
        final = r * h
        return final  # 32 channels


class IGRM(nn.Module):
    def __init__(self):
        super(IGRM, self).__init__()

        self.pfm = PFM_1()
        self.nonlocalblock = NonLocalBlock(32)

        self.conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.baseBlock = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, r, h):
        p = self.pfm(r, h)
        bb_1 = self.baseBlock(p)  # [B, 32, H, W]

        B, C, H, W = bb_1.shape
        target_size = (H, W)

        # LSTM along width dimension
        lstm_input = bb_1.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        lstm_input = lstm_input.view(B * H, W, C)  # treat each row as sequence
        lstm = nn.LSTM(input_size=C, hidden_size=C, num_layers=2, batch_first=True).to(bb_1.device)
        output, _ = lstm(lstm_input)  # [B*H, W, C]

        # reshape back to [B, C, H, W]
        output = output.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        # optional non-local block
        # output = self.nonlocalblock(output)

        bb_2 = self.baseBlock(output)
        con = self.conv(bb_2)
        final = con + r
        return final
