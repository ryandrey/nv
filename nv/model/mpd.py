import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F


class PD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period

        self.discriminators = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, kernel_size=(5, 1), stride=(3, 1))),
            weight_norm(nn.Conv2d(32, 128, kernel_size=(5, 1), stride=(3, 1))),
            weight_norm(nn.Conv2d(128, 512, kernel_size=(5, 1), stride=(3, 1))),
            weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=(3, 1))),
            weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=(1, 1))),
        ])
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        fmaps = []
        batch_size, t_size = x.shape

        # found this pad part of code on jik876 github
        if t_size % self.period != 0:
            n_pad = self.period - (t_size % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t_size += n_pad
        x = x.view(batch_size, 1, t_size // self.period, self.period)

        for disc in self.discriminators:
            x = disc(x)
            x = self.activation(x)
            fmaps.append(x)

        return fmaps[:-1], fmaps[-1]


class MPD(nn.Module):
    def __init__(self, mpd_periods):
        super().__init__()
        self.layers = nn.ModuleList(
            [PD(period) for period in mpd_periods]
        )

    def forward(self, x):
        outs = []
        fmaps = []

        for layer in self.layers:
            fmap, out = layer(x)
            outs.append(out)
            fmaps.append(fmap)

        return outs, fmaps
