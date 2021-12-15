import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm


class SD(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=15, stride=1),
            nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16),
            nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64),
            nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256),
            nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2),
            nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
        ])

        self.norm = norm
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        fmaps = []
        for disc in self.discriminators:
            x = self.norm(disc(x))
            x = self.activation(x)
            fmaps.append(x)
        return fmaps[:-1], fmaps[-1]


class MSD(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [SD(spectral_norm)] + [
                SD(weight_norm) for _ in range(num_layers)
            ]
        )

        self.pools = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2),
            nn.AvgPool1d(4, 2, 2)
        ])

    def forward(self, x):
        fmaps = []
        outs = []

        for i in range(len(self.layers)):
            if i != 0:
                x = self.pools[i - 1](x)

            fmap, out = self.layers[i](x)
            fmaps.append(fmap)
            outs.append(out)

        return outs, fmaps
