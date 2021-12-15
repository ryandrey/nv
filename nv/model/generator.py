import torch
import torch.nn as nn
import torch.nn.functional as F

from nv.featurizer import MelSpectrogramConfig


class ResBlock(nn.Module):
    def __init__(self, hid, kernel, d_ri):
        super().__init__()
        self.dri_blocks = nn.ModuleList()
        for m in range(len(d_ri)):
            drim_body = []
            for l in range(len(d_ri[m])):
                drim_body.extend([
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(hid, hid, kernel, stride=1, dilation=d_ri[m][l])
                ])
            self.dri_blocks.append(nn.Sequential(*drim_body))

    def forward(self, x):
        output = x
        for block in self.dri_blocks:
            output += block(x)

        return output


class MRF(nn.Module):
    def __init__(self, hid, k_r, d_r):
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(len(k_r)):
            self.res_blocks.append(ResBlock(hid, k_r[i], d_r[i]))

    def forward(self, x):
        output = torch.zeros(x.size())

        for res_block in self.res_blocks:
            output += res_block(x)
        return output


class Generator(nn.Module):
    def __init__(self, hid, k_u, k_r, d_r, leaky=0.1):
        super().__init__()
        n_mels = MelSpectrogramConfig.n_mels
        self.prep_conv = nn.Conv1d(n_mels, hid, kernel_size=7, stride=1, padding=3)
        self.leaky = leaky

        cur_hid = hid
        net = []
        for i in range(len(k_u)):
            net.extend([nn.LeakyReLU(self.leaky),
                        nn.ConvTranspose1d(cur_hid, cur_hid // 2, kernel_size=k_u[i], stride=k_u[i] // 2),
                        MRF(cur_hid // 2, k_r, d_r)])
            cur_hid //= 2
        self.gen_net = nn.Sequential(*net)
        self.post_conv = nn.Conv1d(cur_hid, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x):
        output = self.prep_conv(x)

        output = self.gen_net(output)

        output = F.leaky_relu(output, self.leaky)
        output = self.post_conv(output)
        output = nn.Tanh()(output)

        return output
