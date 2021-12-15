import torch


class HiFiGAN_loss:
    def fmaps_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss

    def disc_loss(self, real, gen):
        loss = 0
        for dr, dg in zip(real, gen):
            r_loss = torch.mean((dr - 1) ** 2)
            g_loss = torch.mean(dg ** 2)
            loss += r_loss + g_loss

        return loss

    def gen_loss(self, gen):
        loss = 0
        for dg in gen:
            loss += torch.mean((dg - 1) ** 2)
        return loss
