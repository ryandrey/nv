import io

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from nv.base import BaseTrainer
from nv.logger.utils import plot_spectrogram_to_buf
from nv.utils import inf_loop, MetricTracker
from nv.featurizer import MelSpectrogram, MelSpectrogramConfig


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            generator,
            mpd,
            msd,
            criterion,
            optimizer_g,
            optimizer_d,
            config,
            device,
            data_loader,
            lr_scheduler_g=None,
            lr_scheduler_d=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(generator, mpd, msd, optimizer_g, optimizer_d, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        self.criterion = criterion
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d
        self.log_step = config["trainer"].get("log_step", 50)

        self.train_metrics = MetricTracker(
            "gen_loss", "disc_loss",
            "grad norm gen", "grad norm mpd", "grad norm msd",
            writer=self.writer
        )

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.generator.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.mpd.parameters(), self.config["trainer"]["grad_norm_clip"]
            )
            clip_grad_norm_(
                self.msd.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.generator.train()
        self.mpd.train()
        self.msd.train()

        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            batch = batch.to(self.device)
            batch.melspec = self.featurizer(batch.waveform)
            try:
                gen_l, disc_l = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.generator.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    for p in self.mpd.parameters():
                        if p.grad is not None:
                            del p.grad
                    for p in self.msd.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm gen", self.get_grad_norm(self.generator))
            if not self.config["trainer"].get("overfit", False):
                self.train_metrics.update("grad norm mpd", self.get_grad_norm(self.mpd))
                self.train_metrics.update("grad norm msd", self.get_grad_norm(self.msd))
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Gen_Loss: {:.6f}, Disc_Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), gen_l, disc_l
                    )
                )
                self.writer.add_scalar(
                    "learning rate gen", self.lr_scheduler_g.get_last_lr()[0]
                )
                self.writer.add_scalar(
                    "learning rate disc", self.lr_scheduler_d.get_last_lr()[0]
                )
                self._log_scalars(self.train_metrics)
            if batch_idx >= self.len_epoch:
                break
        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()
        log = self.train_metrics.result()

        self._check_examples()

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):

        output = self.generator(batch.melspec)
        batch.waveform_pred = output

        if not self.config["trainer"].get("overfit", False):
            self.optimizer_d.zero_grad()

            # MPD
            mpd_real, _ = self.mpd(batch.waveform)
            mpd_gen, _ = self.mpd(output.detach())
            loss_mpd = self.criterion.disc_loss(mpd_real, mpd_gen)

            # MSD
            msd_real, _ = self.msd(batch.waveform)
            msd_gen, _ = self.msd(output.detach())
            loss_msd = self.criterion.disc_loss(msd_real, msd_gen)

            total_disc_loss = loss_mpd + loss_msd
            total_disc_loss.backward()
            self.optimizer_d.step()

        self.optimizer_g.zero_grad()

        output_melspec = self.featurizer(output)

        diff_len = output_melspec.shape[-1] - batch.melspec.shape[-1]
        if diff_len < 0:
            output_melspec = F.pad(output_melspec, (0, diff_len), "constant", -11.5129251)
        elif diff_len > 0:
            batch.melspec = F.pad(batch.melspec, (0, diff_len), "constant", -11.5129251)
        loss_mel = nn.L1Loss()(batch.melspec, output_melspec)

        if not self.config["trainer"].get("overfit", False):
            diff_len = output.shape[-1] - batch.waveform.shape[-1]
            waveform = F.pad(batch.waveform, (0, diff_len))
            mpd_real, mpd_fmaps_real = self.mpd(waveform)
            mpd_gen, mpd_fmaps_gen = self.mpd(output)
            loss_fmaps_mpd = self.criterion.fmaps_loss(mpd_fmaps_real, mpd_fmaps_gen)

            msd_real, msd_fmaps_real = self.msd(waveform)
            msd_gen, msd_fmaps_gen = self.msd(output)
            loss_fmaps_msd = self.criterion.fmaps_loss(msd_fmaps_real, msd_fmaps_gen)

            loss_gen_mpd = self.criterion.gen_loss(mpd_gen)
            loss_gen_msd = self.criterion.gen_loss(msd_gen)

        if self.config["trainer"].get("overfit", False):
            total_gen_loss = loss_mel
            total_disc_loss = loss_mel * 45
        else:
            total_gen_loss = 45 * loss_mel + 2 * (loss_fmaps_mpd + loss_fmaps_msd) + loss_gen_mpd + loss_gen_msd

        total_gen_loss.backward()
        #self._clip_grad_norm()
        self.optimizer_g.step()

        metrics.update("gen_loss", total_gen_loss.item())
        metrics.update("disc_loss", total_disc_loss.item())

        return total_gen_loss.item(), total_disc_loss.item()

    def _check_examples(self):
        self.generator.eval()
        self.mpd.eval()
        self.msd.eval()

        with torch.no_grad():
            for batch in self.data_loader:
                batch.to(self.device)
                break

            batch.melspec = self.featurizer(batch.waveform[:1])
            output = self.generator(batch.melspec)

            self._log_spectrogram("pred_spec", self.featurizer(output)[0])
            self._log_spectrogram("true_spec", batch.melspec[0])
            self._log_audio("pred_wav", output[0].detach())
            self._log_audio("true_wav", batch.waveform[0])

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, name, spectrogram):
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram.cpu()))
        self.writer.add_image(name, ToTensor()(image))

    def _log_audio(self, audio_name, wav):
        print(wav.shape)
        self.writer.add_audio(audio_name, wav, sample_rate=22050)

    @torch.no_grad()
    def get_grad_norm(self, model, norm_type=2):
        parameters = model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
