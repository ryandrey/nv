import logging
from typing import Tuple, Dict, Optional, List, Union

import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from nv.featurizer import MelSpectrogram, MelSpectrogramConfig

logger = logging.getLogger(__name__)


@dataclass
class Batch:
    waveform: torch.Tensor
    waveform_length: torch.Tensor
    melspec: Optional[torch.Tensor] = None
    waveform_pred = Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        self.waveform = self.waveform.to(device)
        self.waveform_length = self.waveform_length.to(device)
        if self.melspec is not None:
            self.melspec = self.melspec.to(device)

        return self


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]):
        waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        return Batch(waveform, waveform_length)
