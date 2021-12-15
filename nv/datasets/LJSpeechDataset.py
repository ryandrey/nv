import random
import torchaudio
import torch


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, limit=None, *args, **kwargs):
        super().__init__(root=root)
        self._index = list(range(super().__len__()))
        self.limit = limit
        if limit is not None:
            random.seed(42)
            random.shuffle(self._index)
            self._index = self._index[:limit]

    def __getitem__(self, index: int):
        waveform, _, _, _ = super().__getitem__(self._index[index])
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        return waveform, waveform_length

    def __len__(self):
        if self.limit is None:
            return super().__len__()
        else:
            return self.limit
