import torch
import argparse
import collections
import torchaudio
from nv.utils.parse_config import ConfigParser

import nv.model as module_arch
from nv.featurizer import MelSpectrogramConfig, MelSpectrogram


def main(config):
    device = torch.device('cuda:0')
    generator = config.init_obj(config["arch"], module_arch)
    checkpoint = torch.load(config.resume, device)
    generator.load_state_dict(checkpoint["state_dict"])
    generator.to(device)
    featurizer = MelSpectrogram(MelSpectrogramConfig).to(device)
    generator.eval()
    paths = ["1.wav",
             "2.wav",
             "3.wav"]

    for i, path in enumerate(paths):
        true_wav, _ = torchaudio.load(path)
        spec = featurizer(true_wav.to(device))
        output = generator(spec)
        torchaudio.save(f"{i + 1}_pred.wav", output.cpu(), sample_rate=22050)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
