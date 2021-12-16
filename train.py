import argparse
import collections
import warnings

import numpy as np
import torch
import itertools

import nv.loss as module_loss
import nv.model as module_arch
from nv.datasets.utils import get_dataloaders
from nv.trainer import Trainer
from nv.utils import prepare_device
from nv.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 112
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = config.init_obj(config["arch"], module_arch)
    mpd = config.init_obj(config["mpd"], module_arch)
    msd = config.init_obj(config["msd"], module_arch)
    logger.info(generator)
    logger.info(mpd)
    logger.info(msd)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    print(device)
    generator = generator.to(device)
    mdp = mpd.to(device)
    msd = msd.to(device)
    if len(device_ids) > 1:
        generator = torch.nn.DataParallel(generator, device_ids=device_ids)
        mpd = torch.nn.DataParallel(mdp, device_ids=device_ids)
        msd = torch.nn.DataParallel(msd, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = module_loss.HiFiGAN_loss()

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, generator.parameters())
    print(f"generator_params = {sum([p.numel() for p in generator.parameters()])}")
    optimizer_g = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    lr_scheduler_g = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_g)

    trainable_params_mpd = filter(lambda p: p.requires_grad, mpd.parameters())
    trainable_params_msd = filter(lambda p: p.requires_grad, msd.parameters())
    print(f"mpd_params = {sum([p.numel() for p in mpd.parameters()])}")
    print(f"msd_params = {sum([p.numel() for p in msd.parameters()])}")
    optimizer_d = config.init_obj(config["optimizer"], torch.optim, itertools.chain(trainable_params_mpd,
                                                                                    trainable_params_msd))
    lr_scheduler_d = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer_d)

    trainer = Trainer(
        generator,
        mpd,
        msd,
        criterion,
        optimizer_g,
        optimizer_d,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        lr_scheduler_g=lr_scheduler_g,
        lr_scheduler_d=lr_scheduler_d,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()


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
