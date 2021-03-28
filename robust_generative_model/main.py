from operator import mod
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.vae import VAE
from argparse import ArgumentParser


def cli_main(args=None):
    from pl_bolts.datamodules import CIFAR10DataModule, MNISTDataModule

    pl.seed_everything()

    parser = ArgumentParser()
    parser.add_argument("--dataset", default="mnist", type=str, choices=["cifar10", "mnist"])
    parser.add_argument("--test", action='store_true', help='use test')
    parser.add_argument("--attack", action='store_true', help='test attack')
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--pretrained", default=None, type=str)
    parser.add_argument("--name", default='', type=str)
    parser.add_argument("--log_graph", action='store_true', help='log computational graph to tensorboard')
    script_args, _ = parser.parse_known_args(args)

    if script_args.dataset == "cifar10":
        dm_cls = CIFAR10DataModule
    elif script_args.dataset == "mnist":
        dm_cls = MNISTDataModule
    else:
        raise ValueError(f"undefined dataset {script_args.dataset}")

    parser = VAE.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = dm_cls.from_argparse_args(args)
    args.input_height = dm.size()[-1]

    if args.max_steps == -1:
        args.max_steps = None

    if args.test:
        model = VAE.load_from_checkpoint(args.checkpoint)
        trainer = pl.Trainer.from_argparse_args(args, logger=False)
        trainer.test(model, datamodule=dm)
    elif args.attack:
        from attacks import attack
        dm.setup('test')
        model = VAE.load_from_checkpoint(args.checkpoint)
        attack(model, dm.test_dataloader(), eps=args.eps)
    else:
        if args.pretrained:
            model = VAE(**vars(args)).from_pretrained(args.pretrained)
        else:
            model = VAE(**vars(args))
        logging_dir = args.dataset + 'logs/'
        from pytorch_lightning import loggers as pl_loggers 
        logger_name = args.vae_type + '_' + args.name if len(args.name) > 0 else args.vae_type
        tb_logger = pl_loggers.TensorBoardLogger(args.dataset +'_tblogs/', name=logger_name, log_graph=args.log_graph)
        trainer = pl.Trainer.from_argparse_args(args, logger=tb_logger, default_root_dir=logging_dir,
             callbacks=[EarlyStopping(monitor='val_loss', patience=5)])
        trainer.fit(model, datamodule=dm)
    # return dm, model, trainer


if __name__ == "__main__":
    # patch MNIST download
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    
    cli_main()