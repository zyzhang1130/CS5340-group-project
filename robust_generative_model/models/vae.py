import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import padding
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser
from pl_bolts import _HTTPS_AWS_HUB
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
    resnet50_decoder,
    resnet50_encoder,
)
from models.networks import MLP, mnist_encoder, mnist_decoder
from crfseg import CRF


class VAE(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    Model is available pretrained on different datasets:
    Example::
        # not pretrained
        vae = VAE()
        # pretrained on cifar10
        vae = VAE(input_height=32).from_pretrained('cifar10-resnet18')
    """

    pretrained_urls = {
        'cifar10-resnet18': os.path.join(_HTTPS_AWS_HUB, 'vae/vae-cifar10/checkpoints/epoch%3D89.ckpt'),
    }

    def __init__(
        self,
        input_height: int,
        enc_type: str = 'mnist_conv',
        num_classes: int = 10,
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 64,
        kl_coeff: float = 0.1,
        latent_dim: int = 64,
        lr: float = 1e-4,
        k: int = 10,
        input_channels: int = 1,
        py_mode: int = 0,
        recon_loss_type: str = 'l2',
        mlp_hidden_dim: int = 500,
        vae_type: str = 'gfz',
        no_decoder: bool = False,
        per_class: bool = False,
        binary: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            enc_type: option between resnet18 or resnet50
            first_conv: use standard kernel_size 7, stride 2 at start or
                replace it with kernel_size 3, stride 1 conv
            maxpool1: use standard maxpool to reduce spatial dim of feat by a factor of 2
            enc_out_dim: set according to the out_channel count of
                encoder used (512 for resnet18, 2048 for resnet50)
            kl_coeff: coefficient for kl term of the loss
            latent_dim: dim of latent space
            lr: learning rate for Adam
            k: number of samples on latent variables during prediction time
        """

        super(VAE, self).__init__()

        self.save_hyperparameters()

        self.lr = lr
        self.kl_coeff = kl_coeff
        self.enc_out_dim = enc_out_dim
        self.latent_dim = latent_dim
        self.input_height = input_height
        self.num_classes = num_classes
        self.k = k
        self.input_channels = input_channels
        self.py_mode = py_mode
        self.recon_loss_type = recon_loss_type
        self.mlp_hidden_dim = mlp_hidden_dim
        self.vae_type = vae_type
        self.no_decoder = no_decoder
        self.per_class = per_class
        self.binary = binary

        self.example_input_array = torch.rand((1, 1, 28, 28))

        valid_encoders = {
            'resnet18': {
                'enc': resnet18_encoder,
                'dec': resnet18_decoder,
            },
            'resnet50':{
                'enc': resnet50_encoder, 
                'dec': resnet50_decoder,
            },
            'mnist_conv':{
                'enc': mnist_encoder, 
                'dec': mnist_decoder,
            }
        }
        self.valid_encoders = valid_encoders
        self.enc_type = enc_type
        print("vae type:", vae_type)
        if vae_type == 'gfz':
            self.feat_recon_input_dim = self.latent_dim + self.num_classes
        elif vae_type == 'gbz':
            self.feat_recon_input_dim = self.latent_dim

        mlp_y_layers = 2

        if enc_type not in valid_encoders:
            raise Exception("Invalid encoder "+str(enc_type))
        else:
            # self.encoder = valid_encoders[enc_type]['enc']()
            if 'mnist' in enc_type:
                mlp_y_layers = 1
                self.encoder = valid_encoders[enc_type]['enc'](kernel_sizes=[5, 5, 5], 
                    strides=[1, 1, 1], n_channels=[64, 64, 64], maxpool=True)
                if per_class: 
                    self.decoder = nn.ModuleList([valid_encoders[enc_type]['dec'](self.enc_out_dim, 
                        recon_loss_type=self.recon_loss_type) for i in range(self.num_classes)])
                else:
                    self.decoder = valid_encoders[enc_type]['dec'](self.enc_out_dim, 
                        recon_loss_type=self.recon_loss_type)
            else:
                self.encoder = valid_encoders[enc_type]['enc'](first_conv, maxpool1)
                if per_class: 
                    self.decoder = nn.ModuleList([valid_encoders[enc_type]['dec'](self.enc_out_dim, 
                        self.input_height, first_conv, maxpool1) for i in range(self.num_classes)])
                else:
                    self.decoder = valid_encoders[enc_type]['dec'](self.enc_out_dim, 
                        self.input_height, first_conv, maxpool1)
        if self.no_decoder:
            self.decoder = nn.Identity()
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.mlp_mu_z = MLP(self.enc_out_dim+self.num_classes, self.latent_dim, hidden_dim=self.mlp_hidden_dim)
        self.mlp_var_z = MLP(self.enc_out_dim+self.num_classes, self.latent_dim, hidden_dim=self.mlp_hidden_dim)
        self.mlp_y = MLP(self.latent_dim, self.num_classes, num_layers=mlp_y_layers, hidden_dim=self.mlp_hidden_dim)
        self.mlp_feat_recon = MLP(self.feat_recon_input_dim, self.enc_out_dim, hidden_dim=self.mlp_hidden_dim) 

        if self.recon_loss_type == 'crf':
            # self.crf = CRF(n_spatial_dims=2, returns='proba')
            # self.crf = CRF(n_spatial_dims=2, n_iter=1)
            self.crf = CRF(n_spatial_dims=2)

    @staticmethod
    def pretrained_weights_available():
        return list(VAE.pretrained_urls.keys())

    def from_pretrained(self, checkpoint_name):
        if checkpoint_name not in VAE.pretrained_urls:
            raise KeyError(str(checkpoint_name) + ' not present in pretrained weights.')

        return self.load_from_checkpoint(VAE.pretrained_urls[checkpoint_name], strict=False)

    def neg_recon_loss(self, x, x_mu, x_log_var=None):
        x = x.clone().detach()
        activation = torch.sigmoid
        if self.binary: 
            if self.recon_loss_type == 'crf':
                x_mu = self.crf(x_mu)
            return -F.binary_cross_entropy_with_logits(x_mu, x, reduction='none').flatten(start_dim=1).sum(dim=1)
        elif self.recon_loss_type == 'l2':
            x_mu = activation(x_mu)
            return -F.mse_loss(x_mu, x, reduction='none').flatten(start_dim=1).sum(dim=1)
        elif self.recon_loss_type == 'gaussian':
            x_mu = activation(x_mu)
            return self.log_gaussian_prob(x, x_mu, x_log_var)
        else:
            raise Exception("Unrecognized recon loss type")

    def straight_through_estimator(self, x): 
        # x_hard = torch.zeros_like(x)
        # x_hard[x>=0.5] = 1
        x_hard = x.clone().detach()
        x_hard = (x_hard >= 0.5).int()
        x = x_hard + x - x.detach()
        return x

    def forward(self, x, predict=True):
        batch_size = x.size(0)
        if self.binary:
            x = self.straight_through_estimator(x)

        if self.no_decoder:
            xs = x.unsqueeze(1).repeat(1, self.num_classes*self.k, 1).flatten(end_dim=1)
        else:
            xs = x.unsqueeze(1).repeat(1, self.num_classes*self.k, 1, 1, 1).flatten(end_dim=1)
        # print(xs.size())
        y_all = torch.arange(self.num_classes, device=x.device)
        y_all = y_all.view(1,-1).repeat(batch_size, 1).flatten()
        y_onehot = torch.zeros(batch_size*self.num_classes, self.num_classes, device=x.device)
        y_onehot[torch.arange(y_onehot.size(0)), y_all] = 1
        z, x_hat, y_hat, p, q = self._run_step(x, y_onehot, predict)

        # compute p(x|y, z), using log-gaussian
        x_mu, x_log_var = x_hat
        logpx = self.neg_recon_loss(xs, x_mu, x_log_var)        

        # compute p(y|z), using softmax
        yk = y_all.view(batch_size, -1, 1).repeat(1, 1, self.k).flatten()
        if self.py_mode == 0:
            logpy = -F.cross_entropy(y_hat, yk, reduction='none')
        elif self.py_mode == 1:
            logpy = 1

        # compute kl
        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)
        kl = log_qz - log_pz
        kl = kl.sum(dim=-1)
        kl = self.kl_coeff * kl

        lb = logpx + logpy - kl
        lb = lb.reshape(batch_size, self.num_classes, self.k)
        lb = torch.logsumexp(lb, dim=-1) - torch.log(torch.cuda.FloatTensor([self.k])) # - constant logk
        # lb = torch.min(lb, dim=-1)[0]
        # return lb, (logpx, logpy, kl) # logits, logpx, logpy, kl
        lb = torch.softmax(lb, -1)
        return lb

    def _run_step(self, x, y, predict=False):
        # compute z
        if not self.no_decoder:
            x = self.encoder(x)
        
        if predict:
            x = x.reshape(x.size(0), 1, -1).repeat(1, self.num_classes, 1).flatten(end_dim=1)
        
        xy = torch.cat([x, y], dim=-1)
        z_mu = self.mlp_mu_z(xy)
        z_log_var = self.mlp_var_z(xy)

        # sample from z
        if predict:
            z_mu = z_mu.unsqueeze(1).repeat(1, self.k, 1).view(-1, self.latent_dim)
            z_log_var = z_log_var.unsqueeze(1).repeat(1, self.k, 1).view(-1, self.latent_dim)
        p, q, z = self.sample(z_mu, z_log_var)

        # reconstruct x
        if predict:
            y = y.reshape(y.size(0), 1, -1).repeat(1, self.k, 1).flatten(end_dim=1)
        y_dec = self.mlp_y(z)
        if self.vae_type in ['gfz']:
            feat_recon_in = torch.cat([z, y], dim=-1)
        elif self.vae_type in ['gbz']:
            feat_recon_in = z
        decoder_in = self.mlp_feat_recon(feat_recon_in)
        decoder_out = self.decoder(decoder_in)

        if self.recon_loss_type == 'gaussian':
            x_mu, x_log_var = torch.chunk(decoder_out, 2, dim=1)
        else:
            x_mu = decoder_out
            x_log_var = None
        x_recon = (x_mu, x_log_var)
        return z, x_recon, y_dec, p, q

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def log_gaussian_prob(self, input, mu, log_var):
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        log_prob = q.log_prob(input)
        log_prob = log_prob.reshape(log_prob.size(0), -1).sum(dim=-1)
        return log_prob

    def step(self, batch, batch_idx, predict=False):
        x, y = batch
        if self.binary:
            x[x >= 0.5] = 1
            x[x < 0.5] = 0
        batch_size = x.size(0)
        # self.k = batch_size

        # discriminative baseline
        # out = self.encoder(x).flatten(start_dim=1)
        # out = self.mlp_y(out)
        # loss = F.cross_entropy(out, y)
        # y_pred = torch.argmax(out, dim=-1)
        # acc = (y_pred == y).float().sum()/x.size(0)
        # logs = {
        #     'loss': loss, 
        #     'acc': acc
        # }
        # return loss, logs

        # generative model
        if self.no_decoder:
            x = self.encoder(x)
        if predict:
            # lb, (logpx, logpy, kl) = self(x)
            lb = self(x)
            y_pred = torch.argmax(lb, dim=-1)
            accuracy = y_pred.eq(y).float().sum()/batch_size
            loss = -lb.mean()
            logs = {
                # "logpx": logpx,
                # "logpy": logpy,
                # "kl": kl,
                "loss": loss,
                "accuracy": accuracy,
            }
        else:
            y_onehot = torch.zeros(batch_size, self.num_classes, device=x.device)
            y_onehot[torch.arange(batch_size), y] = 1
            z, x_hat, y_hat, p, q = self._run_step(x, y_onehot)

            # compute p(x|y, z), using log-gaussian
            x_mu, x_log_var = x_hat
            logpx = self.neg_recon_loss(x, x_mu, x_log_var)
            logpx = logpx.mean()
            
            # compute p(y|z), using softmax
            if self.py_mode == 0:
                logpy = -F.cross_entropy(y_hat, y).mean()
            elif self.py_mode == 1:
                logpy = 1 

            log_qz = q.log_prob(z)
            log_pz = p.log_prob(z)

            kl = log_qz - log_pz
            kl = kl.sum(dim=-1).mean()
            kl = kl * self.kl_coeff

            lb = logpx + logpy - kl
            loss = -lb

            logs = {
                "logpx": logpx,
                "logpy": logpy,
                "kl": kl,
                "loss": loss,
            }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx, predict=True)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--enc_type", type=str, default='mnist_conv', help="mnist_conv/resnet_18")
        parser.add_argument("--first_conv", action='store_true')
        parser.add_argument("--maxpool1", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument(
            "--enc_out_dim",
            type=int,
            default=64,
            help="64x16 for mnist, 512 for resnet18, 2048 for bigger resnets, adjust for wider resnets"
        )
        parser.add_argument("--kl_coeff", type=float, default=1)
        parser.add_argument("--latent_dim", type=int, default=64)
        parser.add_argument("--mlp_hidden_dim", type=int, default=500)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--data_dir", type=str, default=".")
        parser.add_argument("--py_mode", type=int, default=0)
        parser.add_argument("--recon_loss_type", type=str, default='l2', help='type of reconstruction loss')
        parser.add_argument("--vae_type", type=str, default='gfz', choices=['gfz', 'gbz'])
        parser.add_argument("--no_decoder", action='store_true')
        parser.add_argument("--binary", action='store_true', help='use binary input')
        return parser