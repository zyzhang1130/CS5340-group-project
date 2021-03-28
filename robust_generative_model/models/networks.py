import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from argparse import ArgumentParser


def mnist_encoder(kernel_sizes=[5,5,5], strides=[1,1,1],
            n_channels=[32,32,64], paddings=[2,2,2], input_channel=1, maxpool=False):
    return CNNEncoder(kernel_sizes, strides,
            n_channels, paddings, input_channel, maxpool)


def mnist_decoder(self, latent_dim=64, kernel_sizes=[5,5,5], strides=[2,2,2],
            n_channels=[64,64,1], input_shape=[64,4,4], paddings=[2,2,2], output_paddings=[0,1,1],
            num_classes=10, recon_loss_type='gaussian'):
    return CNNDecoder(latent_dim, kernel_sizes, strides,
            n_channels, input_shape, paddings, output_paddings,
            num_classes, recon_loss_type)


class CNNEncoder(nn.Module):
    # consistent across different factorization
    # def __init__(self, kernel_sizes=[5,4,3,5], strides=[1,2,2,1],
    #         n_channels=[32,32,64,64], input_channel=1):
    def __init__(self, kernel_sizes=[5,5,5], strides=[1,1,1],
            n_channels=[32,32,64], paddings=[2,2,2], input_channel=1, maxpool=False):
        super(CNNEncoder, self).__init__()
        self.num_layers = len(kernel_sizes)
        convs = []
        bns = []
        n_channels = [input_channel] + n_channels
        for i in range(self.num_layers):
            conv_i = nn.Conv2d(n_channels[i], n_channels[i+1], kernel_sizes[i], 
                    strides[i], padding=paddings[i])
            convs.append(conv_i)
            bns.append(nn.BatchNorm2d(n_channels[i+1]))
        self.convs = nn.ModuleList(convs)
        self.batch_norms = nn.ModuleList(bns)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) if maxpool else None
        # self.maxpool = nn.Identity()
        self.num_class = 10

    def forward(self, x):
        for i in range(self.num_layers):
            conv = self.convs[i]
            x = conv(x)
            if i < self.num_layers-1:
                x = F.relu(x) 
                x = self.batch_norms[i](x)
            if self.maxpool is not None:
                # if x.size(-1) % 2 == 1:
                #     x = F.pad(x, (0, 1, 0, 1), value=-float('inf'))
                padding = (x.size(-1) % 2)
                x = F.pad(x, (0, padding, 0, padding), value=-float('inf'))
                x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        return x


class CNNDecoder(nn.Module):
    # p(x|y, z)
    # p(y|z)
    # def __init__(self, latent_dim, input_height, kernel_sizes=[4,5,5,3], strides=[1,2,2,1],
    #         n_channels=[32,16,16,1], input_shape=[64,1,1], num_classes=10, recon_loss_type='gaussian', 
    #         maxpool=None):
    def __init__(self, latent_dim, kernel_sizes=[5,5,5], strides=[2,2,2],
            n_channels=[64,64,1], input_shape=[64,4,4], paddings=[2,2,2], output_paddings=[0,1,1],
            num_classes=10, recon_loss_type='gaussian', mlp_hidden_dim=500):
        super(CNNDecoder, self).__init__()
        self.latent_dim = latent_dim
        # self.input_height = input_height
        self.recon_loss_type = recon_loss_type

        # p(x|z, y) 
        input_channel = input_shape[0]
        self.input_shape = input_shape
        self.input_height = input_shape[1]
        # self.mlp = MLP(latent_dim+num_classes, input_channel*self.input_height**2, hidden_dim=mlp_hidden_dim)
        self.num_layers = len(kernel_sizes)
        convs = []
        bns = []
        n_channels = [input_channel] + n_channels
        if self.recon_loss_type == 'gaussian':
            n_channels[-1] = n_channels[-1]*2
        for i in range(self.num_layers):
            # conv_i = nn.ConvTranspose2d(n_channels[i], n_channels[i+1], kernel_sizes[i], strides[i])
            conv_i = nn.ConvTranspose2d(n_channels[i], n_channels[i+1], kernel_sizes[i], 
                    strides[i], padding=paddings[i], output_padding=output_paddings[i])
            convs.append(conv_i)
            bns.append(nn.BatchNorm2d(n_channels[i+1]))
        self.convs = nn.ModuleList(convs)
        self.batch_norms = nn.ModuleList(bns)

        # p(y|z)
        # self.mlp_y = MLP(latent_dim, num_classes, num_layers=1)

    def forward(self, z):
        # compute p(x|y,z)
        # yz = torch.cat([z, y], dim=-1)

        # out = self.mlp(z)
        out = z
        out = out.view(out.size(0), -1, self.input_height, self.input_height)
        for i in range(self.num_layers):
            out = self.convs[i](out)
            if i < self.num_layers - 1:
                out = torch.relu(out)
                # out = self.batch_norms[i](out)
            else: 
                # NOTE: might need to modification here
                out = out
                # out = torch.sigmoid(out)
        x = out 

        # compute p(y|z)
        # y = self.mlp_y(z)
        return x


class MLP(nn.Module):
    # default has 2 hidden layers = 3 linear layers
    def __init__(self, input_dim, output_dim, hidden_dim=500, num_layers=2):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        layers = []
        for i in range(self.num_layers+1):
            if i == 0:
                layer = nn.Linear(input_dim, hidden_dim)
            elif i == self.num_layers:
                layer = nn.Linear(hidden_dim, output_dim)
            else:
                layer = nn.Linear(hidden_dim, hidden_dim)
            layers.append(layer)
            if i < self.num_layers:
                layers.append(nn.ReLU())
            else:
                # layers.append(nn.Sigmoid())
                continue
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
