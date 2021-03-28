from operator import mod
from absl import app, flags
from pytorch_lightning.metrics.classification import accuracy
from torch._C import device
from easydict import EasyDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from models.vae import VAE
import pytorch_lightning as pl
from pytorch_lightning.metrics.classification.accuracy import Accuracy

from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.projected_gradient_descent import (
    projected_gradient_descent,
)

FLAGS = flags.FLAGS

def attack(model, dataset, attack_alg='fgsm', eps=0.1, device='cuda'):
    accuracy = Accuracy().to(device)
    model = model.to(device)
    model.eval()
    for x, y in dataset:
        x, y = x.to(device), y.to(device)
        # if model.binary:
        #     x[x >= 0.5] = 1
        #     x[x < 0.5] = 0
        x_fgsm = fast_gradient_method(model, x, eps, np.inf, clip_min=0, clip_max=1)
        _, y_pred_fgsm = model(x_fgsm).max(1)
        accuracy(y_pred_fgsm, y)
    acc = accuracy.compute()
    print(acc)

