from argparse import ArgumentParser
from subprocess import call, run
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, tensor, FloatTensor
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from tensorboard import program
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class MLP(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.fc1 = nn.Linear(784, 128, bias=True)
        self.fc2 = nn.Linear(128, 10, bias=True)

    def forward(self, xb): 
        xb = xb.view([-1, 784])
        xb = F.relu(self.fc1(xb))
        xb = F.log_softmax(self.fc2(xb), dim=1)
        return (xb)

    def train_dataloader(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_ds = datasets.MNIST('.data/', download=True, train=True, transform=self.transform)
        return (torch.utils.data.DataLoader(train_ds, batch_size=self.hparams.batch_size, num_workers=6, shuffle=True))

    def val_dataloader(self):
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        valid_ds = datasets.MNIST('.data/', download=True, train=False, transform=self.transform)
        return (torch.utils.data.DataLoader(valid_ds, batch_size=self.hparams.batch_size, num_workers=6, shuffle=True))

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        train_loss = F.nll_loss(Y_hat, Y)
        logs = {'train_loss' : train_loss}
        return {'loss' : train_loss, 'log' : logs}
    
    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y_hat = self(X)
        val_loss = F.nll_loss(Y_hat, Y)
        logs = {'val_loss' : val_loss}
        return {'val_loss' : val_loss, 'log' : logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'valid_loss_for_epoch' : avg_loss}
        return {'avg_val_loss' : avg_loss, 'log' : logs}

if __name__ == "__main__":
    # call(["rm", "-rf", "./lightning_logs"]) # clear the logs
    parser = ArgumentParser()
    # parametrize the network
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--experiment_name', type=str, default='default')
    # add default args
    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    logger = TensorBoardLogger("./lightning_logs", name=hparams.experiment_name)
    model = MLP(hparams)
    trainer = Trainer(max_epochs=hparams.max_epochs, logger=logger)
    trainer.fit(model)