"""Some explanation to this script and what modules are and why we need them.
Examples: MNIST, CIFAR"""
import torch
from torch import optim
import torch.nn.functional as F
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.core.lightning import LightningModule
from utils.utils import get_argparser_group
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
import os


class MultiClassClassification(LightningModule):
    def __init__(self, hparams, model, logger = None):
        super(MultiClassClassification, self).__init__()
        self.hparams = hparams
        self.model = model
        self.example_input_array = torch.zeros(self.hparams.batch_size, 1, 256, 256)
        self.accuracy = Accuracy()

        self.logger = logger[0] if logger is not None else None  # setting the tensorboard logger
        self.counter = 0
        self.loss = torch.nn.CrossEntropyLoss()

    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, x):
        x = self.model.forward(x)
        return x

    # 2: Optimizer (configure_optimizers hook)
    # see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        # where T_max is the maximum number of iterations
        return [optimizer], [scheduler]

    def loss_function(self, y_pred, y_true):

        loss = self.loss(y_pred, y_true)
        return loss

    # 3: Data
    # was moved to the dataset python file

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, train_batch, batch_idx):
        x = train_batch['image']
        y_true = train_batch['label']
        # display a few images
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment[0].add_images('Epoch {} Batch {}'.format(self.current_epoch, batch_idx), x, 0)
        # forward pass
        y_pred = self.forward(x)
        # calculate loss
        train_loss = self.loss_function(y_pred, y_true)
        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            train_loss = train_loss.unsqueeze(0)

        if batch_idx % 500 == 0:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            self.log_images(x=x,
                            y_true=y_true,
                            y_pred=y_pred.detach(),
                            current_epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            phase='train')

        return {'loss': train_loss}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):
        self.log('Train Total Loss', train_step_output['loss'], on_step=True, on_epoch=True)
        return train_step_output

    # 5 Validation Loop
    def validation_step(self, val_batch, batch_idx):
        x = val_batch['image']
        y_true = val_batch['label']
        y_pred = self.forward(x)
        val_loss = self.loss_function(y_pred, y_true)

        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
        correct_predictions = torch.sum((y_pred == y_true))
        val_acc = correct_predictions/y_true.size(0)

        if batch_idx % 500 == 0:
            y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
            self.log_images(x=x,
                            y_true=y_true,
                            y_pred=y_pred.detach(),
                            current_epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            phase='val')

        return {'val_loss': val_loss, 'val_acc': val_acc}

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the validation_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def validation_step_end(self, val_step_output):
        self.log('Val Total Loss', val_step_output['val_loss'], on_step=True, on_epoch=True)
        return val_step_output

    # 6 Test Loop
    def test_step(self, test_batch, batch_idx):
        x = test_batch['image']
        y_true = test_batch['label']
        y_pred = self(x)
        loss = self.loss_function(y_pred, y_true)
        y_pred = torch.argmax(y_pred, dim=1, keepdim=True)

        return {'test_loss': loss, 'test_label':y_true, 'test_pred': y_pred}

    def test_epoch_end(self, test_step_output):
        correctly_labeled = 0
        total_num_samples = 0

        y_true_list = []
        y_pred_list = []
        for item in test_step_output:
            y_true = item['test_label'].to("cpu").numpy().flatten()
            y_pred = item['test_pred'].to("cpu").numpy().flatten()

            correctly_labeled += np.sum(np.where(y_pred == y_true, 1, 0))
            total_num_samples += y_true.size

            if y_true.size == 1:
                y_true_list.extend(y_true)
                y_pred_list.extend(y_pred)
                continue

            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())

        pd_frame = pd.DataFrame({'y_true':y_true_list, 'y_pred':y_pred_list})
        pd_frame_metric = pd.DataFrame({'accuracy':[correctly_labeled/total_num_samples]})

        pd_frame.to_csv(os.path.join(self.hparams.output_path, "test_results.csv"))
        pd_frame_metric.to_csv(os.path.join(self.hparams.output_path, "metrics.csv"))
        self.log('Test Accuracy', correctly_labeled/total_num_samples)

    def log_images(self, x, y_true, y_pred, current_epoch, batch_idx, filename=" ", phase='', augmentations=" "):

        if phase == 'train':
            name = f'Train Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]}'
            f'(augmentation: {augmentations[0]})'

        elif phase == 'val':
            name = f'Val Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]}'
            f'(augmentation: {augmentations[0]})'

        else:
            name = f'Unknown Phase Epoch: {current_epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        images = np.squeeze(x.to("cpu").numpy(), axis=1)
        labels = np.squeeze(y_true.to("cpu").numpy())
        preds = np.squeeze(y_pred.to("cpu").numpy())

        # Taking the first image, label prediction from the input batch
        num_subplots = min(images.shape[0], 3)

        fig, axs = plt.subplots(1, num_subplots)
        for i, ax in enumerate(axs):

            image = np.squeeze(images[i, :, :])
            label = labels[i]
            pred = preds[i]

            pos0 = ax.imshow(image, cmap='gray')
            ax.set_axis_off()
            ax.set_title("Labels: {}\n Pred: {}".format(label, pred))
            divider = make_axes_locatable(ax)
            cax0 = divider.append_axes("right", size="5%", pad=0.05)
            tick_list = np.linspace(np.min(image), np.max(image), 5)
            cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
            cbar0.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

        fig.tight_layout()

        fig.suptitle(name, fontsize=16)
        self.logger[0].experiment.add_figure(tag=name, figure=fig)


    @staticmethod
    def add_module_specific_args(parser):
        specific_args = get_argparser_group(title='Model options', parser=parser)
        specific_args.add_argument('--input_channels', default=3, type=int,
                                              help='number of input channels (default: 3)')
        specific_args.add_argument('--probability_threshold', default=0.5, type=float)

        return parser