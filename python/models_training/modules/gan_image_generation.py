import pytorch_lightning as pl
from utils.utils import get_argparser_group
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from PIL import Image
import torch

class GanImageGeneration(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):

        super().__init__()

        self.hparams = hparams
        self.model = model
        # self.example_input_array = torch.zeros(1, 1, 256, 256)

        # TODO: losses might be moved here
        # self.adversarial_criterion = nn.BCEWithLogitsLoss()
        # self.recon_criterion = nn.L1Loss()

    def configure_optimizers(self):
        return self.model.configure_optimizers()

    # def forward(self, real, condition = None, optimizer_idx=0):
    #
    #     if condition is None:
    #         return 0
    #     if optimizer_idx == 0:
    #
    #         fake_images = self.gen(condition).detach()
    #         logits = self.patch_gan(fake_images, condition)
    #
    #         real_logits = self.patch_gan(real, condition)
    #
    #
    #     elif optimizer_idx == 1:
    #         fake_images = self.gen(condition)
    #         logits = self.patch_gan(fake_images, condition)
    #         adversarial_loss = self.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))
    #
    #         # calculate reconstruction loss
    #         recon_loss = self.recon_criterion(fake_images, real)
    #         lambda_recon = self.hparams.lambda_recon
    #         loss = self.model.forward(real, condition, step='gen')
    #
    #     return fake_images, logits, real_logits

    def training_step(self, batch, batch_idx, optimizer_idx):

        real, condition, _ = batch

        # Discriminator step
        if optimizer_idx == 0:
            fake_images = self.model.gen(condition).detach()
            fake_logits = self.model.patch_gan(fake_images, condition)

            real_logits = self.model.patch_gan(real, condition)

            fake_loss = self.model.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
            real_loss = self.model.adversarial_criterion(real_logits, torch.ones_like(real_logits))

            loss = (real_loss + fake_loss) / 2
            self.log('Discriminator Loss', loss)

        # Generator step
        elif optimizer_idx == 1:
            # Pix2Pix has adversarial and a reconstruction loss
            # First calculate the adversarial loss
            fake_images = self.model.gen(condition)
            disc_logits = self.model.patch_gan(fake_images, condition)
            adversarial_loss = self.model.adversarial_criterion(disc_logits, torch.ones_like(disc_logits))

            # calculate reconstruction loss
            recon_loss = self.model.recon_criterion(fake_images, real)
            lambda_recon = self.model.hparams.lambda_recon

            loss = adversarial_loss + lambda_recon * recon_loss
            self.log('Generator Loss', loss)

        if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx == 0 and optimizer_idx == 0:
            print("Logging images")
            # pass
            # fake = self.model.gen(condition).detach()  # TODO: change this!
            self.log_images(condition_t=condition,
                            real_t=real,
                            fake_t=fake_images,
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            filename=" ",
                            phase='train')

        return loss

    def validation_step(self, batch, batch_idx):

        real, condition, _ = batch

        fake_images = self.model.gen(condition).detach()
        fake_logits = self.model.patch_gan(fake_images, condition)

        real_logits = self.model.patch_gan(real, condition)

        fake_loss = self.model.adversarial_criterion(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.model.adversarial_criterion(real_logits, torch.ones_like(real_logits))

        val_loss = (real_loss + fake_loss) / 2

        if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx % 100 == 0:
            self.log_images(condition_t=condition,
                            real_t=real,
                            fake_t=fake_images,
                            epoch=self.current_epoch,
                            batch_idx=batch_idx,
                            filename=" ",
                            phase='val')

        print(val_loss)

        return val_loss

    def test_step(self, batch, batch_idx):

        # self.model.set_eval()

        real, condition, filenames = batch

        fake = self.model.gen(condition).detach()  # TODO: change this!
        # self.log_images(condition_t=condition,
        #                 real_t=real,
        #                 fake_t=fake,
        #                 epoch=self.current_epoch,
        #                 batch_idx=batch_idx,
        #                 filename=" ",
        #                 phase='test')

        self.save_test_image(fake, filenames, self.hparams.output_path)

        return 0

    @staticmethod
    def image_with_colorbar(fig, ax, image, cmap=None, title=""):

        if cmap is None:
            pos0 = ax.imshow(image)
        else:
            pos0 = ax.imshow(image, cmap=cmap)
        ax.set_axis_off()
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax0 = divider.append_axes("right", size="5%", pad=0.05)
        tick_list = np.linspace(np.min(image), np.max(image), 5)
        cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
        cbar0.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

    @staticmethod
    def save_test_image(fake_images, filenames, savepath):

        images = np.squeeze(fake_images.to("cpu").numpy(), axis=1)

        batch_size = images.shape[0]
        for i in range(batch_size):
            filename = filenames[i] + ".png"
            image = np.squeeze(images[i])

            cropped_image = image[0: -9, 24:-24]

            cropped_image = cropped_image + np.min(cropped_image)
            cropped_image = cropped_image/np.max(cropped_image) * 255
            cropped_image = cropped_image.astype(np.uint8)

            image_filepath = os.path.join(savepath, filename)

            im = Image.fromarray(cropped_image)
            im.save(image_filepath)

    def log_images(self, condition_t, real_t, fake_t, epoch, batch_idx, filename=" ", phase=''):

        if phase == 'train':
            name = f'Train Epoch: {epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        elif phase == 'val':
            name = f'Val Epoch: {epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        elif phase == 'test':
            name = f'Test Epoch: {epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        else:
            name = f'Unknown Phase Epoch: {epoch}, Batch: {batch_idx}, filename: {filename[0]}'

        condition = np.squeeze(condition_t.to("cpu").numpy()[0, :, :, :])
        real = np.squeeze(real_t.to("cpu").numpy()[0, :, :, :])
        fake = np.squeeze(fake_t.to("cpu").numpy()[0, :, :, :])

        fig, axs = plt.subplots(1, 3)

        if real.shape[0] == 3:
            self.image_with_colorbar(fig, axs[0], np.rollaxis(condition, 0, 3), title='Condition')
            self.image_with_colorbar(fig, axs[1], np.rollaxis(real, 0, 3), cmap='gray', title='Real')
            self.image_with_colorbar(fig, axs[2], np.rollaxis(fake, 0, 3), cmap='gray', title='Fake')
        else:
            self.image_with_colorbar(fig, axs[0], np.squeeze(condition), title='Condition')
            self.image_with_colorbar(fig, axs[1], np.squeeze(real), cmap='gray', title='Real')
            self.image_with_colorbar(fig, axs[2], np.squeeze(fake), cmap='gray', title='Fake')
        fig.suptitle(name, fontsize=16)

        fig.tight_layout()
        self.logger[0].experiment.add_figure(tag=name, figure=fig)

    @staticmethod
    def add_module_specific_args(parser):
        module_specific_args = get_argparser_group(title='Dataset options', parser=parser)

        return parser
