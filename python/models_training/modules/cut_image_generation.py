import pytorch_lightning as pl
from utils.utils import get_argparser_group
import torch
from models import networks
from utils.image_logging import log_images
from utils.utils import tensor2np_array, save_data
import os
import itertools
from utils.image_pool import ImagePool

class GanImageGeneration(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):

        super().__init__()

        if self.hparams.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            assert (self.hparams.input_nc == self.hparams.output_nc)

        self.hparams = hparams
        self.model = model

        self.logger = logger[0] if logger is not None else None

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']  # TODO: maybe not needed

        # Setting image pools
        self.fake_A_pool = ImagePool(self.hparams.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(self.hparams.pool_size)  # create image buffer to store previously generated images


        # define loss functions
        self.criterionGAN = networks.GANLoss(hparams.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        # define loss functions
        self.criterionGAN = networks.GANLoss(self.hparams.gan_mode).to(self.device)  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()

    def configure_optimizers(self):

        discriminator_optimizer = torch.optim.Adam(
            params=itertools.chain(self.model.discriminator_A.parameters(), self.model.discriminator_B.parameters()),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, 0.999))

        generator_optimizer = torch.optim.Adam(
            params=itertools.chain(self.model.generator_A.parameters(), self.model.generator_B.parameters()),
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, 0.999))

        return discriminator_optimizer, generator_optimizer

    def forward(self, conditions, real_images):
        return self.model.forward(conditions, real_images)

    def discriminator_loss_basic(self):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def training_step(self, batch, batch_idx, optimizer_idx):

        real_images, conditions, _, _ = batch

        # forward
        fake_image, reconstructed_condition, fake_condition, reconstructed_image = self.forward(conditions, real_images)  # fake_B, rec_A, fake_A, rec_B

        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.forward()      # compute fake images and reconstruction images.

        # Discriminator step
        if optimizer_idx == 0:
            # self.set_requires_grad([self.netD_A, self.netD_B], True)
            # self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
            # self.backward_D_A()  # calculate gradients for D_A
            # self.backward_D_B()  # calculate graidents for D_B
            # self.optimizer_D.step()  # update D_A and D_B's weights

            # 1. self.backward_D_A()  # calculate gradients for D_A
            fake_B = self.fake_B_pool.query(self.fake_B)
            discriminatorA_loss = self.backward_D_basic(self.netD_A, self.real_B, fake_B)


            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_D.step()  # update D_A and D_B's weights

        # generator step
        elif optimizer_idx == 1:

            # G_A and G_B
            self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs

            self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
            self.backward_G()             # calculate gradients for G_A and G_B
            self.optimizer_G.step()       # update G_A and G_B's weights





    def save_batch(self, conditions, fake_images, real_images=None, filenames="", fmt='npy'):
        if real_images is None:
            real_images = [None for _ in conditions]

        for condition, fake_image, real_images, filename in zip(conditions, fake_images, real_images, filenames):
            condition_filename = os.path.join(self.hparams.output_path, filename + "_label")
            real_filename = os.path.join(self.hparams.output_path, filename + "_real")
            fake_filename = os.path.join(self.hparams.output_path, filename + "_fake")

            save_data(condition, condition_filename, fmt=fmt)
            save_data(real_images, real_filename, fmt=fmt)
            save_data(fake_image, fake_filename, fmt=fmt)

    @staticmethod
    def add_module_specific_args(parser):
        module_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        module_specific_args.add_argument('--gan_mode', default='vanilla', type=str)
        module_specific_args.add_argument('--beta1', default=0.5, type=float)
        module_specific_args.add_argument('--lambda_identity', default=50.0, type=float)  # was 100  # TODO: check this
        module_specific_args.add_argument('--pool_size', default=4, type=int)  # was 100  # TODO: check this

        return parser
