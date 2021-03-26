import pytorch_lightning as pl
from utils.utils import get_argparser_group
import torch
from models import networks
from utils.image_logging import log_images
from utils.utils import tensor2np_array, save_data
import os
import itertools
from utils.image_pool import ImagePool

class CUTImageGeneration(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):

        super().__init__()
        self.automatic_optimization = False
        self.hparams = hparams

        if self.hparams.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
            assert (self.hparams.input_nc == self.hparams.output_nc)

        self.model = model

        self.t_logger = logger

        # Setting image pools
        self.fake_A_pool = ImagePool(self.hparams.pool_size)  # create image buffer to store previously generated images
        self.fake_B_pool = ImagePool(self.hparams.pool_size)  # create image buffer to store previously generated images

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

        return generator_optimizer, discriminator_optimizer

    def forward(self, conditions, real_images):
        return self.model.forward(conditions, real_images)

    def compute_discriminator_loss(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def compute_generator_loss(self, real_A, real_B, fake_A, fake_B, rec_A, rec_B):
        """Calculate the loss for generators G_A and G_B"""

        # Identity loss
        if self.hparams.lambda_identity > 0:

            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = self.model.generator_A(real_B)
            loss_idt_A = self.criterionIdt(idt_A, real_B) * self.hparams.lambda_B * self.hparams.lambda_identity

            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = self.model.generator_B(real_A)
            loss_idt_B = self.criterionIdt(idt_B, real_A) * self.hparams.lambda_A * self.hparams.lambda_identity
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        loss_G_A = self.criterionGAN(self.model.discriminator_A(fake_B), True)

        # GAN loss D_B(G_B(B))
        loss_G_B = self.criterionGAN(self.model.discriminator_B(fake_A), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = self.criterionCycle(rec_A, real_A) * self.hparams.lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = self.criterionCycle(rec_B, real_B) * self.hparams.lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        return loss_G

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # real_A = condition
    # real_B = real_image
    def training_step(self, batch, batch_idx, optimizer_idx):

        g_opt, d_opt = self.optimizers()

        real_A, real_B, _, _ = batch  # image, label

        # forward
        fake_B, rec_A, fake_A, rec_B = self.forward(real_A, real_B)  # fake_B, rec_A, fake_A, rec_B

        # GENERATOR STEP
        self.set_requires_grad([self.model.discriminator_A, self.model.discriminator_B], False)  # Ds require no gradients when optimizing Gs
        g_opt.zero_grad()  # set G_A and G_B's gradients to zero
        generator_loss = self.compute_generator_loss(real_A, real_B, fake_A, fake_B, rec_A, rec_B)
        self.manual_backward(generator_loss)
        g_opt.step()  # update G_A and G_B's weights

        # DISCRIMINATOR STEP
        self.set_requires_grad([self.model.discriminator_A, self.model.discriminator_B], True)
        d_opt.zero_grad()  # set D_A and D_B's gradients to zero
        fake_B = self.fake_B_pool.query(fake_B)
        loss_D_A = self.compute_discriminator_loss(self.model.discriminator_A, real_B, fake_B)
        self.manual_backward(loss_D_A)

        # 2. self.backward_D_B()  # calculate gradients for D_B
        fake_A = self.fake_A_pool.query(fake_A)
        loss_D_B = self.compute_discriminator_loss(self.model.discriminator_B, real_A, fake_A)
        self.manual_backward(loss_D_B)
        d_opt.step()

        if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx % 50 == 0:
            figs, titles = log_images(epoch=self.current_epoch,
                                      batch_idx=batch_idx,
                                      image_list=[real_A, real_B, fake_B],
                                      image_name_list=['condition', 'real image', 'fake image'],
                                      cmap_list=['hot', 'gray', 'gray'],
                                      filename=[''],
                                      phase='train',
                                      clim=(-1, 1))

            self.t_logger[-1].log_image(figs, titles, "Training Results")

        self.t_logger[-1].log_metrics({'g_loss': generator_loss, 'd_loss_A': loss_D_A, 'd_loss_B': loss_D_B})

    def validation_step(self, batch, batch_idx):

        real_A, real_B, _, _ = batch  # image, label

        # forward
        fake_B, rec_A, fake_A, rec_B = self.forward(real_A, real_B)  # fake_B, rec_A, fake_A, rec_B

        if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx % 30 == 0:
            figs, titles = log_images(epoch=self.current_epoch,
                                      batch_idx=batch_idx,
                                      image_list=[real_A, real_B, fake_B],
                                      image_name_list=['condition', 'real image', 'fake image'],
                                      cmap_list=['hot', 'gray', 'gray'],
                                      filename=[''],
                                      phase='train',
                                      clim=(-1, 1))

            self.t_logger[-1].log_image(figs, titles, "Training Results")

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
        module_specific_args.add_argument('--lambda_identity', default=0.5, type=float)
        module_specific_args.add_argument('--lambda_A', default=10.0, type=float)
        module_specific_args.add_argument('--lambda_B', default=10.0, type=float)
        module_specific_args.add_argument('--pool_size', default=4, type=int)

        return parser
