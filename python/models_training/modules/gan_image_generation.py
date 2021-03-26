import pytorch_lightning as pl
from utils.utils import get_argparser_group
import torch
from models import networks
from utils.image_logging import log_images
from utils.utils import tensor2np_array, save_data
import os
from utils.utils import str2bool
from torchgeometry.image.gaussian import gaussian_blur

class GanImageGeneration(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):

        super().__init__()

        self.hparams = hparams
        self.model = model

        # define loss functions
        self.criterionGAN = networks.GANLoss(hparams.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        self.t_logger = logger if logger is not None else None

    def configure_optimizers(self):

        discriminator_optimizer = torch.optim.Adam(self.model.discriminator.parameters(),
                                                   lr=self.hparams.learning_rate,
                                                   betas=(self.hparams.beta1, 0.999))
        generator_optimizer = torch.optim.Adam(self.model.generator.parameters(),
                                               lr=self.hparams.learning_rate,
                                               betas=(self.hparams.beta1, 0.999))

        return discriminator_optimizer, generator_optimizer

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx, optimizer_idx):

        real_images, conditions, _ = batch

        # Discriminator step
        if optimizer_idx == 0:

            # Loss on discriminating fake images
            fake_images = self.forward(conditions).detach()
            fake_discriminator_input = torch.cat((conditions, fake_images), 1)
            fake_predictions = self.model.discriminator(fake_discriminator_input)
            discriminator_loss_fake = self.criterionGAN(fake_predictions, False)

            # Loss on discriminating real images
            real_discriminator_input = torch.cat((conditions, real_images), 1)
            real_predictions = self.model.discriminator(real_discriminator_input)
            discriminator_loss_real = self.criterionGAN(real_predictions, True)

            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) * 0.5

            if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx == 0:
                fig, title = log_images(epoch=self.current_epoch,
                                        batch_idx=batch_idx,
                                        image_list=[conditions, real_images, fake_images],
                                        image_name_list=['condition', 'real image', 'fake image'],
                                        cmap_list=['hot', 'gray', 'gray'],
                                        filename=[''],
                                        phase='train',
                                        clim=(-1, 1))

                self.t_logger[0].experiment.add_figure(tag=title, figure=fig)

            self.log('Train discriminator loss', discriminator_loss)

            output = {'loss': discriminator_loss}
            return output

        # Generator step
        elif optimizer_idx == 1:
            # Pix2Pix has adversarial and a reconstruction loss
            fake_images = self.forward(conditions)
            fake_discriminator_input = torch.cat((conditions, fake_images), 1)
            fake_predictions = self.model.discriminator(fake_discriminator_input)
            generator_adversarial_loss = self.criterionGAN(fake_predictions, True)

            if not self.hparams.L1_blur:
                loss_L1 = self.criterionL1(fake_images, real_images) * self.hparams.lambda_L1
            else:
                blur_real = gaussian_blur(real_images, kernel_size=(23, 23), sigma=(6, 6))
                blur_fake = gaussian_blur(fake_images, kernel_size=(23, 23), sigma=(6, 6))

                loss_L1 = self.criterionL1(blur_fake, blur_real) * self.hparams.lambda_L1

            generator_loss = generator_adversarial_loss + loss_L1

            self.log('Train generator loss_L1', loss_L1)
            self.log('Train generator adversarial loss', generator_adversarial_loss)
            self.log('Train generator loss', generator_loss)

            output = {'loss': generator_loss}
            return output

    def validation_step(self, batch, batch_idx):

        real_images, conditions, _ = batch
        fake_images = self.model.generator(conditions).detach()

        if self.current_epoch % self.hparams.log_every_n_steps == 0 and batch_idx % 100 == 0:
            fig, title = log_images(epoch=self.current_epoch,
                                    batch_idx=batch_idx,
                                    image_list=[conditions, real_images, fake_images],
                                    image_name_list=['condition', 'real image', 'fake image'],
                                    cmap_list=['hot', 'gray', 'gray'],
                                    filename=[''],
                                    phase='val',
                                    clim=(-1, 1))

            self.t_logger[0].experiment.add_figure(tag=title, figure=fig)

        return {'val_loss': -1}

    def test_step(self, batch, batch_idx):

        real_images, conditions, filenames = batch
        fake_images = self.model.generator(conditions).detach()

        fig, title = log_images(epoch=self.current_epoch,
                                batch_idx=batch_idx,
                                image_list=[conditions, real_images, fake_images],
                                image_name_list=['condition', 'real image', 'fake image'],
                                cmap_list=['hot', 'gray', 'gray'],
                                filename=[''],
                                phase='test',
                                clim=(-1, 1))

        self.t_logger[0].experiment.add_figure(tag=title, figure=fig)

        np_conditions = tensor2np_array(conditions.cpu())
        np_fake_images = tensor2np_array(fake_images.cpu())

        self.save_batch(conditions=np_conditions,
                        fake_images=np_fake_images,
                        filenames=filenames,
                        fmt='npy')

        return {'test_loss': -1}

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
        module_specific_args.add_argument('--lambda_L1', default=50.0, type=float)  # was 100
        module_specific_args.add_argument('--L1_blur', default=False, type=str2bool)  # was 100

        return parser
