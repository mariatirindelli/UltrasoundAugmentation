import pytorch_lightning as pl
from utils.utils import get_argparser_group
import torch
from models import networks
from utils.utils import tensor2np_array, save_data
import os
import pytorch_ssim
import numpy as np

class GanSemiUnpairedModule(pl.LightningModule):
    def __init__(self, hparams, model, logger=None):

        super().__init__()

        self.hparams = hparams
        self.model = model

        # define loss functions
        self.criterionGAN = networks.GANLoss(hparams.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()
        # self.criterionL1 = pytorch_ssim.SSIM(window_size=11)

        self.val_criterion = pytorch_ssim.SSIM(window_size=11)
        self.output_dataset_path = os.path.join(self.hparams.output_path, "output_db")

        self.db_weights = {'CT_cropped_masked': self.hparams.ct_weight, 'convex_probe':  1 - self.hparams.ct_weight}

        if not os.path.exists(self.output_dataset_path):
            os.mkdir(self.output_dataset_path)

    def val_dataloader(self):
        tmp = self.val_dataloader()
        return tmp

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

    def _get_discriminator_loss(self, real_images, conditions, fake_images):

        fake_discriminator_input = torch.cat((conditions, fake_images), 1)
        fake_predictions = self.model.discriminator(fake_discriminator_input)
        discriminator_loss_fake = self.criterionGAN(fake_predictions, False)

        # Loss on discriminating real images
        real_discriminator_input = torch.cat((conditions, real_images), 1)
        real_predictions = self.model.discriminator(real_discriminator_input)
        discriminator_loss_real = self.criterionGAN(real_predictions, True)

        discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) * 0.5
        return discriminator_loss

    def _get_generator_adversarial_loss(self, conditions, fake_images):
        fake_discriminator_input = torch.cat((conditions, fake_images), 1)
        fake_predictions = self.model.discriminator(fake_discriminator_input)
        generator_adversarial_loss = self.criterionGAN(fake_predictions, True)
        return generator_adversarial_loss

    def update_visual_queue(self, image_dict, phase, batch_idx):

        log_every_n_batch = 50 if phase == 'train' else 20

        if phase == 'train':
            title = 'Training Result'
        elif phase == 'val':
            title = 'Validation Result'
        else:
            title = 'Result'

        if self.current_epoch % self.hparams.log_every_n_steps != 0 or batch_idx % log_every_n_batch != 0:
            return

        full_queue = self.logger.update_image_queue(image_dict = image_dict,
                                                    phase=phase,
                                                    epoch=self.current_epoch)
        if full_queue:
            self.logger.log_image_queue(phase=phase,
                                        title=title,
                                        images_keys=None)

    def training_step(self, batch_dict, batch_idx, optimizer_idx):

        num_dataset = len(batch_dict.keys())

        # Discriminator step
        if optimizer_idx == 0:

            discriminator_loss_total = 0
            for dataloader_idx, dataset_name in enumerate(batch_dict.keys()):
                real_images = batch_dict[dataset_name]['Image']  # are the US for dataloader_idx = 0, the CT otherwise
                conditions = batch_dict[dataset_name]['Label']

                fake_images = self.forward(conditions).detach()
                discriminator_loss = self._get_discriminator_loss(real_images=real_images,
                                                                  conditions=conditions,
                                                                  fake_images=fake_images)

                # Loss on discriminating fake images

                self.update_visual_queue({'condition': conditions,
                                          'real_image': real_images,
                                          'fake_image': fake_images},
                                         phase='train',
                                         batch_idx=batch_idx)

                self.log('Train discriminator loss - ' + str(dataset_name), discriminator_loss)
                discriminator_loss_total += discriminator_loss * self.db_weights[dataset_name]

            output = {'loss': discriminator_loss_total}
            self.log('Train discriminator loss total', discriminator_loss_total)
            return output

        # Generator step
        elif optimizer_idx == 1:
            generator_loss_total = 0
            for dataloader_idx, dataset_name in enumerate(batch_dict.keys()):
                real_images = batch_dict[dataset_name]['Image']  # are the US for dataloader_idx = 0, the CT otherwise
                conditions = batch_dict[dataset_name]['Label']
                # Pix2Pix has adversarial and a reconstruction loss
                fake_images = self.forward(conditions)
                generator_adversarial_loss = self._get_generator_adversarial_loss(conditions=conditions,
                                                                                  fake_images=fake_images)

                self.log('Train generator adversarial loss- ' + str(dataset_name), generator_adversarial_loss)

                generator_loss_total += generator_adversarial_loss * self.db_weights[dataset_name]

                if 'convex' in dataset_name:
                    similarity_loss = self.criterionL1(fake_images, real_images) * self.hparams.lambda_L1
                    generator_loss_total += similarity_loss * self.db_weights[dataset_name]
                    self.log('Train generator similarity loss ' + str(dataset_name), similarity_loss)

            output = {'loss': generator_loss_total}
            self.log('Train generator loss total', generator_loss_total)
            return output

    def validation_step(self, batch, batch_idx, dataloader_idx=0):

        real_images = batch['Image']  # are the US for dataloader_idx = 0, the CT otherwise
        conditions = batch['Label']
        batch_ids = batch['ImageName']

        fake_images = self.model.forward(conditions).detach()
        self.update_visual_queue({'condition': conditions,
                                  'real_image': real_images,
                                  'fake_image': fake_images},
                                 phase='val',
                                 batch_idx=batch_idx)

        if self.current_epoch > 0 and self.current_epoch % self.hparams.save_every_k_epochs == 0:
            self.save_val_epochs_results({'labels': conditions,
                                          'ct': real_images,
                                          'ultrasound': fake_images,
                                          'id': batch_ids})

        if dataloader_idx == 1:  # on the US dataset

            ssim_val = 1 - 2 * self.val_criterion(fake_images, real_images)
            self.log('Paired Val Accuracy', ssim_val)
            return {'Paired Val Accuracy': ssim_val}

        else:  # on the CT dataset

            fake_discriminator_input = torch.cat((conditions, fake_images), 1)
            fake_predictions = self.model.discriminator(fake_discriminator_input)
            generator_adversarial_loss = self.criterionGAN(fake_predictions, True)

            validation_loss = generator_adversarial_loss
            self.log('Unpaired Val Loss', validation_loss)
            return {'Unpaired Val Loss': validation_loss}

    def save_val_epochs_results(self, data_dict):

        current_save_folder = os.path.join(self.output_dataset_path, "epoch_" + str(self.current_epoch))
        if not os.path.exists(current_save_folder):
            os.mkdir(current_save_folder)

        labels_list = tensor2np_array(data_dict['labels'])
        ct_list = tensor2np_array(data_dict['ct'])
        us_list = tensor2np_array(data_dict['ultrasound'])
        id_list = data_dict['id']

        for i, _ in enumerate(labels_list):

            save_data(labels_list[i], os.path.join(current_save_folder, id_list[i]) + '_label', fmt='png', is_label=True)
            save_data(ct_list[i], os.path.join(current_save_folder, id_list[i]) + '_ct', fmt='png', is_label=False)
            save_data(us_list[i], os.path.join(current_save_folder, id_list[i]) + '_sim_us', fmt='png', is_label=False)


    @staticmethod
    def add_module_specific_args(parser):
        module_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        module_specific_args.add_argument('--gan_mode', default='vanilla', type=str)
        module_specific_args.add_argument('--beta1', default=0.5, type=float)
        module_specific_args.add_argument('--lambda_L1', default=50.0, type=float)  # was 100
        module_specific_args.add_argument('--ct_weight', default=0.5, type=float)  # was 100

        return parser
