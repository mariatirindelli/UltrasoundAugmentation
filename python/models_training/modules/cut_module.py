"""
Adapted from
https://gitlab.lrz.de/CAMP_IFL/covid19_seg/-/blob/master/modules/seg.py
See the monai docs
https://docs.monai.io/en/latest/
"""
import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim
import models.gan_models as networks
import nn_common_modules.losses as losses
from utils.utils import get_argparser_group


class CUT2D(pl.LightningModule):
    def __init__(self, hparams, *args, **kwargs):
        super(CUT2D, self).__init__()
        self.hparams = hparams

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(input_nc=hparams.input_nc,
                                      output_nc=hparams.output_nc,
                                      ngf=hparams.ngf,
                                      netG=hparams.netG,
                                      norm=hparams.normG,
                                      use_dropout=not hparams.no_dropout,
                                      init_type=hparams.init_type,
                                      init_gain=hparams.init_gain,
                                      no_antialias=hparams.no_antialias,
                                      no_antialias_up=hparams.no_antialias_up)

        self.netF = networks.define_F(input_nc=hparams.input_nc,
                                      netF=hparams.netF,
                                      norm=hparams.normG,
                                      use_dropout=not hparams.no_dropout,
                                      init_type=hparams.init_type,
                                      init_gain=hparams.init_gain,
                                      no_antialias=hparams.no_antialias,
                                      opt=hparams)

        self.netD = networks.define_D(input_nc=hparams.output_nc,
                                      ndf=hparams.ndf,
                                      netD=hparams.netD,
                                      n_layers_D=hparams.n_layers_D,
                                      norm=hparams.normD,
                                      init_type=hparams.init_type,
                                      init_gain=hparams.init_gain,
                                      no_antialias=hparams.no_antialias)

        self.nce_layers = [int(i) for i in self.hparams.nce_layers.split(',')]

        # if the example_input_array is defined, Tensorboard will automatically generate a graph of the model
        # self.example_input_array = (torch.zeros(self.hparams.batch_size, 1, *self.hparams.inshape),
        #                             torch.zeros(self.hparams.batch_size, 1, *self.hparams.inshape))

        self.criterionGAN_syn = losses.GANLoss(hparams.gan_mode)
        self.criterionNCE = []

        for _ in self.nce_layers:
            self.criterionNCE.append(losses.PatchNCELoss(hparams))

        self.criterionIdt = torch.nn.L1Loss()
        self.netF_initialized = False

        # Initializing forward variables to None
        self.real_A = None
        self.real_B = None
        self.real = None
        self.flipped_for_equivariance = False
        self.fake = None
        self.fake_B = None
        self.idt_B = None
        self.pred_real = None

        # Initializing losses to 0
        self.loss_disc_fake = torch.Tensor(0)
        self.loss_disc_real = torch.Tensor(0)
        self.loss_D = torch.Tensor(0)
        self.loss_G_GAN = torch.Tensor(0)
        self.loss_G = torch.Tensor(0)

        self.loss_NCE = torch.Tensor(0)
        self.loss_NCE_bd = torch.Tensor(0)
        self.loss_NCE_Y = torch.Tensor(0)

    def configure_optimizers(self):
        """
         2: Optimizer (configure_optimizers hook)
        see https://pytorch-lightning.readthedocs.io/en/latest/optimizers.html
        """
        hparams = self.hparams
        optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                       lr=hparams.lr,
                                       betas=(hparams.beta1, hparams.beta2))
        # merge optimizers for G and F if net F should be used
        if self.hparams.lambda_NCE:
            if not self.netF_initialized:
                raise Exception("NetF was not initialized")
            paramsF = list(self.netF.parameters())
            paramsG = list(self.netG.parameters())
            optimizer_G = torch.optim.Adam(paramsF + paramsG,
                                           lr=hparams.lr,
                                           betas=(hparams.beta1, hparams.beta2))
        else:
            optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                           lr=hparams.lr,
                                           betas=(hparams.beta1, hparams.beta2))
        return [optimizer_G, optimizer_D]

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward(self.real_A, self.real_B)  # compute fake images: G(A)
        if self.training:
            self.compute_disc_loss().backward()  # calculate gradients for D
            self.compute_gen_loss().backward()  # calculate gradients for G
            self.netF_initialized = True

        return

    def set_input(self, x):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            x (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """

        self.real_A = x['Label']
        self.real_B = x['Image']

    # 1: Forward step (forward hook), Lightning calls this inside the training loop
    def forward(self, real_A, real_B):

        torch.cuda.empty_cache()
        # self.set_input(input)
        self.real = torch.cat((real_A, real_B), dim=0) if self.hparams.nce_idt else real_A

        # Inspired by GcGAN, FastCUT is trained with flip-equivariance augmentation, where
        # the input image to the generator is horizontally flipped, and the output features
        # are flipped back before computing the PatchNCE loss
        if self.hparams.flip_equivariance:
            self.flipped_for_equivariance = self.hparams.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)

        self.fake_B = self.fake[:real_A.size(0)]
        if self.hparams.nce_idt:
            self.idt_B = self.fake[real_A.size(0):]

        return self.fake

    def toggle_optimizer(self, optimizer: optim.Optimizer, optimizer_idx: int):
        if optimizer_idx == 1:
            # update D
            self.set_requires_grad(self.netD, True)
        elif optimizer_idx == 0:
            # update G
            self.set_requires_grad(self.netD, False)

    # 4: Training Loop (this is where the magic happens)(training_step hook)
    def training_step(self, x, batch_idx, optimizer_idx):

        self.set_input(x)
        self.forward(self.real_A, self.real_B)

        torch.cuda.empty_cache()
        loss_G = None
        loss_D = None
        if optimizer_idx == 1:
            # update D
            loss = self.compute_disc_loss()

            # todo handle this better
            fake = self.fake_B.detach()
            self.logger.direct_plot({'real': self.real,
                                     'fake': fake},
                                    epoch=self.current_epoch,
                                    phase='train')
            loss_D = loss.detach()
        elif optimizer_idx == 0:
            # update G
            loss = self.compute_gen_loss()
            loss_G = loss.detach()
        else:
            raise Exception("optimizer id is out of range")

        return {'loss': loss, 'loss_G': loss_G, 'loss_D': loss_D}

    def compute_disc_loss(self) -> torch.Tensor:
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_disc_fake = self.criterionGAN_syn(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_disc_real = self.criterionGAN_syn(self.pred_real, True)
        self.loss_disc_real = loss_disc_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_disc_fake + self.loss_disc_real) * 0.5
        return self.loss_D

    def compute_gen_loss(self) -> torch.Tensor:
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B

        # First, G(A) should fake the discriminator
        if self.hparams.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            # The adversarial loss for the algorithm to also change the domain
            self.loss_G_GAN = self.criterionGAN_syn(pred_fake, True).mean() * self.hparams.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        # Lambda = 1 by default
        if self.hparams.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_nce_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = torch.tensor(0.0), torch.tensor(0.0)

        # For contrastive loss between Y and G(Y) but nce_idt is by default 0.0
        # Lambda = 1 by default
        if self.hparams.nce_idt and self.hparams.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_nce_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            self.loss_NCE_Y = torch.tensor(0.0)
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both
        return self.loss_G

    def calculate_nce_loss(self, src, tgt):
        n_layers = len(self.nce_layers)

        # Only use the encoder part of the networks
        # the encoder learns to pay attention to the commonalities between the
        # two domains, such as object parts and shapes, while being invariant
        # to the differences, such as the textures of the animals.
        feat_q = self.netG(tgt, layers=self.nce_layers, encode_only=True)

        if self.hparams.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, layers=self.nce_layers, encode_only=True)

        #  ....add the two layered MLP network that projects both input and
        #  output patch to a shared embedding space. For each layer’s features, we sample 256
        # random locations, and apply the 2-layer MLP (Net F) to acquire 256-dim final features
        feat_k_pool, sample_ids = self.netF(feat_k, self.hparams.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.hparams.num_patches, sample_ids)

        total_nce_loss = 0.0

        # Go through the feature layers
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            # Calculate patch loss in criterion, f_q is from the transformed image and f_k is from the original one
            loss = crit(f_q, f_k) * self.hparams.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # If using metrics in data parallel mode (dp), the metric update/logging should be done in the training_step_end
    # This is due to metric states else being destroyed after each forward pass, leading to wrong accumulation.
    def training_step_end(self, train_step_output):

        self.log('total loss', train_step_output['loss'], on_step=True, on_epoch=True)
        if train_step_output['loss_G'] is not None:
            self.log('loss G', train_step_output['loss_G'], on_step=True, on_epoch=True)
        if train_step_output['loss_D'] is not None:
            self.log('loss D', train_step_output['loss_D'], on_step=True, on_epoch=True)
        if self.loss_NCE_Y is not None:
            self.log('NCE Y', self.loss_NCE_Y.detach(), on_step=True, on_epoch=True)
        if self.loss_NCE is not None:
            self.log('NCE', self.loss_NCE.detach(), on_step=True, on_epoch=True)

        return {'loss': train_step_output['loss']}

    def training_epoch_end(self, train_epoch_output):
        self.train_dataloader.dataloader.dataset.update_epoch()

    # 5 Validation Loop
    def validation_step(self, x, batch_idx):
        self.set_input(x)
        self.forward(self.real_A, self.real_B)

        # todo: implement the visualization method
        self.display_3d_volumes(batch_idx, phase='Validation', global_step=self.current_epoch)
        return None

    # 6 Test Loop
    def test_step(self, x, batch_idx):
        self.set_input(x)
        # forward pass
        self.forward(self.real_A, self.real_B)
        # calculate loss
        loss = self.compute_gen_loss()
        return {'test_loss': loss}

    def test_step_end(self, test_step_output):
        self.log('Loss G', test_step_output['test_loss'], on_step=True, on_epoch=True)
        return {'test_loss': test_step_output['test_loss']}

    @staticmethod
    def set_requires_grad(nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def add_module_specific_args(parser):
        module_parser = get_argparser_group(title="Module options", parser=parser)
        module_parser = CUT2D.add_gan_args(module_parser)
        module_parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        module_parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        module_parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        module_parser.add_argument('--nce_idt', action='store_true', default=False,
                                   help='use NCE loss for identity mapping: NCE(G(Y), Y))')

        module_parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which '
                                                                                         'layers')
        module_parser.add_argument('--netF', type=str, default='mlp_sample',
                                   choices=['sample', 'reshape', 'mlp_sample'],
                                   help='how to down-sample the feature map')
        module_parser.add_argument('--netF_nc', type=int, default=256)
        module_parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        module_parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        module_parser.add_argument('--flip_equivariance', action='store_true', default=False,
                                   help="Enforce flip-equivariance as additional regularization. "
                                        "It's used by FastCUT, but not CUT")
        module_parser.add_argument('--nce_includes_all_negatives_from_minibatch', action='store_true',
                                   help='(used for single image translation) If True, include the negatives from the '
                                        'other samples of the minibatch when computing the contrastive loss. '
                                        'Please see models/patchnce.py for more details.')

        module_parser.set_defaults(pool_size=0, dataset_mode='volume')  # no image pooling

        module_parser.set_defaults(nce_idt=True, lambda_NCE=1.0)

        return parser

    @staticmethod
    def add_gan_args(parser):
        # model parameters
        parser.add_argument('--input_nc', type=int, default=1,
                            help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1,
                            help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--load_init_models', type=str, default=None, help='the path to load the saved models from')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--netD', type=str, default='basic',
                            choices=['basic', 'n_layers', 'pixel', 'patch', 'tilestylegan2', 'stylegan2'],
                            help='specify discriminator architecture. The basic model is a 70x70 PatchGAN. '
                                 'n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_4blocks',
                            choices=['global', 'NormalNet', 'local', 'encoder', 'resnet_7blocks', 'resnet_9blocks',
                                     'resnet_4blocks', 'unet_3D', 'resnet_6blocks', 'unet_256', 'unet_128',
                                     'unet_small', 'stylegan2', 'unet_small', 'smallstylegan2', 'resnet_cat'],
                            help='specify generator architecture')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'],
                            help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'],
                            help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier',
                            choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--no_antialias', action='store_true',
                            help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true',
                            help='if specified, use [upconv(learned filter)] instead of '
                                 '[upconv(hard-coded [1,3,3,1] filter), conv]')

        # training parameters
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan',
                            help='the type of GAN objective. [vanilla| lsgan | wgangp]. '
                                 'vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=1,
                            help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        # for discriminators
        parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true',
                            help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true',
                            help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--no_lsgan', action='store_true',
                            help='do *not* use least square GAN, if false, use vanilla GAN')
        return parser

    def display_3d_volumes(self,batch_idx, phase, global_step=0):
        # todo: change this with my logging

        # axs, fig = tensorboard_utils.init_figure(3, 4)
        # tensorboard_utils.set_axs_attribute(axs)
        # tensorboard_utils.fill_subplots(self.real_A.cpu(), axs=axs[0, :], img_name='A')
        # tensorboard_utils.fill_subplots(self.fake_B.detach().cpu(), axs=axs[1, :], img_name='fake')
        # tensorboard_utils.fill_subplots(self.real_B.cpu(), axs=axs[2, :], img_name='B')
        # tensorboard_utils.fill_subplots(self.idt_B.cpu(), axs=axs[3, :], img_name='idt_B')
        # tag = f'{phase} - epoch: {global_step}'
        # self.logger.experiment[0].add_figure(tag=tag, figure=fig, global_step=batch_idx)
        pass
