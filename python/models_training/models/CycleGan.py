from . import networks
from torch.nn import Module
from utils.utils import get_argparser_group
from utils.utils import str2bool
import torch

class CycleGan(Module):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    def __init__(self, hparams):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CycleGan, self).__init__()

        self.hparams = hparams

        self.gpu_ids = [0]  # TODO change this with config and check with multigpus

        self.generator_A = networks.define_G(input_nc=self.hparams.input_nc,
                                             output_nc=self.hparams.output_nc,
                                             ngf=self.hparams.ngf,
                                             netG=self.hparams.netG,
                                             norm=self.hparams.norm,
                                             use_dropout=not self.hparams.no_dropout,
                                             init_type= self.hparams.init_type,
                                             init_gain=self.hparams.init_gain,
                                             gpu_ids=self.gpu_ids)  # netG_A

        self.generator_B = networks.define_G(input_nc=self.hparams.output_nc,
                                             output_nc=self.hparams.input_nc,
                                             ngf=self.hparams.ngf,
                                             netG=self.hparams.netG,
                                             norm=self.hparams.norm,
                                             use_dropout=not self.hparams.no_dropout,
                                             init_type=self.hparams.init_type,
                                             init_gain=self.hparams.init_gain,
                                             gpu_ids=self.gpu_ids)  # netG_B

        self.discriminator_A = networks.define_D(input_nc=self.hparams.output_nc,
                                                 ndf=self.hparams.ndf,
                                                 netD=self.hparams.netD,
                                                 n_layers_D=self.hparams.n_layers_D,
                                                 norm=self.hparams.norm,
                                                 init_type=self.hparams.init_type,
                                                 init_gain=self.hparams.init_gain,
                                                 gpu_ids=self.gpu_ids)  # netD_A

        self.discriminator_B = networks.define_D(input_nc=self.hparams.input_nc,
                                                 ndf=self.hparams.ndf,
                                                 netD=self.hparams.netD,
                                                 n_layers_D=self.hparams.n_layers_D,
                                                 norm=self.hparams.norm,
                                                 init_type=self.hparams.init_type,
                                                 init_gain=self.hparams.init_gain,
                                                 gpu_ids=self.gpu_ids)  # netD_B

    # real_A = condition, real_B = real_image, fake_A = fake_condition, fake_B = fake_image,
    # rec_A = reconstructed_condition, rec_B = reconstructed_image
    def forward(self, real_A, real_B):
        """x = real_A"""
        fake_B = self.generator_A(real_A)  # G_A(A)
        rec_A = self.generator_B(fake_B)  # G_B(G_A(A))

        fake_A = self.generator_B(real_B)  # G_B(B)
        rec_B = self.generator_A(fake_A)  # G_A(G_B(B))
        return fake_B, rec_A, fake_A, rec_B

    @staticmethod
    def add_model_specific_args(parser):
        module_specific_args = get_argparser_group(title='Dataset options', parser=parser)

        module_specific_args.add_argument('--ndf', default=64, type=int)
        module_specific_args.add_argument('--ngf', default=64, type=int)

        module_specific_args.add_argument('--netD', default='basic', type=str)
        module_specific_args.add_argument('--n_layers_D', default=3, type=int)
        module_specific_args.add_argument('--netG', default='unet_256', type=str)
        module_specific_args.add_argument('--norm', default='batch', type=str)
        module_specific_args.add_argument('--init_type', default='normal', type=str)
        module_specific_args.add_argument('--init_gain', default=0.02, type=float)
        module_specific_args.add_argument('--input_nc', default=3, type=int)
        module_specific_args.add_argument('--output_nc', default=3, type=int)
        module_specific_args.add_argument("--no_dropout", default=False, type=str2bool)

        return parser
