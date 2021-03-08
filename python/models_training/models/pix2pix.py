from . import networks
from torch.nn import Module
from utils.utils import get_argparser_group
from utils.utils import str2bool

class Pix2PixModel(Module):
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
        super(Pix2PixModel, self).__init__()
        self.generator = networks.define_G(hparams.input_nc,
                                           hparams.output_nc,
                                           hparams.ngf,
                                           hparams.netG,
                                           hparams.norm,
                                           not hparams.no_dropout,
                                           hparams.init_type,
                                           hparams.init_gain,
                                           [0])  # TODO: change this to gpu list but check function

        self.discriminator = networks.define_D(hparams.input_nc + hparams.output_nc,
                                               hparams.ndf,
                                               hparams.netD,
                                               hparams.n_layers_D,
                                               hparams.norm,
                                               hparams.init_type,
                                               hparams.init_gain,
                                               [0]) # TODO: change this to gpu list but check function

    def forward(self, x):
        """x = real_A"""
        fake_B = self.generator(x)  # G(A)
        return fake_B

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
