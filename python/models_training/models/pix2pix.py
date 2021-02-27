import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.utils import get_argparser_group, str2bool

# https://github.com/aniketmaurya/pytorch-gans/blob/main/pix2pix/Pix2Pix_pytorch.ipynb

def _center_crop(image, new_shape):
    h, w = image.shape[-2:]
    n_h, n_w = new_shape[-2:]
    cy, cx = int(h / 2), int(w / 2)
    xmin, ymin = cx - n_w // 2, cy - n_h // 2
    xmax, ymax = xmin + n_w, ymin + n_h
    cropped_image = image[..., ymin:ymax, xmin:xmax]
    return cropped_image

def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)

class Pix2Pix(pl.LightningModule):
    def __init__(self, hparams):

        super().__init__()
        #self.save_hyperparameters()  #TODO: maybe comment this

        self.hparams = hparams

        self.gen = Generator(self.hparams.in_channels, self.hparams.out_channels, self.hparams.hidden_channels,
                             self.hparams.depth, use_dropout=self.hparams.use_dropout, use_bn=self.hparams.use_bn)
        self.patch_gan = PatchGAN(self.hparams.in_channels + self.hparams.out_channels, hidden_channels=8)

        # initializing weights
        self.gen = self.gen.apply(_weights_init)
        self.patch_gan = self.patch_gan.apply(_weights_init)

        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()

    def forward(self, real_images, conditioned_images, step=None):
        if step == 'gen':
            return self._gen_step(real_images, conditioned_images)
        elif step == 'disc':
            return self._disc_step(real_images, conditioned_images)
        else:
            raise ValueError("In GAN model you always have to specify the step - either disc or gen")

    def configure_optimizers(self):
        gen_opt = torch.optim.Adam(self.gen.parameters(), lr=self.hparams.learning_rate)
        disc_opt = torch.optim.Adam(self.patch_gan.parameters(), lr=self.hparams.learning_rate)
        return disc_opt, gen_opt

    @staticmethod
    def add_model_specific_args(parser):
        module_specific_args = get_argparser_group(title="Model options", parser=parser)
        module_specific_args.add_argument('--in_channels', default=1, type=int)
        module_specific_args.add_argument('--out_channels', default=1, type=int)
        module_specific_args.add_argument('--depth', default=6, type=int)
        module_specific_args.add_argument('--lambda_recon', default=200, type=float)
        module_specific_args.add_argument('--hidden_channels', default=32, type=int)
        parser.add_argument("--use_bn", type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument("--use_dropout", type=str2bool, nargs='?', const=True, default=True)
        return parser


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False, use_bn=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_channels)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class UpSampleConv(nn.Module):

    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=2)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=2, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):

        x = self.upsample(x)
        x = self.conv1(x)
        skip_con_x = _center_crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x

class DownSampleConv(nn.Module):

    def __init__(self, in_channels, use_dropout=False, use_bn=False):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        if use_bn:
            self.batchnorm = nn.BatchNorm2d(in_channels * 2)
        self.use_bn = use_bn

        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

        self.conv_block1 = ConvBlock(in_channels, in_channels * 2, use_dropout, use_bn)
        self.conv_block2 = ConvBlock(in_channels * 2, in_channels * 2, use_dropout, use_bn)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.maxpool(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32, depth=6, use_dropout=True, use_bn=True):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)

        self.conv_final = nn.Conv2d(hidden_channels,
                                    out_channels,
                                    kernel_size=1)
        self.depth = depth

        self.contracting_layers = []
        self.expanding_layers = []
        self.sigmoid = nn.Sigmoid()

        # encoding/contracting path of the Generator
        for i in range(depth):

            # Adding dropout only if flag is true. Careful: bn is never used for DownSampleConv so we leave it to false
            if use_dropout:
                down_sample_conv = DownSampleConv(hidden_channels * 2 ** i,
                                                  use_dropout=(True if i < 3 else False))
            else:
                down_sample_conv = DownSampleConv(hidden_channels * 2 ** i,
                                                  use_dropout=False)
            self.contracting_layers.append(down_sample_conv)

        # Upsampling/Expanding path of the Generator
        for i in range(depth):
            upsample_conv = UpSampleConv(hidden_channels * 2 ** (i + 1), use_bn=use_bn)
            self.expanding_layers.append(upsample_conv)

        self.contracting_layers = nn.ModuleList(self.contracting_layers)
        self.expanding_layers = nn.ModuleList(self.expanding_layers)

    def forward(self, x):
        depth = self.depth
        contractive_x = []

        x = self.conv1(x)
        contractive_x.append(x)

        for i in range(depth):
            x = self.contracting_layers[i](x)
            contractive_x.append(x)

        for i in range(depth - 1, -1, -1):
            x = self.expanding_layers[i](x, contractive_x[i])
        x = self.conv_final(x)

        return self.sigmoid(x)

class PatchGAN(nn.Module):

    def __init__(self, input_channels, hidden_channels=8):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)

        # use_bn is always False in DownSampleConv
        self.contract1 = DownSampleConv(hidden_channels, use_bn=False)
        self.contract2 = DownSampleConv(hidden_channels * 2)
        self.contract3 = DownSampleConv(hidden_channels * 4)
        self.contract4 = DownSampleConv(hidden_channels * 8)
        self.final = nn.Conv2d(hidden_channels * 16, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.conv1(x)
        x1 = self.contract1(x0)
        x2 = self.contract2(x1)
        x3 = self.contract3(x2)
        x4 = self.contract4(x3)
        xn = self.final(x4)
        return xn