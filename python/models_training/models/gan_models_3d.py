# ###############################################################################
# # Code originally developed by "Amos Newswanger" (neoamos). Check this repo:
# # https://github.com/neoamos/3d-pix2pix-CycleGAN/
# ###############################################################################
#
# from collections import OrderedDict
# from torch.autograd import Variable
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import init
# import functools
# import torch.nn.functional as F
# import torch.utils.checkpoint as cp
#
#
# ###############################################################################
# # Functions
# ###############################################################################
#
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     elif classname.find('BatchNorm3d') != -1:
#         m.weight.data.normal_(1.0, 0.02)
#         m.bias.data.fill_(0)
#
#
# def get_filter(filt_size=3):
#     if filt_size == 1:
#         a = np.array([1., ])
#     elif filt_size == 2:
#         a = np.array([1., 1.])
#     elif filt_size == 3:
#         a = np.array([[1., 2., 1.], [1., 2., 1.], [1., 2., 1.]])
#     elif filt_size == 4:
#         a = np.array([[1., 3., 3., 1.], [1., 3., 3., 1.], [1., 3., 3., 1.], [1., 3., 3., 1.]])
#     elif filt_size == 5:
#         a = np.array([[1., 4., 6., 4., 1.], [1., 4., 6., 4., 1.], [1., 4., 6., 4., 1.], [1., 4., 6., 4., 1.],
#                       [1., 4., 6., 4., 1.]])
#     elif filt_size == 6:
#         a = np.array([[1., 5., 10., 10., 5., 1.], [1., 5., 10., 10., 5., 1.], [1., 5., 10., 10., 5., 1.],
#                       [1., 5., 10., 10., 5., 1.], [1., 5., 10., 10., 5., 1.], [1., 5., 10., 10., 5., 1.]])
#     elif filt_size == 7:
#         a = np.array([[1., 6., 15., 20., 15., 6., 1.], [1., 6., 15., 20., 15., 6., 1.], [1., 6., 15., 20., 15., 6., 1.],
#                       [1., 6., 15., 20., 15., 6., 1.], [1., 6., 15., 20., 15., 6., 1.], [1., 6., 15., 20., 15., 6., 1.],
#                       [1., 6., 15., 20., 15., 6., 1.]])
#
#     if filt_size == 3:
#         filt = torch.Tensor(a[:, None] * a[None, :])
#     else:
#         filt = torch.Tensor(a[:, None] * a[None, :])
#         filt = filt[0:3, :, :]
#
#     filt = filt / torch.sum(filt)
#
#     return filt
#
#
# def get_pad_layer(pad_type):
#     if (pad_type in ['refl', 'reflect']):
#         PadLayer = nn.ReplicationPad3d
#     elif (pad_type in ['repl', 'replicate']):
#         PadLayer = nn.ReplicationPad3d
#     elif (pad_type == 'zero'):
#         PadLayer = nn.ConstantPad3d(1, value=0)
#     else:
#         print('Pad type [%s] not recognized' % pad_type)
#     return PadLayer
#
#
# class Identity(nn.Module):
#     def forward(self, x):
#         return x
#
#
# def get_norm_layer(norm_type='instance'):
#     """Return a normalization layer
#
#     Parameters:
#         norm_type (str) -- the name of the normalization layer: batch | instance | none
#
#     For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
#     For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
#     """
#     if norm_type == 'batch':
#         norm_layer = functools.partial(nn.BatchNorm3d, affine=True, track_running_stats=True)
#     elif norm_type == 'instance':
#         norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=False)
#     elif norm_type == 'none':
#         def norm_layer(x):
#             return Identity()
#     else:
#         raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
#     return norm_layer
#
#
# def init_weights3d(net, init_type='normal', init_gain=0.02, debug=False):
#     """Initialize network weights.
#
#     Parameters:
#         net (network)   -- network to be initialized
#         init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
#
#     We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
#     work better for some applications. Feel free to try yourself.
#     """
#
#     def init_func(m):  # define the initialization function
#         classname = m.__class__.__name__
#         if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
#             if debug:
#                 print(classname)
#             if init_type == 'normal':
#                 init.normal_(m.weight.data, 0.0, init_gain)
#             elif init_type == 'xavier':
#                 init.xavier_normal_(m.weight.data, gain=init_gain)
#             elif init_type == 'kaiming':
#                 init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
#             elif init_type == 'orthogonal':
#                 init.orthogonal_(m.weight.data, gain=init_gain)
#             else:
#                 raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
#             if hasattr(m, 'bias') and m.bias is not None:
#                 init.constant_(m.bias.data, 0.0)
#         elif classname.find(
#                 'BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
#             init.normal_(m.weight.data, 1.0, init_gain)
#             init.constant_(m.bias.data, 0.0)
#
#     net.apply(init_func)  # apply the initialization function <init_func>
#
#
# def init_net(net, init_type='normal', init_gain=0.02, debug=False, initialize_weights=True):
#     """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
#     Parameters:
#         net (network)      -- the network to be initialized
#         init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
#         gain (float)       -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#
#     Return an initialized network.
#     """
#     if initialize_weights:
#         init_weights3d(net, init_type, init_gain=init_gain, debug=debug)
#     return net
#
#
# class Generator(nn.Module):
#     def __init__(self, hparams):
#         super(Generator, self).__init__()
#         self.model = define_G(input_nc=hparams.input_nc,
#                               output_nc=hparams.output_nc,
#                               ngf=hparams.ngf,
#                               netG=hparams.netG,
#                               norm=hparams.norm,
#                               use_dropout=hparams.use_dropout,
#                               init_type=hparams.init_type,
#                               init_gain=hparams.init_gain,
#                               no_antialias=hparams.no_antialias,
#                               no_antialias_up=hparams.no_antialias_up,
#                               is_seg_net=hparams.is_seg_net,
#                               n_downsample_global=hparams.n_downsample_global,
#                               n_blocks_local=hparams.n_blocks_local,
#                               n_blocks_global=hparams.n_blocks_global,
#                               n_local_enhancers=hparams.n_local_enhancers)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
#              no_antialias=False, no_antialias_up=False, is_seg_net=False, n_downsample_global=3, n_blocks_local=3,
#              n_blocks_global=9,n_local_enhancers=1) -> nn.Module:
#     """Create a generator
#
#     Parameters:
#         input_nc (int) -- the number of channels in input images
#         output_nc (int) -- the number of channels in output images
#         ngf (int) -- the number of filters in the last conv layer
#         netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
#         norm (str) -- the name of normalization layers used in the network: batch | instance | none
#         use_dropout (bool) -- if use dropout layers.
#         init_type (str)    -- the name of our initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#
#     Returns a generator
#
#     Our current implementation provides two types of generators:
#         U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
#         The original U-Net paper: https://arxiv.org/abs/1505.04597
#
#         Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
#         Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
#         We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
#
#
#     The generator has been initialized by <init_net>. It uses RELU for non-linearity.
#     """
#     net = None
#     if is_seg_net:
#         last_layer = None
#         # TODO: softmax as last layer for resnet is not implemented
#     else:
#         last_layer = nn.Tanh()
#     norm_layer = get_norm_layer(norm_type=norm)
#     if netG == 'resnet_9blocks':
#         net = ResnetGenerator3d(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
#                                 no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=9)
#     elif netG == 'resnet_6blocks':
#         net = ResnetGenerator3d(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
#                                 no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6)
#     elif netG == 'resnet_4blocks':
#         net = ResnetGenerator3d(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
#                                 no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=4)
#     elif netG == 'unet_128':
#         net = UnetGenerator3d(input_nc, output_nc, 7, ngf, norm_layer=norm_layer,
#                               use_dropout=use_dropout, last_layer=last_layer, is_seg_net=is_seg_net)
#     elif netG == 'unet_small':
#         net = UnetGenerator3d(input_nc, output_nc, 4, ngf, norm_layer=norm_layer,
#                               use_dropout=use_dropout, last_layer=last_layer, is_seg_net=is_seg_net)
#     elif netG == 'unet_256':
#         net = UnetGenerator3d(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
#                               use_dropout=use_dropout, last_layer=last_layer, is_seg_net=is_seg_net)
#     elif netG == 'unet_3D':
#         net = RegGenerator3D(input_nc, 8, norm_layer=norm_layer, use_dropout=False)
#     elif netG == 'resnet_cat':
#         n_blocks = 5
#         net = G_Resnet3d(input_nc, output_nc, 0, num_downs=2, n_res=n_blocks - 4, ngf=ngf, norm='inst', nl_layer='relu')
#     elif netG == 'global':
#         net = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
#     elif netG == 'local':
#         net = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
#                                   n_local_enhancers, n_blocks_local, norm_layer)
#     elif netG == 'encoder':
#         net = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
#     else:
#         raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
#     return init_net(net=net, init_type=init_type, init_gain=init_gain)
#
#
# def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02,
#              no_antialias=False, opt=None) -> nn.Module:
#     if netF == 'global_pool':
#         net = PoolingF3d()
#     elif netF == 'reshape':
#         net = ReshapeF3d()
#     elif netF == 'sample':
#         net = PatchSampleF3d(use_mlp=False, init_type=init_type, init_gain=init_gain, nc=opt.netF_nc)
#     elif netF == 'mlp_sample':
#         net = PatchSampleF3d(use_mlp=True, init_type=init_type, init_gain=init_gain, nc=opt.netF_nc)
#     elif netF == 'strided_conv':
#         net = StridedConvF3d(init_type=init_type, init_gain=init_gain)
#     else:
#         raise NotImplementedError('projection model name [%s] is not recognized' % netF)
#     return init_net(net, init_type, init_gain)
#
#
# def define_D_HD(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False):
#     norm_layer = get_norm_layer(norm_type=norm)
#     netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
#     netD.apply(weights_init)
#     return netD
#
#
# def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02,
#              no_antialias=False, use_sigmoid=False) -> nn.Module:
#     """Create a discriminator
#
#     Parameters:
#         input_nc (int)     -- the number of channels in input images
#         ndf (int)          -- the number of filters in the first conv layer
#         netD (str)         -- the architecture's name: basic | n_layers | pixel
#         n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
#         norm (str)         -- the type of normalization layers used in the network.
#         init_type (str)    -- the name of the initialization method.
#         init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
#         gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
#
#     Returns a discriminator
#
#     Our current implementation provides three types of discriminators:
#         [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
#         It can classify whether 70×70 overlapping patches are real or fake.
#         Such a patch-level discriminator architecture has fewer parameters
#         than a full-image discriminator and can work on arbitrarily-sized images
#         in a fully convolutional fashion.
#
#         [n_layers]: With this mode, you can specify the number of conv layers in the discriminator
#         with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)
#
#         [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
#         It encourages greater color diversity but has no effect on spatial statistics.
#
#     The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
#     """
#
#     net = None
#     norm_layer = get_norm_layer(norm_type=norm)
#
#     if netD == 'basic':  # default PatchGAN classifier
#         net = NLayerDiscriminator3d(input_nc, ndf, n_layers=3, norm_layer=norm_layer,
#                                     no_antialias=no_antialias, use_sigmoid=use_sigmoid)
#     elif netD == 'n_layers':
#         net = NLayerDiscriminator3d(input_nc, ndf, n_layers_D, norm_layer=norm_layer,
#                                     no_antialias=no_antialias, use_sigmoid=use_sigmoid)
#     elif netD == 'pixel':  # classify if each pixel is real or fake
#         net = PixelDiscriminator3d(input_nc, ndf, norm_layer=norm_layer)
#     # elif 'stylegan2' in netD:
#     #     net = StyleGAN2Discriminator(input_nc, ndf, n_layers_D, no_antialias=no_antialias, opt=opt)
#     else:
#         raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
#     return init_net(net, init_type, init_gain)
#
#
# ##############################################################################
# # Classes
# ##############################################################################
#
# class MultiscaleDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
#                  use_sigmoid=False, num_D=3, getIntermFeat=False):
#         super(MultiscaleDiscriminator, self).__init__()
#         self.num_D = num_D
#         self.n_layers = n_layers
#         self.getIntermFeat = getIntermFeat
#
#         for i in range(num_D):
#             netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
#             if getIntermFeat:
#                 for j in range(n_layers + 2):
#                     setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
#             else:
#                 setattr(self, 'layer' + str(i), netD.model)
#
#         self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
#
#     def singleD_forward(self, model, input):
#         if self.getIntermFeat:
#             result = [input]
#             for i in range(len(model)):
#                 result.append(model[i](result[-1]))
#             return result[1:]
#         else:
#             return [model(input)]
#
#     def forward(self, input):
#         num_D = self.num_D
#         result = []
#         input_downsampled = input
#         for i in range(num_D):
#             if self.getIntermFeat:
#                 model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
#                          range(self.n_layers + 2)]
#             else:
#                 model = getattr(self, 'layer' + str(num_D - 1 - i))
#             result.append(self.singleD_forward(model, input_downsampled))
#             if i != (num_D - 1):
#                 input_downsampled = self.downsample(input_downsampled)
#         return result
#
# # Defines the PatchGAN discriminator with the specified arguments.
# class NLayerDiscriminator(nn.Module):
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, getIntermFeat=False):
#         super(NLayerDiscriminator, self).__init__()
#         self.getIntermFeat = getIntermFeat
#         self.n_layers = n_layers
#
#         kw = 4
#         padw = int(np.ceil((kw - 1.0) / 2))
#         sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]
#
#         nf = ndf
#         for n in range(1, n_layers):
#             nf_prev = nf
#             nf = min(nf * 2, 512)
#             sequence += [[
#                 nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
#                 norm_layer(nf), nn.LeakyReLU(0.2, True)
#             ]]
#
#         nf_prev = nf
#         nf = min(nf * 2, 512)
#         sequence += [[
#             nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
#             norm_layer(nf),
#             nn.LeakyReLU(0.2, True)
#         ]]
#
#         sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
#
#         if use_sigmoid:
#             sequence += [[nn.Sigmoid()]]
#
#         if getIntermFeat:
#             for n in range(len(sequence)):
#                 setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
#         else:
#             sequence_stream = []
#             for n in range(len(sequence)):
#                 sequence_stream += sequence[n]
#             self.model = nn.Sequential(*sequence_stream)
#
#     def forward(self, input):
#         if self.getIntermFeat:
#             res = [input]
#             for n in range(self.n_layers + 2):
#                 model = getattr(self, 'model' + str(n))
#                 res.append(model(res[-1]))
#             return res[1:]
#         else:
#             return self.model(input)
#
#
# class Encoder3d(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm3d):
#         super(Encoder3d, self).__init__()
#         self.output_nc = output_nc
#
#         model = [nn.ReplicationPad3d(3), nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0),
#                  norm_layer(ngf), nn.ReLU(True)]
#         ### downsample
#         for i in range(n_downsampling):
#             mult = 2**i
#             model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
#                       norm_layer(ngf * mult * 2), nn.ReLU(True)]
#
#         ### upsample
#         for i in range(n_downsampling):
#             mult = 2**(n_downsampling - i)
#             model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
#                        norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
#
#         model += [nn.ReplicationPad3d(3), nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input, inst):
#         outputs = self.model(input)
#
#         # instance-wise average pooling
#         outputs_mean = outputs.clone()
#         inst_list = np.unique(inst.cpu().numpy().astype(int))
#         for i in inst_list:
#             for b in range(input.size()[0]):
#                 indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
#                 for j in range(self.output_nc):
#                     output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
#                     mean_feat = torch.mean(output_ins).expand_as(output_ins)
#                     outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
#         return outputs_mean
#
#
# class LocalEnhancer3d(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
#                  n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm3d, padding_type='reflect'):
#         super(LocalEnhancer3d, self).__init__()
#         self.n_local_enhancers = n_local_enhancers
#
#         ###### global generator model #####
#         ngf_global = ngf * (2 ** n_local_enhancers)
#         model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
#                                        norm_layer).model
#         model_global = [model_global[i] for i in
#                         range(len(model_global) - 3)]  # get rid of final convolution layers
#         self.model = nn.Sequential(*model_global)
#
#         ###### local enhancer layers #####
#         for n in range(1, n_local_enhancers + 1):
#             ### downsample
#             ngf_global = ngf * (2 ** (n_local_enhancers - n))
#             model_downsample = [nn.ReplicationPad3d(3), nn.Conv3d(input_nc, ngf_global, kernel_size=7, padding=0),
#                                 norm_layer(ngf_global), nn.ReLU(True),
#                                 nn.Conv3d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
#                                 norm_layer(ngf_global * 2), nn.ReLU(True)]
#             ### residual blocks
#             model_upsample = []
#             for i in range(n_blocks_local):
#                 model_upsample += [ResnetBlock3d(ngf=ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]
#
#             ### upsample
#             model_upsample += [
#                 nn.ConvTranspose3d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
#                 norm_layer(ngf_global), nn.ReLU(True)]
#
#             ### final convolution
#             if n == n_local_enhancers:
#                 model_upsample += [nn.ReplicationPad3d(3), nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0),
#                                    nn.Tanh()]
#
#             setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))
#             setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))
#
#         self.downsample = nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False)
#
#     def forward(self, input):
#         ### create input pyramid
#         input_downsampled = [input]
#         for i in range(self.n_local_enhancers):
#             input_downsampled.append(self.downsample(input_downsampled[-1]))
#
#         ### output at coarest level
#         output_prev = self.model(input_downsampled[-1])
#         ### build up one layer at a time
#         for n_local_enhancers in range(1, self.n_local_enhancers + 1):
#             model_downsample = getattr(self, 'model' + str(n_local_enhancers) + '_1')
#             model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')
#             input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
#             output_prev = model_upsample(model_downsample(input_i) + output_prev)
#         return output_prev
#
#
# class Normalize(nn.Module):
#
#     def __init__(self, power=2):
#         super(Normalize, self).__init__()
#         self.power = power
#
#     def forward(self, x):
#         norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
#         out = x.div(norm + 1e-7)
#         return out
#
#
# class PoolingF3d(nn.Module):
#     def __init__(self):
#         super(PoolingF3d, self).__init__()
#         model = [nn.AdaptiveMaxPool3d(1)]
#         self.model = nn.Sequential(*model)
#         self.l2norm = Normalize(2)
#
#     def forward(self, x):
#         return self.l2norm(self.model(x))
#
#
# class ReshapeF3d(nn.Module):
#     def __init__(self):
#         super(ReshapeF3d, self).__init__()
#         model = [nn.AdaptiveAvgPool3d(4)]
#         self.model = nn.Sequential(*model)
#         self.l2norm = Normalize(2)
#
#     def forward(self, x):
#         x = self.model(x)
#         x_reshape = x.permute(0, 2, 3, 4, 1).flatten(0, 3)
#         return self.l2norm(x_reshape)
#
#
# class StridedConvF3d(nn.Module):
#     def __init__(self, init_type='normal', init_gain=0.02):
#         super().__init__()
#         # self.conv1 = nn.Conv3d(256, 128, 3, stride=2)
#         # self.conv2 = nn.Conv3d(128, 64, 3, stride=1)
#         self.l2_norm = Normalize(2)
#         self.mlps = {}
#         self.moving_averages = {}
#         self.init_type = init_type
#         self.init_gain = init_gain
#
#     def create_mlp(self, x):
#         C, H = x.shape[1], x.shape[2]
#         n_down = int(np.rint(np.log2(H / 32)))
#         mlp = []
#         for i in range(n_down):
#             mlp.append(nn.Conv3d(C, max(C // 2, 64), 3, stride=2))
#             mlp.append(nn.ReLU())
#             C = max(C // 2, 64)
#         mlp.append(nn.Conv3d(C, 64, 3))
#         mlp = nn.Sequential(*mlp)
#         init_net(mlp, self.init_type, self.init_gain)
#         return mlp
#
#     def update_moving_average(self, key, x):
#         if key not in self.moving_averages:
#             self.moving_averages[key] = x.detach()
#
#         self.moving_averages[key] = self.moving_averages[key] * 0.999 + x.detach() * 0.001
#
#     def forward(self, x, use_instance_norm=False):
#         C, H = x.shape[1], x.shape[2]
#         key = '%d_%d' % (C, H)
#         if key not in self.mlps:
#             self.mlps[key] = self.create_mlp(x)
#             self.add_module("child_%s" % key, self.mlps[key])
#         mlp = self.mlps[key]
#         x = mlp(x)
#         self.update_moving_average(key, x)
#         x = x - self.moving_averages[key]
#         if use_instance_norm:
#             x = F.instance_norm(x)
#         return self.l2_norm(x)
#
#
# class PatchSampleF3d(nn.Module):
#     def __init__(self, use_mlp=False, init_type='normal', init_gain=0.02, nc=256):
#         # potential issues: currently, we use the same patch_ids for multiple images in the batch
#         super(PatchSampleF3d, self).__init__()
#         self.l2norm = Normalize(2)
#         self.use_mlp = use_mlp
#         self.nc = nc  # hard-coded
#         self.mlp_init = False
#         self.init_type = init_type
#         self.init_gain = init_gain
#
#     def create_mlp(self, feats):
#         for mlp_id, feat in enumerate(feats):
#             input_nc = feat.shape[1]
#             #  mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
#             mlp = nn.Sequential(*[nn.Linear(input_nc, self.nc), nn.ReLU(), nn.Linear(self.nc, self.nc)])
#             setattr(self, 'mlp_%d' % mlp_id, mlp)
#         init_net(self, self.init_type, self.init_gain)
#         self.mlp_init = True
#
#     def forward(self, feats, num_patches=64, patch_ids=None):
#         return_ids = []
#         return_feats = []
#         if self.use_mlp and not self.mlp_init:
#             self.create_mlp(feats)
#         for feat_id, feat in enumerate(feats):
#             B, H, W, D = feat.shape[0], feat.shape[2], feat.shape[3], feat.shape[4]
#             feat_reshape = feat.permute(0, 2, 3, 4, 1).flatten(1, 3)
#             if num_patches > 0:
#                 if patch_ids is not None:
#                     patch_id = patch_ids[feat_id]
#                 else:
#                     patch_id = torch.randperm(feat_reshape.shape[1], device=feats[0].device)
#                     patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
#                 x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
#             else:
#                 x_sample = feat_reshape
#                 patch_id = []
#             if self.use_mlp:
#                 mlp = getattr(self, 'mlp_%d' % feat_id)
#                 x_sample = mlp(x_sample)
#             return_ids.append(patch_id)
#             x_sample = self.l2norm(x_sample)
#
#             if num_patches == 0:
#                 x_sample = x_sample.permute(0, 2, 1).reshape([B, x_sample.shape[-1], H, W, D])
#             return_feats.append(x_sample)
#         return return_feats, return_ids
#
#
# class G_Resnet3d(nn.Module):
#     def __init__(self, input_nc, output_nc, nz, num_downs, n_res, ngf=64,
#                  norm=None, nl_layer=None):
#         super(G_Resnet3d, self).__init__()
#         n_downsample = num_downs
#         pad_type = 'reflect'
#         self.enc_content = ContentEncoder3d(n_downsample, n_res, input_nc, ngf, norm, nl_layer, pad_type=pad_type)
#         if nz == 0:
#             self.dec = Decoder3d(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm, activ=nl_layer,
#                                  pad_type=pad_type, nz=nz)
#         else:
#             self.dec = DecoderAll3d(n_downsample, n_res, self.enc_content.output_dim, output_nc, norm=norm,
#                                     activ=nl_layer, pad_type=pad_type, nz=nz)
#
#     def decode(self, content, style=None):
#         return self.dec(content, style)
#
#     def forward(self, image, style=None, layers=[], encode_only=False):
#         content, feats = self.enc_content(image, nce_layers=layers, encode_only=encode_only)
#         if encode_only:
#             return feats
#         else:
#             images_recon = self.decode(content, style)
#             if len(layers) > 0:
#                 return images_recon, feats
#             else:
#                 return images_recon
#
#
# class ResnetGenerator3d(nn.Module):
#     """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
#
#     We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
#     """
#
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6,
#                  padding_type='zero', no_antialias=False, no_antialias_up=False):
#         """Construct a Resnet-based generator
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert (n_blocks >= 0)
#         super(ResnetGenerator3d, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         model = [nn.ReplicationPad3d(3),
#                  nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             if (no_antialias):
#                 model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True)]
#             else:
#                 model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True),
#                           Downsample(ngf * mult * 2)]
#
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):  # add ResNet blocks
#
#             model += [
#                 ResnetBlock3d(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                               use_bias=use_bias)]
#
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             if no_antialias_up:
#                 model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
#                                              kernel_size=3, stride=1,
#                                              padding=1, output_padding=[1,1,0],
#                                              bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]
#             else:
#
#                 model += [Upsample3d(ngf * mult),
#                           nn.Conv3d(ngf * mult, int(ngf * mult / 2),
#                                     kernel_size=3, stride=1,
#                                     padding=[1,1,0],  # output_padding=1,
#                                     bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]
#         model += [nn.ReplicationPad3d(3)]
#         model += [nn.Conv3d(ngf, output_nc, kernel_size=7, stride=1, padding=[0,0,0])]
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input, layers=[], encode_only=False):
#         if -1 in layers:
#             layers.append(len(self.model))
#         if len(layers) > 0:
#             feat = input
#             feats = []
#             for layer_id, layer in enumerate(self.model):
#                 feat = layer(feat)
#                 if layer_id in layers:
#                     # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
#                     feats.append(feat)
#                 else:
#                     # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
#                     pass
#                 if layer_id == layers[-1] and encode_only:
#                     # print('encoder only return features')
#                     return feats  # return intermediate features alone; stop in the last layers
#
#             return feat, feats  # return both output and intermediate features
#         else:
#             """Standard forward"""
#             fake = self.model(input)
#             return fake
#
#
# class GlobalGenerator3d(nn.Module):
#     def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm3d,
#                  padding_type='reflect'):
#         assert (n_blocks >= 0)
#         super(GlobalGenerator3d, self).__init__()
#         activation = nn.ReLU(True)
#
#         model = [nn.ReplicationPad3d(3), nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
#         ### downsample
#         for i in range(n_downsampling):
#             mult = 2 ** i
#             model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
#                       norm_layer(ngf * mult * 2), activation]
#
#         ### resnet blocks
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):
#             model += [ResnetBlock3d(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
#
#         ### upsample
#         for i in range(n_downsampling):
#             mult = 2 ** (n_downsampling - i)
#             model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
#                                          output_padding=1),
#                       norm_layer(int(ngf * mult / 2)), activation]
#         model += [nn.ReplicationPad3d(3), nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         return self.model(input)
#
#
# class ResnetDecoder3d(nn.Module):
#     """Resnet-based decoder that consists of a few Resnet blocks + a few upsampling operations.
#     """
#
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6,
#                  padding_type='reflect', no_antialias=False):
#         """Construct a Resnet-based decoder
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert (n_blocks >= 0)
#         super(ResnetDecoder3d, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#         model = []
#         n_downsampling = 2
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):  # add ResNet blocks
#
#             model += [
#                 ResnetBlock3d(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                               use_bias=use_bias)]
#
#         for i in range(n_downsampling):  # add upsampling layers
#             mult = 2 ** (n_downsampling - i)
#             if (no_antialias):
#                 model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
#                                              kernel_size=3, stride=2,
#                                              padding=1, output_padding=[1, 1],
#                                              bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]
#             else:
#                 model += [Upsample3d(ngf * mult),
#                           nn.Conv3d(ngf * mult, int(ngf * mult / 2),
#                                     kernel_size=3, stride=1,
#                                     padding=1,
#                                     bias=use_bias),
#                           norm_layer(int(ngf * mult / 2)),
#                           nn.ReLU(True)]
#         model += [nn.ReplicationPad3d(3)]
#         model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
#         model += [nn.Tanh()]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)
#
#
# class ResnetEncoder3d(nn.Module):
#     """Resnet-based encoder that consists of a few downsampling + several Resnet blocks
#     """
#
#     def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6,
#                  padding_type='reflect', no_antialias=False):
#         """Construct a Resnet-based encoder
#
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers
#             n_blocks (int)      -- the number of ResNet blocks
#             padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
#         """
#         assert (n_blocks >= 0)
#         super(ResnetEncoder3d, self).__init__()
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         model = [nn.ReplicationPad3d(3),
#                  nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
#                  norm_layer(ngf),
#                  nn.ReLU(True)]
#
#         n_downsampling = 2
#         for i in range(n_downsampling):  # add downsampling layers
#             mult = 2 ** i
#             if (no_antialias):
#                 model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True)]
#             else:
#                 model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
#                           norm_layer(ngf * mult * 2),
#                           nn.ReLU(True),
#                           Downsample(ngf * mult * 2)]
#
#         mult = 2 ** n_downsampling
#         for i in range(n_blocks):  # add ResNet blocks
#
#             model += [
#                 ResnetBlock3d(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
#                               use_bias=use_bias)]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         """Standard forward"""
#         return self.model(input)
#
#
# class ResnetBlock3d(nn.Module):
#     """Define a Resnet block"""
#
#     def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False,use_bias=True):
#         """Initialize the Resnet block
#
#         A resnet block is a conv block with skip connections
#         We construct a conv block with build_conv_block function,
#         and implement skip connections in <forward> function.
#         Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
#         """
#         super(ResnetBlock3d, self).__init__()
#         self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias,activation)
#
#     def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias,activation):
#         """Construct a convolutional block.
#
#         Parameters:
#             dim (int)           -- the number of channels in the conv layer.
#             padding_type (str)  -- the name of padding layer: reflect | replicate | zero
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#             use_bias (bool)     -- if the conv layer uses bias or not
#
#         Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
#         """
#         conv_block = []
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReplicationPad3d(1)]
#         if padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad3d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             conv_block += [nn.ReplicationPad3d(1)]
#
#         conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), activation]
#         if use_dropout:
#             conv_block += [nn.Dropout(0.5)]
#
#         p = 0
#         if padding_type == 'reflect':
#             conv_block += [nn.ReplicationPad3d(0)]
#         elif padding_type == 'replicate':
#             conv_block += [nn.ReplicationPad3d(1)]
#         elif padding_type == 'zero':
#             p = 1
#         else:
#             raise NotImplementedError('padding [%s] is not implemented' % padding_type)
#         conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]
#
#         return nn.Sequential(*conv_block)
#
#     def forward(self, x):
#         """Forward function (with skip connections)"""
#         out = x + self.conv_block(x)  # add skip connections
#         return out
#
#
# #############################
# # 3D version of UnetGenerator
# # ###########################
#
# class UnetGenerator3d(nn.Module):
#     def __init__(self, input_nc, output_nc, num_downs, ngf=64,
#                  norm_layer=nn.BatchNorm3d, use_dropout=False, last_layer=nn.Tanh(), is_seg_net=False):
#         super(UnetGenerator3d, self).__init__()
#
#         coeff = np.power(2, np.min([num_downs, 5]) - 2).astype(int)
#         # construct unet structure
#         unet_block = UnetSkipConnectionBlock3d(ngf * coeff, ngf * coeff, norm_layer=norm_layer, innermost=True)
#         for i in range(num_downs - 5):
#             unet_block = UnetSkipConnectionBlock3d(ngf * coeff, ngf * coeff, submodule=unet_block, norm_layer=norm_layer,
#                                                    use_dropout=use_dropout)
#         if num_downs >= 5:
#             unet_block = UnetSkipConnectionBlock3d(ngf * int(coeff/2), ngf * coeff, submodule=unet_block, norm_layer=norm_layer)
#             coeff = int(coeff / 2)
#         unet_block = UnetSkipConnectionBlock3d(ngf * int(coeff/2), ngf * coeff, submodule=unet_block, norm_layer=norm_layer)
#         coeff = int(coeff / 2)  # this should equal to 2
#         unet_block = UnetSkipConnectionBlock3d(ngf, ngf * coeff, submodule=unet_block, norm_layer=norm_layer)
#         if is_seg_net:
#             self.model = UnetSkipConnectionBlock3d(output_nc, ngf, input_nc=1, submodule=unet_block, outermost=True,
#                                                    norm_layer=norm_layer, last_layer=last_layer)
#         else:
#             self.model = UnetSkipConnectionBlock3d(output_nc, ngf, submodule=unet_block, outermost=True,
#                                                    norm_layer=norm_layer, last_layer=last_layer)
#
#     def forward(self, input, layers=[], encode_only=False):
#         return self.model(input, layers=layers, encode_only=encode_only)
#
#
# class UnetSkipConnectionBlock3d(nn.Module):
#     """Defines the Unet submodule with skip connection.
#             X -------------------identity----------------------
#             |-- downsampling -- |submodule| -- upsampling --|
#         """
#
#     def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False,
#                  innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False,
#                  last_layer=nn.Tanh()):
#         """Construct a Unet submodule with skip connections.
#
#         Parameters:
#             outer_nc (int) -- the number of filters in the outer conv layer
#             inner_nc (int) -- the number of filters in the inner conv layer
#             input_nc (int) -- the number of channels in input images/features
#             submodule (UnetSkipConnectionBlock) -- previously defined submodules
#             outermost (bool)    -- if this module is the outermost module
#             innermost (bool)    -- if this module is the innermost module
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- if use dropout layers.
#         """
#         super(UnetSkipConnectionBlock3d, self).__init__()
#         self.outermost = outermost
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#         if input_nc is None:
#             input_nc = outer_nc
#         downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
#                              stride=2, padding=1, bias=use_bias)
#         downrelu = nn.LeakyReLU(0.2, True)
#         downnorm = norm_layer(inner_nc)
#         uprelu = nn.ReLU(True)
#         upnorm = norm_layer(outer_nc)
#
#         if outermost:
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1)
#             down = [downconv]
#             up = [uprelu, upconv]
#             if last_layer is not None:
#                 up.append(last_layer)
#             model = down + [submodule] + up
#             self.nce_layers_id = 1
#         elif innermost:
#             upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv]
#             up = [uprelu, upconv, upnorm]
#             model = down + up
#             self.nce_layers_id = 2
#         else:
#             upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
#                                         kernel_size=4, stride=2,
#                                         padding=1, bias=use_bias)
#             down = [downrelu, downconv, downnorm]
#             up = [uprelu, upconv, upnorm]
#
#             if use_dropout:
#                 model = down + [submodule] + up + [nn.Dropout(0.5)]
#             else:
#                 model = down + [submodule] + up
#             self.nce_layers_id = 3
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x, encode_only=False, layers=[]):
#         if encode_only:
#             feat = x
#             feats = []
#             for layer_id, layer in enumerate(self.model):
#                 if isinstance(layer, UnetSkipConnectionBlock3d):
#                     feats.append(feat)
#                     feats_ = layer(feat, encode_only)
#                     feats += feats_
#                     return feats
#                 else:
#                     feat = layer(feat)
#                 if layer_id == self.nce_layers_id:
#                     if len(feats) == 0:
#                         feats.append(feat)
#                     return feats
#         if self.outermost:
#             return self.model(x)
#         else:  # add skip connections
#             return torch.cat([self.model(x), x], 1)
#
#
# class NLayerDiscriminator3d(nn.Module):
#     """ Defines the PatchGAN discriminator with the specified arguments. """
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d,
#                  no_antialias=False, use_sigmoid=False):
#         """Construct a PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             n_layers (int)  -- the number of conv layers in the discriminator
#             norm_layer      -- normalization layer
#         """
#         super(NLayerDiscriminator3d, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         kw = 4
#         padw = 1
#         if no_antialias:
#             sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
#         else:
#             sequence = [nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
#                         Downsample(ndf)]
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             if no_antialias:
#                 sequence += [
#                     nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                     norm_layer(ndf * nf_mult),
#                     nn.LeakyReLU(0.2, True)
#                 ]
#             else:
#                 sequence += [
#                     nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#                     norm_layer(ndf * nf_mult),
#                     nn.LeakyReLU(0.2, True),
#                     Downsample(ndf * nf_mult)]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [
#             nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
#
#         if use_sigmoid:
#             sequence += [nn.Sigmoid()]
#
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input):
#         return self.model(input)
#
#
# class PixelDiscriminator3d(nn.Module):
#     """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""
#
#     def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
#         """Construct a 1x1 PatchGAN discriminator
#
#         Parameters:
#             input_nc (int)  -- the number of channels in input images
#             ndf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#         """
#         super(PixelDiscriminator3d, self).__init__()
#         if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         self.net = [
#             nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
#             norm_layer(ndf * 2),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]
#
#         self.net = nn.Sequential(*self.net)
#
#     def forward(self, input):
#         """Standard forward."""
#         return self.net(input)
#
#
# class PatchDiscriminator3d(NLayerDiscriminator3d):
#     """Defines a PatchGAN discriminator"""
#
#     def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm3d, no_antialias=False):
#         super().__init__(input_nc, ndf, 2, norm_layer, no_antialias)
#
#     def forward(self, input):
#         B, C, D, H, W = input.size(0), input.size(1), input.size(2), input.size(3), input.size(4)
#         size = 16
#         Y = H // size
#         X = W // size
#         input = input.view(B, C, Y, size, X, size)
#         input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, size, size)
#         return super().forward(input)
#
#
# class GroupedChannelNorm(nn.Module):
#     def __init__(self, num_groups):
#         super().__init__()
#         self.num_groups = num_groups
#
#     def forward(self, x):
#         shape = list(x.shape)
#         new_shape = [shape[0], self.num_groups, shape[1] // self.num_groups] + shape[2:]
#         x = x.view(*new_shape)
#         mean = x.mean(dim=2, keepdim=True)
#         std = x.std(dim=2, keepdim=True)
#         x_norm = (x - mean) / (std + 1e-7)
#         return x_norm.view(*shape)
#
#
# #############################
# # 3D network for GAN based rigid registration
# # ###########################
#
#
# class RegGenerator3D(nn.Module):
#     def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
#         """Construct a Unet generator
#             Parameters:
#             input_nc (int)  -- the number of channels in input images
#             output_nc (int) -- the number of channels in output images
#             ngf (int)       -- the number of filters in the last conv layer
#             norm_layer      -- normalization layer
#
#         We construct the U-Net from the innermost layer to the outermost layer.
#         It is a recursive process.
#         """
#         super(RegGenerator3D, self).__init__()
#
#         ######### Model #########
#         ### Atrous conv to enlarge the perceptive field (128 filters, dilation of 2, size 3×3)
#         ### 2 Conv layers 128 filters and stride of 2
#         ### 3 Residual block, 128 filters
#         ### 1×1 convolutional layer
#         ### 2 FC layers. The first FC: 256 hidden units. Second FC: is # transformation parameters
#         #########################
#
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(input_nc, 128, kernel_size=3, stride=1, padding=1, dilation=2),
#             nn.ReLU())
#         self.conv = nn.Sequential(
#             nn.Conv3d(128, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU())
#
#         self.block = nn.Sequential(
#             nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1,
#                       bias=False),
#             nn.BatchNorm3d(128),
#             nn.ReLU(),
#             nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1,
#                       bias=False),
#             nn.BatchNorm3d(128))
#
#         self.conv2 = nn.Conv3d(128, 8, kernel_size=1, stride=2, padding=1)
#         self.flatt = Flatten()
#         self.linj1 = nn.Sequential(
#             nn.Linear(20328, 256),
#             nn.ReLU())
#         self.linj2 = nn.Linear(256, 6)
#
#     def forward(self, input):
#         out = self.conv1(input)
#         out = self.conv(out)
#         out = self.conv(out)
#         # SKIP CONNECTION
#         x1 = self.block(out)
#         x1 += out
#         x2 = self.block(x1)
#         x2 += x1
#         x3 = self.block(x2)
#         x3 += x2
#         out = self.conv2(x3)
#         out = self.flatt(out)
#         out = self.linj1(out)
#         out = self.linj2(out)
#         """Standard forward"""
#         return out
#
#
# class Flatten(nn.Module):
#     def forward(self, input: torch.Tensor) -> torch.Tensor:
#         return input.view(input.size(0), -1)
#
#
# ########################################################
# # Encoder and Decoders
# ########################################################
#
#
# class E_adaIN3d(nn.Module):
#     def __init__(self, input_nc, output_nc=1, nef=64, n_layers=4,
#                  norm=None, nl_layer=None, vae=False):
#         # style encoder
#         super(E_adaIN3d, self).__init__()
#         self.enc_style = StyleEncoder3d(n_layers, input_nc, nef, output_nc, norm='none', activ='relu', vae=vae)
#
#     def forward(self, image):
#         style = self.enc_style(image)
#         return style
#
#
# class StyleEncoder3d(nn.Module):
#     def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, vae=False):
#         super(StyleEncoder3d, self).__init__()
#         self.vae = vae
#         self.model = []
#         self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
#         for i in range(2):
#             self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
#             dim *= 2
#         for i in range(n_downsample - 2):
#             self.model += [Conv3dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
#         self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
#         if self.vae:
#             self.fc_mean = nn.Linear(dim, style_dim)  # , 1, 1, 0)
#             self.fc_var = nn.Linear(dim, style_dim)  # , 1, 1, 0)
#         else:
#             self.model += [nn.Conv3d(dim, style_dim, 1, 1, 0)]
#
#         self.model = nn.Sequential(*self.model)
#         self.output_dim = dim
#
#     def forward(self, x):
#         if self.vae:
#             output = self.model(x)
#             output = output.view(x.size(0), -1)
#             output_mean = self.fc_mean(output)
#             output_var = self.fc_var(output)
#             return output_mean, output_var
#         else:
#             return self.model(x).view(x.size(0), -1)
#
#
# class ContentEncoder3d(nn.Module):
#     def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type='zero'):
#         super(ContentEncoder3d, self).__init__()
#         self.model = []
#         self.model += [Conv3dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type='reflect')]
#         # downsampling blocks
#         for i in range(n_downsample):
#             self.model += [Conv3dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type='reflect')]
#             dim *= 2
#         # residual blocks
#         self.model += [ResBlocks3d(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
#         self.model = nn.Sequential(*self.model)
#         self.output_dim = dim
#
#     def forward(self, x, nce_layers=[], encode_only=False):
#         if len(nce_layers) > 0:
#             feat = x
#             feats = []
#             for layer_id, layer in enumerate(self.model):
#                 feat = layer(feat)
#                 if layer_id in nce_layers:
#                     feats.append(feat)
#                 if layer_id == nce_layers[-1] and encode_only:
#                     return None, feats
#             return feat, feats
#         else:
#             return self.model(x), None
#
#         for layer_id, layer in enumerate(self.model):
#             print(layer_id, layer)
#
#
# class DecoderAll3d(nn.Module):
#     def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
#         super(DecoderAll3d, self).__init__()
#         # AdaIN residual blocks
#         self.resnet_block = ResBlocks3d(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)
#         self.n_blocks = 0
#         # upsampling blocks
#         for i in range(n_upsample):
#             block = [Upsample2(scale_factor=2),
#                      Conv3dBlock(dim + nz, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
#             setattr(self, 'block_{:d}'.format(self.n_blocks), nn.Sequential(*block))
#             self.n_blocks += 1
#             dim //= 2
#         # use reflection padding in the last conv layer
#         setattr(self, 'block_{:d}'.format(self.n_blocks),
#                 Conv3dBlock(dim + nz, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect'))
#         self.n_blocks += 1
#
#     def forward(self, x, y=None):
#         if y is not None:
#             output = self.resnet_block(cat_feature3d(x, y))
#             for n in range(self.n_blocks):
#                 block = getattr(self, 'block_{:d}'.format(n))
#                 if n > 0:
#                     output = block(cat_feature3d(output, y))
#                 else:
#                     output = block(output)
#             return output
#
#
# class Decoder3d(nn.Module):
#     def __init__(self, n_upsample, n_res, dim, output_dim, norm='batch', activ='relu', pad_type='zero', nz=0):
#         super(Decoder3d, self).__init__()
#
#         self.model = []
#         # AdaIN residual blocks
#         self.model += [ResBlocks3d(n_res, dim, norm, activ, pad_type=pad_type, nz=nz)]
#         # upsampling blocks
#         for i in range(n_upsample):
#             if i == 0:
#                 input_dim = dim + nz
#             else:
#                 input_dim = dim
#             self.model += [Upsample2(scale_factor=2),
#                            Conv3dBlock(input_dim, dim // 2, 5, 1, 2, norm='ln', activation=activ, pad_type='reflect')]
#             dim //= 2
#         # use reflection padding in the last conv layer
#         self.model += [Conv3dBlock(dim, output_dim, 7, 1, 3, norm='none', activation='tanh', pad_type='reflect')]
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x, y=None):
#         if y is not None:
#             return self.model(cat_feature3d(x, y))
#         else:
#             return self.model(x)
#
#
# ###########################################
# # Sequential Models
# ###########################################
#
#
# class ResBlocks3d(nn.Module):
#     def __init__(self, num_blocks, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
#         super(ResBlocks3d, self).__init__()
#         self.model = []
#         for i in range(num_blocks):
#             self.model += [ResBlock3d(dim, norm=norm, activation=activation, pad_type=pad_type, nz=nz)]
#         self.model = nn.Sequential(*self.model)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# ###########################################
# # Basic Blocks
# ###########################################
# def cat_feature3d(x, y):
#     y_expand = y.view(y.size(0), y.size(1), 1, 1, 1).expand(
#         y.size(0), y.size(1), x.size(2), x.size(3), x.size(4))
#     x_cat = torch.cat([x, y_expand], 1)
#     return x_cat
#
#
# class ResBlock3d(nn.Module):
#     def __init__(self, dim, norm='inst', activation='relu', pad_type='zero', nz=0):
#         super(ResBlock3d, self).__init__()
#
#         model = []
#         model += [Conv3dBlock(dim + nz, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
#         model += [Conv3dBlock(dim, dim + nz, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x):
#         residual = x
#         out = self.model(x)
#         out += residual
#         return out
#
#
# class Conv3dBlock(nn.Module):
#     def __init__(self, input_dim, output_dim, kernel_size, stride,
#                  padding=0, norm='none', activation='relu', pad_type='zero'):
#         super(Conv3dBlock, self).__init__()
#         self.use_bias = True
#         # initialize padding
#         if pad_type == 'reflect':
#             self.pad = nn.ReplicationPad3d(padding)
#         elif pad_type == 'zero':
#             self.pad = nn.ZeroPad3d(padding)
#         else:
#             assert 0, "Unsupported padding type: {}".format(pad_type)
#
#         # initialize normalization
#         norm_dim = output_dim
#         if norm == 'batch':
#             self.norm = nn.BatchNorm3d(norm_dim)
#         elif norm == 'inst':
#             self.norm = nn.InstanceNorm3d(norm_dim, track_running_stats=False)
#         elif norm == 'ln':
#             self.norm = LayerNorm(norm_dim)
#         elif norm == 'none':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)
#
#         # initialize activation
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace=True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)
#
#         # initialize convolution
#         self.conv = nn.Conv3d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)
#
#     def forward(self, x):
#         x = self.conv(self.pad(x))
#         if self.norm:
#             x = self.norm(x)
#         if self.activation:
#             x = self.activation(x)
#         return x
#
#
# class LinearBlock(nn.Module):
#     def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
#         super(LinearBlock, self).__init__()
#         use_bias = True
#         # initialize fully connected layer
#         self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)
#
#         # initialize normalization
#         norm_dim = output_dim
#         if norm == 'batch':
#             self.norm = nn.BatchNorm1d(norm_dim)
#         elif norm == 'inst':
#             self.norm = nn.InstanceNorm1d(norm_dim)
#         elif norm == 'ln':
#             self.norm = LayerNorm(norm_dim)
#         elif norm == 'none':
#             self.norm = None
#         else:
#             assert 0, "Unsupported normalization: {}".format(norm)
#
#         # initialize activation
#         if activation == 'relu':
#             self.activation = nn.ReLU(inplace=True)
#         elif activation == 'lrelu':
#             self.activation = nn.LeakyReLU(0.2, inplace=True)
#         elif activation == 'prelu':
#             self.activation = nn.PReLU()
#         elif activation == 'selu':
#             self.activation = nn.SELU(inplace=True)
#         elif activation == 'tanh':
#             self.activation = nn.Tanh()
#         elif activation == 'none':
#             self.activation = None
#         else:
#             assert 0, "Unsupported activation: {}".format(activation)
#
#     def forward(self, x):
#         out = self.fc(x)
#         if self.norm:
#             out = self.norm(out)
#         if self.activation:
#             out = self.activation(out)
#         return out
#
#
# ############################################
# # Normalization layers
# ############################################
#
#
# class LayerNorm(nn.Module):
#     def __init__(self, num_features, eps=1e-5, affine=True):
#         super(LayerNorm, self).__init__()
#         self.num_features = num_features
#         self.affine = affine
#         self.eps = eps
#
#         if self.affine:
#             self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
#             self.beta = nn.Parameter(torch.zeros(num_features))
#
#     def forward(self, x):
#         shape = [-1] + [1] * (x.dim() - 1)
#         mean = x.view(x.size(0), -1).mean(1).view(*shape)
#         std = x.view(x.size(0), -1).std(1).view(*shape)
#         x = (x - mean) / (std + self.eps)
#
#         if self.affine:
#             shape = [1, -1] + [1] * (x.dim() - 2)
#             x = x * self.gamma.view(*shape) + self.beta.view(*shape)
#         return x
#
#
# ####
# # down and up sample
# # #
#
# class Downsample3d(nn.Module):
#     def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
#         super(Downsample3d, self).__init__()
#         self.filt_size = filt_size
#         self.pad_off = pad_off
#         self.pad_sizes = [int(1. * (filt_size - 1) / 2),
#                           int(np.ceil(1. * (filt_size - 1) / 2)),
#                           int(1. * (filt_size - 1) / 2),
#                           int(np.ceil(1. * (filt_size - 1) / 2)),
#                           int(np.ceil(1. * (filt_size - 1) / 2)),
#                           int(np.ceil(1. * (filt_size - 1) / 2))]
#
#         self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
#
#         self.stride = stride
#         self.off = int((self.stride - 1) / 2.)
#         self.channels = channels
#
#         filt = get_filter(filt_size=self.filt_size)
#         self.register_buffer('filt', filt[None, :, :, :].repeat((self.channels, 1, 1, 1)))
#
#         self.pad = get_pad_layer(pad_type)(self.pad_sizes)
#
#     def forward(self, inp):
#         if self.filt_size == 1:
#             if self.pad_off == 0:
#                 return inp[:, :, ::self.stride, ::self.stride]
#             else:
#                 return self.pad(inp)[:, :, ::self.stride, ::self.stride]
#         else:
#             return F.conv3d(input=self.pad(inp), weight=self.filt.unsqueeze(dim=1), padding=[0], stride=[self.stride],
#                             groups=inp.shape[1], dilation=[1])
#
# # todo: maybe put in 2d if it is the 2d version
# class Upsample2(nn.Module):
#     def __init__(self, scale_factor, mode='nearest'):
#         super().__init__()
#         self.factor = scale_factor
#         self.mode = mode
#
#     def forward(self, x):
#         return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
#
#
# class Upsample3d(nn.Module):
#     def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
#         super(Upsample3d, self).__init__()
#         self.filt_size = filt_size
#         self.filt_odd = np.mod(filt_size, 2) == 1
#         self.pad_size = int((filt_size - 1) / 2)
#         self.stride = stride
#         self.off = int((self.stride - 1) / 2.)
#         self.channels = channels
#
#         filt = get_filter(filt_size=self.filt_size) * (stride ** 2)
#         self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1, 1)))
#
#         self.pad = get_pad_layer(pad_type)([1, 1, 1, 1, 1, 1])
#
#     def forward(self, inp):
#
#         ret_val = F.conv_transpose3d(self.pad(inp), weight=self.filt, stride=self.stride, padding=1 + self.pad_size,
#                                      groups=inp.shape[1], output_padding=[1,0,0])[:, :, 1:, 1:]
#         if (self.filt_odd):
#             return ret_val
#         else:
#             return ret_val[:, :, :-1, :-1]
#
#
# ############################################################
# ## DenseNet 3D
# ## taken from: https://github.com/kenshohara/3D-ResNets-PyTorch/
# ## merged with efficient Densenet from: https://github.com/gpleiss/efficient_densenet_pytorch/
# ############################################################
#
# def _bn_function_factory(norm, relu, conv):
#     def bn_function(*inputs):
#         concated_features = torch.cat(inputs, 1)
#         bottleneck_output = conv(relu(norm(concated_features)))
#         return bottleneck_output
#
#     return bn_function
#
#
# class _DenseLayer(nn.Module):
#
#     def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
#         super().__init__()
#         self.add_module('norm1', nn.BatchNorm3d(num_input_features))
#         self.add_module('relu1', nn.ReLU(inplace=True))
#         self.add_module(
#             'conv1',
#             nn.Conv3d(num_input_features,
#                       bn_size * growth_rate,
#                       kernel_size=1,
#                       stride=1,
#                       bias=False))
#         self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate))
#         self.add_module('relu2', nn.ReLU(inplace=True))
#         self.add_module(
#             'conv2',
#             nn.Conv3d(bn_size * growth_rate,
#                       growth_rate,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       bias=False))
#         self.drop_rate = drop_rate
#         self.efficient = efficient
#
#     def forward(self, *prev_features):
#         bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
#         if self.efficient:
#             bottleneck_output = cp.checkpoint(bn_function, *prev_features)
#         else:
#             bottleneck_output = bn_function(*prev_features)
#         new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
#         if self.drop_rate > 0:
#             new_features = F.dropout(new_features,
#                                      p=self.drop_rate,
#                                      training=self.training)
#         return new_features
#
#
# class _DenseBlock(nn.Module):
#
#     def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
#                  drop_rate, efficient=False):
#         super().__init__()
#         for i in range(num_layers):
#             layer = _DenseLayer(num_input_features + i * growth_rate,
#                                 growth_rate, bn_size, drop_rate, efficient=efficient)
#             self.add_module('denselayer{}'.format(i + 1), layer)
#
#     def forward(self, init_features):
#         features = [init_features]
#         for name, layer in self.named_children():
#             new_features = layer(*features)
#             features.append(new_features)
#         return torch.cat(features, 1)
#
#
# class _Transition(nn.Sequential):
#
#     def __init__(self, num_input_features, num_output_features):
#         super().__init__()
#         self.add_module('norm', nn.BatchNorm3d(num_input_features))
#         self.add_module('relu', nn.ReLU(inplace=True))
#         self.add_module(
#             'conv',
#             nn.Conv3d(num_input_features,
#                       num_output_features,
#                       kernel_size=1,
#                       stride=1,
#                       bias=False))
#         self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))
#
#
# class DenseNet(nn.Module):
#     """Densenet-BC model class
#     Args:
#         growth_rate (int) - how many filters to add each layer (k in paper)
#         block_config (list of 4 ints) - how many layers in each pooling block
#         num_init_features (int) - the number of filters to learn in the first convolution layer
#         bn_size (int) - multiplicative factor for number of bottle neck layers
#           (i.e. bn_size * k features in the bottleneck layer)
#         drop_rate (float) - dropout rate after each dense layer
#         num_classes (int) - number of classification classes
#     """
#
#     def __init__(self,
#                  n_input_channels=3,
#                  conv1_size=5,
#                  conv1_stride=2,
#                  no_max_pool=False,
#                  growth_rate=32,
#                  block_config=(6, 12, 24, 16),
#                  num_init_features=64,
#                  bn_size=4,
#                  drop_rate=0,
#                  num_classes=1000,
#                  compression=0.5,
#                  efficient=False):
#
#         super().__init__()
#
#         # First convolution
#         self.features = [('conv1',
#                           nn.Conv3d(n_input_channels,
#                                     num_init_features,
#                                     kernel_size=conv1_size,
#                                     stride=conv1_stride,
#                                     padding=conv1_size // 2,
#                                     bias=False)),
#                          ('norm1', nn.BatchNorm3d(num_init_features)),
#                          ('relu1', nn.ReLU(inplace=True))]
#         if not no_max_pool:
#             self.features.append(
#                 ('pool1', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)))
#         self.features = nn.Sequential(OrderedDict(self.features))
#
#         # Each denseblock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers=num_layers,
#                                 num_input_features=num_features,
#                                 bn_size=bn_size,
#                                 growth_rate=growth_rate,
#                                 drop_rate=drop_rate,
#                                 efficient=efficient
#                                 )
#             self.features.add_module('denseblock{}'.format(i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features,
#                                     num_output_features=int(num_features * compression))
#                 self.features.add_module('transition{}'.format(i + 1), trans)
#                 num_features = int(num_features * compression)
#
#         # Final batch norm
#         self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
#
#         # Linear layer
#         self.classifier = nn.Linear(num_features, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight,
#                                         mode='fan_out',
#                                         nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool3d(out, output_size=(1, 1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out
#
#
# ##############################
# # Simple rigid registration network
# ##############################
# class NormalNet(nn.Module):
#     def __init__(self, input_nc=2, ndf=64, n_layers=5, norm_layer=nn.BatchNorm3d,
#                  num_classes=6, img_shape=None, **kwargs):
#         super(NormalNet, self).__init__()
#         if img_shape is None:
#             img_shape = [128, 128, 128]
#         if type(norm_layer) == functools.partial:
#             use_bias = norm_layer.func == nn.InstanceNorm3d
#         else:
#             use_bias = norm_layer == nn.InstanceNorm3d
#
#         kw = 4
#         padw = int(np.ceil((kw - 1) / 2))
#         sequence = [
#             nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         nf_mult = 1
#         nf_mult_prev = 1
#         for n in range(1, n_layers):
#             nf_mult_prev = nf_mult
#             nf_mult = min(2 ** n, 8)
#             sequence += [
#                 nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
#                           kernel_size=kw, stride=2, padding=padw, bias=use_bias),
#                 norm_layer(ndf * nf_mult),
#                 nn.LeakyReLU(0.2, True)
#             ]
#
#         nf_mult_prev = nf_mult
#         nf_mult = min(2 ** n_layers, 8)
#         sequence += [
#             nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
#                       kernel_size=kw, stride=1, padding=padw, bias=use_bias),
#             norm_layer(ndf * nf_mult),
#             nn.LeakyReLU(0.2, True)
#         ]
#
#         sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
#
#         sequence += [nn.Flatten()]
#         self.model = nn.Sequential(*sequence)
#
#         input_ = torch.zeros((1, input_nc, *img_shape))
#         out = self.model(input_)
#         n = out.shape[1]
#         sequence += [nn.Linear(n, num_classes, bias=False)]  # TODO: make this dynamic
#         self.model = nn.Sequential(*sequence)
#
#     def forward(self, input_):
#         return self.model(input_)
#
#
# def define_reg_model(model_type='121', init_type='normal', init_gain=0.02,
#                      gpu_ids=None, efficient=True, **kwargs):
#     if gpu_ids is None:
#         gpu_ids = []
#     model_types = ['121', '169', '201', '264', 'NormalNet']
#     if model_type == '121':
#         model = DenseNet(num_init_features=64,
#                          growth_rate=32,
#                          efficient=efficient,
#                          block_config=(6, 12, 24, 16),
#                          **kwargs)
#     elif model_type == '169':
#         model = DenseNet(num_init_features=64,
#                          growth_rate=32,
#                          efficient=efficient,
#                          block_config=(6, 12, 32, 32),
#                          **kwargs)
#     elif model_type == '201':
#         model = DenseNet(num_init_features=64,
#                          growth_rate=32,
#                          efficient=efficient,
#                          block_config=(6, 12, 48, 32),
#                          **kwargs)
#     elif model_type == '264':
#         model = DenseNet(num_init_features=64,
#                          growth_rate=32,
#                          efficient=efficient,
#                          block_config=(6, 12, 64, 48),
#                          **kwargs)
#     elif model_type == 'NormalNet':
#         model = NormalNet(**kwargs)
#     else:
#         raise NotImplementedError(f'model_type can be among {model_types}')
#
#     return init_net(model, init_type, init_gain, gpu_ids)
