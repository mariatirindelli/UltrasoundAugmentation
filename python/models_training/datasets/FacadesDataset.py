from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.utils import get_argparser_group
from pathlib import Path
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os
import random
import numpy as np
from utils.utils import str2bool

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)

def __scale_width(img, target_size, crop_size, method=Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir_path, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir_path), '%s is not a valid directory' % dir_path

    for root, _, fnames in sorted(os.walk(dir_path)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class PairedDataset(pl.LightningDataModule):
    """
    This class defines the MNIST dataset.
    It splits the dataset into train, validation and a test dataset
    as : .data['train'], .data['val'] and .data['test'].
    The dataframes can be accessed in the same fashion e.g. .df['train'] with
    the addition of 'all' to access all the data.
    Arguments:
        hparams: config from pytorch lightning
    """

    def __init__(self, hparams):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.output_nc
        self.input_channels = self.hparams.input_nc
        self.batch_size = self.hparams.batch_size
        self.data = {}

    def prepare_data(self):
        self.data['train'] = USBones(self.hparams, 'train')
        self.data['val'] = USBones(self.hparams, 'val')
        self.data['test'] = USBones(self.hparams, 'test')

    def __dataloader(self, split=None):
        dataset = self.data[split]
        shuffle = split == 'train'  # shuffle also for
        train_sampler = None
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=train_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
        return loader

    def train_dataloader(self):
        dataloader = self.__dataloader(split='train')
        return dataloader

    def val_dataloader(self):
        dataloader = self.__dataloader(split='val')
        return dataloader

    def test_dataloader(self):
        dataloader = self.__dataloader(split='test')
        return dataloader

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument('--preprocess', default='resize_and_crop', type=str)
        dataset_specific_args.add_argument('--load_size', default=256, type=int)
        dataset_specific_args.add_argument('--crop_size', default=256, type=int)
        dataset_specific_args.add_argument('--max_dataset_size', default=np.inf, type=float)
        dataset_specific_args.add_argument("--no_flip", default=True, type=str2bool)

        return parser


class FacadesDataset(Dataset):
    def __init__(self, hparams, split):
        self.hparams = hparams
        self.root = hparams.data_root
        self.dir_AB = os.path.join(hparams.data_root, split)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, self.hparams.max_dataset_size))  # get image paths
        assert (self.hparams.load_size >= self.hparams.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = 1
        self.output_nc = 1

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, idx):

        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[idx]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return A, B


class USBones(Dataset):
    def __init__(self, hparams, split):
        self.hparams = hparams
        self.root = hparams.data_root
        self.dir_AB = os.path.join(hparams.data_root, split)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, self.hparams.max_dataset_size))  # get image paths
        self.AB_paths = [item for item in self.AB_paths if "label" not in item]
        assert (self.hparams.load_size >= self.hparams.crop_size)
        self.input_nc = 1
        self.output_nc = 1

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, idx):

        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.AB_paths[idx].replace(".png", "_label.png")
        B_path = self.AB_paths[idx]

        A = Image.open(A_path).convert('LA')
        B = Image.open(B_path).convert('LA')

        # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return B, A
