from glob import glob
from PIL import Image
import torch.nn as nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset
from utils.utils import get_argparser_group
from pathlib import Path
import torchvision.transforms as transforms
import pytorch_lightning as pl
import os

class Facades(pl.LightningDataModule):
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
        self.out_features = self.hparams.out_channels
        self.input_channels = self.hparams.in_channels
        self.batch_size = self.hparams.batch_size
        self.data = {}

    def prepare_data(self):
        self.data['train'] = BoneUSDataset(self.hparams.data_root, self.hparams.target_size)
        self.data['val'] = BoneUSDataset(self.hparams.data_root, self.hparams.target_size)
        self.data['test'] = BoneUSDataset(self.hparams.data_root, self.hparams.target_size)

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
        mnist_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        mnist_specific_args.add_argument('--dataset_disable_train_transform', action='store_true')
        mnist_specific_args.add_argument('--target_size', default=256, type=int)
        return parser


class FacadesDataset(Dataset):
    def __init__(self, path, target_size=None):
        self.filenames = glob(str(Path(path) / '*'))
        self.target_size = target_size

        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = Image.open(filename)
        image = self.transform(image)
        image_width = image.shape[2]

        real = image[:, :, :image_width // 2]
        condition = image[:, :, image_width // 2:]

        target_size = self.target_size
        if target_size:
            condition = interpolate(condition, size=target_size)
            real = interpolate(real, size=target_size)

        return real, condition


class BoneUSDataset(Dataset):
    def __init__(self, path, target_size=None):
        self.root_dir = path
        self.images_dir = os.path.join(self.root_dir, 'images')
        self.labels_dir = os.path.join(self.root_dir, 'labels')
        self.filenames = glob(str(Path(self.images_dir) / '*'))
        self.target_size = target_size

        self.transform = transforms.ToTensor()
        self.padding = nn.ZeroPad2d([24, 24, 0, 9])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        image_filename = self.filenames[idx]
        image_id = os.path.split(image_filename)[-1].strip(".png")
        condition_name = os.path.join(self.labels_dir, image_id + "_label.png")

        real = Image.open(image_filename)
        condition = Image.open(condition_name)

        real = self.transform(real)
        condition = self.transform(condition)

        # Padding to avoid network breaks
        condition = condition.unsqueeze(0)
        real = real.unsqueeze(0)
        condition = self.padding(condition)
        real = self.padding(real)
        condition = condition.squeeze(0)
        real = real.squeeze(0)

        condition = condition*255

        return real, condition, image_id
