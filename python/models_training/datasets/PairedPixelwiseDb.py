from datasets.dataset_utils import *
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.utils import get_argparser_group
from pathlib import Path
import pytorch_lightning as pl
import os
import numpy as np
from utils.utils import str2bool
from abc import ABC

# TODO: add description on how it expects the db to be structured

class BaseDbModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.output_nc
        self.input_channels = self.hparams.input_nc
        self.batch_size = self.hparams.batch_size
        self.data = {}

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

class FolderSplitdDb(BaseDbModule):

    def __init__(self, hparams):
        super().__init__(hparams)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = USBones(self.hparams, split, data_structure='folder_based')

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

class SubjectSplitdDb(BaseDbModule):

    def __init__(self, hparams):
        super().__init__(hparams)

    def _get_split_subjects(self, split):

        if split not in ['train', 'val', 'test']:
            return []

        split_subjects = getattr(self.hparams, split + '_subjects')
        subject_list = split_subjects.split(",")
        subject_list = [item.replace(" ", "") for item in subject_list]

        return subject_list

    def prepare_data(self):

        self.hparams.train_subjects = self._get_split_subjects('train')
        self.hparams.val_subjects = self._get_split_subjects('val')
        self.hparams.test_subjects = self._get_split_subjects('test')

        subject_ids = get_subject_ids_from_data(os.listdir(self.data_root))

        if len(self.hparams.test_subjects) == len(self.hparams.train_subjects) == len(self.hparams.val_subjects) == 0:

            self.hparams.train_subjects, self.hparams.val_subjects, test_subjects = \
                get_subject_based_random_split(subject_ids, split_percentage=self.hparams.random_split)

        elif len(self.hparams.test_subjects) != 0 and \
                len(self.hparams.train_subjects) == len(self.hparams.val_subjects) == 0:

            assert len(self.hparams.random_split) == 3, "If test set is given, random split must contain only two " \
                                                        "values, one for train and one for validation"
            subject_ids = [item for item in subject_ids if item not in self.hparams.test_subjects]
            self.hparams.train_subjects, self.hparams.val_subjects = \
                get_subject_based_random_split(subject_ids, split_percentage=self.hparams.random_split)

        elif len(self.hparams.test_subjects) != 0 and len(self.hparams.val_subjects) != 0 \
                and len(self.hparams.train_subjects) == 0:
            self.hparams.train_subjects = [item for item in subject_ids if
                                           item not in self.hparams.test_subjects and
                                           item not in self.hparams.val_subjects]
            pass

        elif len(self.hparams.test_subjects) != 0 and \
                len(self.hparams.train_subjects) != 0 and len(self.hparams.val_subjects) != 0:
            pass

        else:
            raise ValueError("If test set is given, train and val sets must be either both given or none of them given")

        for subject_list, split in \
                zip([self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects],
                    ['train', 'val', 'test']):

            self.data[split] = USBones(hparams=self.hparams,
                                       subject_list=subject_list,
                                       data_structure='subject_based')

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
        dataset_specific_args.add_argument("--train_subjects", default='', type=str)
        dataset_specific_args.add_argument("--val_subjects", default='', type=str)
        dataset_specific_args.add_argument("--test_subjects", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)
        dataset_specific_args.add_argument("--no_flip", default=True, type=str2bool)

        return parser

class BaseDataset(Dataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):

        if split is None and subject_list is None:
            raise ValueError("Either split or subject list must be not None")

        self.hparams = hparams
        self.input_nc = 1
        self.output_nc = 1

        if data_structure == 'folder_based':
            assert isinstance(split, str), "Split must be a string - usually train, test or val"
            self.dir_AB = os.path.join(hparams.data_root, split)
        else:
            self.dir_AB = hparams.data_root

        if data_list is not None:
            self.dir_AB = data_list
            return

        self.AB_paths = sorted(make_dataset(self.dir_AB, self.hparams.max_dataset_size))  # get image paths
        self.AB_paths = [item for item in self.AB_paths if "label" not in item]

        if data_structure == 'subject_based':
            self.AB_paths = get_split_subjects_data(self.AB_paths, subject_list)

        assert (self.hparams.load_size >= self.hparams.crop_size)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, idx):
        raise NotImplementedError

class USBones(BaseDataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):
        super().__init__(hparams, split, subject_list, data_list, data_structure)

    def __getitem__(self, idx):
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

        return B, A  # return image, label

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


