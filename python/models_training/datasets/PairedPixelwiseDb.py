from abc import ABC

from datasets.dataset_utils import *
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from utils.utils import get_argparser_group
from pathlib import Path
import pytorch_lightning as pl
import os
import numpy as np
from utils.utils import str2bool
import abc
# TODO: add description on how it expects the db to be structured


class BaseDbModuleMeta(type(pl.LightningDataModule), type(abc.ABC)):
    pass

class BaseDbModule(pl.LightningDataModule, abc.ABC, metaclass=BaseDbModuleMeta):
    def __init__(self, hparams):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.output_nc
        self.input_channels = self.hparams.input_nc
        self.batch_size = self.hparams.batch_size
        self.data = {}

    def _set_random_splits(self):

        if isinstance(self.hparams.random_split, str):
            splits = self.hparams.random_split.split("_")
            splits = [int(item) for item in splits]
            self.hparams.random_split = splits

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


class BasedModuleChildMeta(type(BaseDbModule), type(abc.ABC)):
    pass

class FolderSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):

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

class SubjectSplitdDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams):
        super().__init__(hparams)

        self._set_split_subjects()
        self._set_random_splits()

    def _set_split_subjects(self):

        for split in ['train', 'val', 'test']:

            split_subjects = getattr(self.hparams, split + '_subjects')
            subject_list = split_subjects.split(",")
            subject_list = [item.replace(" ", "") for item in subject_list]

            subject_list = [item for item in subject_list if item != "" and item != " "]
            setattr(self.hparams, split + "_subjects", subject_list)

    def prepare_data(self):

        train_given = len(self.hparams.train_subjects) != 0
        val_given = len(self.hparams.val_subjects) != 0
        test_given = len(self.hparams.test_subjects) != 0

        subject_ids = get_subject_ids_from_data(os.listdir(self.data_root))

        # if the test is not given either we have a random split with the percentages in self.hparams.random_split or
        # we assume the test set is empty
        if not test_given:
            if not train_given and not val_given:
                self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects = \
                    get_subject_based_random_split(subject_ids, split_percentage=self.hparams.random_split)

            elif train_given and val_given:
                self.hparams.test_subjects = []

            elif not train_given and val_given:
                self.hparams.train_subjects = [item for item in subject_ids if item not in self.hparams.val_subjects]

            elif train_given and not val_given:
                self.hparams.val_subjects = [item for item in subject_ids if item not in self.hparams.train_subjects]

            else:
                raise ValueError("Unsupported configuration")

        if test_given:
            subject_ids = [item for item in subject_ids if item not in self.hparams.test_subjects]

            if train_given and val_given:
                pass

            elif not train_given and not val_given:
                assert len(self.hparams.random_split) == 2, "If test set is given, random split must contain only" \
                                                            " two values, one for train and one for validation"
                self.hparams.train_subjects, self.hparams.val_subjects, _ = \
                    get_subject_based_random_split(subject_ids, split_percentage=self.hparams.random_split)

            elif not train_given and val_given:
                self.hparams.train_subjects = [item for item in subject_ids if item not in self.hparams.val_subjects]

            elif train_given and not val_given:
                self.hparams.val_subjects = [item for item in subject_ids if item not in self.hparams.train_subjects]

            else:
                raise ValueError("Unsupported configuration")

        for subject_list, split in zip(
                [self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects],
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

class RandomSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):
    def __init__(self, hparams):
        super().__init__(hparams)
        self._set_random_splits()

    def prepare_data(self):

        data_list = sorted(make_dataset(self.data_root, self.hparams.max_dataset_size))  # get image paths
        data_list = [item for item in data_list if "label" not in item]
        dataset_size = len(data_list)

        if self.hparams.test_folder != '':

            test_data_list = sorted(make_dataset(self.hparams.test_folder, self.hparams.max_dataset_size))
            test_data_list = [item for item in test_data_list if "label" not in item]

            n_training_samples = round(self.hparams.random_split[0] * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            val_data_list = [item for item in data_list if item not in train_data_list]

        else:
            n_training_samples = round(self.hparams.random_split[0] * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            n_val_samples = round(self.hparams.random_split[1] * dataset_size)
            val_data_list = random.sample(data_list, n_val_samples)

            test_data_list = [item for item in data_list if item not in train_data_list and item not in val_data_list]

        for split, data_list in zip(['train', 'val', 'test'], [train_data_list, val_data_list, test_data_list]):
            self.data[split] = USBones(self.hparams, data_list=data_list)

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
        dataset_specific_args.add_argument("--test_folder", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)
        return parser

class BaseDataset(Dataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):

        if split is None and subject_list is None:
            raise ValueError("Either split or subject list must be not None")

        self.hparams = hparams
        self.input_nc = 1
        self.output_nc = 1

        if data_list is not None:
            self.AB_paths = data_list
            return

        if data_structure == 'folder_based':
            assert isinstance(split, str), "Split must be a string - usually train, test or val"
            self.dir_AB = os.path.join(hparams.data_root, split)
        else:
            self.dir_AB = hparams.data_root

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


