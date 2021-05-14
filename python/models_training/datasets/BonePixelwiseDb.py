from abc import ABC

from datasets.torch_us_datasets import *
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

        if self.hparams.db_kind == 'paired':
            self.torch_dataset = USBonesPaired
        elif self.hparams.db_kind == 'unpaired':
            self.torch_dataset = USBonesUnpaired

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
        dataset_specific_args.add_argument("--db_kind", default='paired', type=str)
        dataset_specific_args.add_argument("--serial_batches", default=True, type=str2bool)

        return parser


class BasedModuleChildMeta(type(BaseDbModule), type(abc.ABC)):
    pass

class FolderSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams):
        super().__init__(hparams)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = self.torch_dataset(self.hparams, split, data_structure='folder_based')

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        parser = BaseDbModule.add_dataset_specific_args(parser)

        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--unpaired_percentage", default=30, type=float)

        return parser

class SubjectSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams):
        super().__init__(hparams)

        if hparams.max_dataset_size < 0:
            hparams.max_dataset_size = np.inf

        self._set_split_subjects()
        self._set_random_splits()

    def _set_split_subjects(self):

        for split in ['train', 'val', 'test']:

            split_subjects = getattr(self.hparams, split + '_subjects')
            subject_list = split_subjects.split(",")
            subject_list = [item.replace(" ", "") for item in subject_list]

            subject_list = [item for item in subject_list if item != "" and item != " "]
            setattr(self.hparams, split + "_subjects", subject_list)

    def log_db_info(self):

        print("\n---------------------------------------------------------------------------------")
        if len(self.hparams.random_split) == 3:
            print("Db split: train: {} - val: {} - test: {}".format(self.hparams.random_split[0],
                                                                    self.hparams.random_split[1],
                                                                    self.hparams.random_split[2]))

        if len(self.hparams.random_split) == 2:
            print("Db split: train: {} - val: {} - test: {}".format(self.hparams.random_split[0],
                                                                    self.hparams.random_split[1],
                                                                    0))

        for split in ['train', 'val', 'test']:

            subject_list = getattr(self.hparams, split + "_subjects")

            print("Num subjects in {} split: {} - num data: {}".format(split,
                                                                       len(subject_list),
                                                                       len(self.data[split].AB_paths) ))

            string_to_plot = "{} split ids : ".format(split)

            for i in subject_list:
                string_to_plot += "{}, ".format(i)

            print(string_to_plot[0:-2])

        print("---------------------------------------------------------------------------------\n")

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
                    get_subject_based_random_split(subject_ids, split_percentages=self.hparams.random_split)

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
                    get_subject_based_random_split(subject_ids, split_percentages=self.hparams.random_split)

            elif not train_given and val_given:
                self.hparams.train_subjects = [item for item in subject_ids if item not in self.hparams.val_subjects]

            elif train_given and not val_given:
                self.hparams.val_subjects = [item for item in subject_ids if item not in self.hparams.train_subjects]

            else:
                raise ValueError("Unsupported configuration")

        for subject_list, split in zip(
                [self.hparams.train_subjects, self.hparams.val_subjects, self.hparams.test_subjects],
                ['train', 'val', 'test']):

            self.data[split] = self.torch_dataset(hparams=self.hparams,
                                                  subject_list=subject_list,
                                                  data_structure='subject_based')

        self.log_db_info()

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """

        parser = BaseDbModule.add_dataset_specific_args(parser)

        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--train_subjects", default='', type=str)
        dataset_specific_args.add_argument("--val_subjects", default='', type=str)
        dataset_specific_args.add_argument("--test_subjects", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)

        return parser

class RandomSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):
    def __init__(self, hparams):
        super().__init__(hparams)
        self._set_random_splits()

    @staticmethod
    def _filter_list(data_list, keep):

        if keep == "images":
            return [item for item in data_list if "label" not in item]
        elif keep == "labels":
            return [item for item in data_list if "label" in item]

    def _get_splits_for_data(self, extract="images"):
        data_list = sorted(make_dataset(self.data_root, self.hparams.max_dataset_size))  # get image paths
        data_list = self._filter_list(data_list, keep=extract)
        dataset_size = len(data_list)

        if self.hparams.test_folder != '':

            test_data_list = sorted(make_dataset(self.hparams.test_folder, self.hparams.max_dataset_size))
            test_data_list = self._filter_list(test_data_list, keep=extract)

            n_training_samples = round(self.hparams.random_split[0] * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            val_data_list = [item for item in data_list if item not in train_data_list]

        else:
            n_training_samples = round(self.hparams.random_split[0] / 100 * dataset_size)
            train_data_list = random.sample(data_list, n_training_samples)

            if len(self.hparams.random_split) == 2:
                val_data_list = [item for item in data_list if item not in train_data_list]
                test_data_list = []

            else:
                n_val_samples = round(self.hparams.random_split[1] / 100 * dataset_size)
                val_data_list = random.sample(data_list, n_val_samples)

                test_data_list = [item for item in data_list if
                                  item not in train_data_list and item not in val_data_list]

        return [train_data_list, val_data_list, test_data_list]

    def prepare_data(self):

        if self.hparams.db_kind == 'paired':
            splits_data_lists = self._get_splits_for_data(extract="images")

        else:
            splits_data_lists_images = self._get_splits_for_data(extract="images")
            splits_data_lists_labels = self._get_splits_for_data(extract="labels")

            splits_data_lists = [[splits_data_lists_images[0], splits_data_lists_labels[0]],
                                 [splits_data_lists_images[1], splits_data_lists_labels[1]],
                                 [splits_data_lists_images[2], splits_data_lists_labels[2]]]

        for split, data_list in zip(['train', 'val', 'test'], splits_data_lists):
            self.data[split] = self.torch_dataset(self.hparams, data_list=data_list, split=split)

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        parser = BaseDbModule.add_dataset_specific_args(parser)
        dataset_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        dataset_specific_args.add_argument("--test_folder", default='', type=str)
        dataset_specific_args.add_argument("--random_split", default='80_10_10', type=str)

        return parser

class MixedDb(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self._reformat_hparams()
        self.data = {}
        self.torch_dataset = USBonesPaired  # todo make it parameterizable

        self.data_roots = dict()

        for split in ['train', 'val', 'test']:
            self.data_roots[split] = [os.path.join(self.hparams.data_root, item)
                                      for item in getattr(self.hparams, split + "_folders")]

    # todo: do this in argparse instead
    def _reformat_hparams(self):
        for item in ['train_folders', 'val_folders', 'test_folders']:
            split_folders = getattr(self.hparams, item)
            split_folders = split_folders.replace(" ", "")
            split_folders = split_folders.split(",")
            setattr(self.hparams, item, split_folders)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = [self.torch_dataset(hparams=self.hparams,
                                                   split=split,
                                                   data_root=item) for item in self.data_roots[split]]

    def __dataloader(self, split=None):

        # todo: change this!
        if split != 'train':

            dataloaders = [DataLoader(dataset=item,
                                      batch_size=self.hparams.batch_size,
                                      shuffle=False,
                                      sampler=None,
                                      num_workers=self.hparams.num_workers,
                                      pin_memory=True) for item in self.data[split]]
            return dataloaders



        dataloaders = dict()
        for dataset, folder_name in zip(self.data[split], getattr(self.hparams, split + "_folders")):
            shuffle_db = split == 'train'  # shuffle also for
            train_sampler = None
            dataloaders[folder_name] = DataLoader(dataset=dataset,
                                                   batch_size=self.hparams.batch_size,
                                                   shuffle=shuffle_db,
                                                   sampler=train_sampler,
                                                   num_workers=self.hparams.num_workers,
                                                   pin_memory=True)

        return dataloaders

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
        dataset_specific_args.add_argument("--serial_batches", default=True, type=str2bool)
        dataset_specific_args.add_argument("--train_folders", default="", type=str)
        dataset_specific_args.add_argument("--val_folders", default="", type=str)
        dataset_specific_args.add_argument("--test_folders", default="", type=str)

        return parser
