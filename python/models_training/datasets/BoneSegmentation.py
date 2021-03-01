from abc import ABC
from typing import Optional, Any

from torch.utils.data import DataLoader, Dataset
from utils.utils import get_argparser_group
from pathlib import Path
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torch
from skimage import io
from us_augmentation import *
import math
import pathlib
import re
import random

import torchvision.transforms.functional as TF


# dyndata returns a dict with keys
# (['image': image, 'label': label, 'filename': self.data_list[idx], 'weights':pos_weights])

# augmentation_mode : 'tilting' or 'deformation' or 'm_reflection' or 'all' or 'none' or 'classical'

class BoneSegmentationDb(pl.LightningDataModule):
    """
    This class defines the MNIST dataset.
    It splits the dataset into train, validation and a test dataset
    as : .data['train'], .data['val'] and .data['test'].
    The dataframes can be accessed in the same fashion e.g. .df['train'] with
    the addition of 'all' to access all the data.
    Arguments:
        hparams: config from pytorch lightning
    """

    # # todo: check this two methods - what should they do?
    # def setup(self, stage: Optional[str] = None):
    #     pass
    #
    # def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
    #     pass

    def __init__(self, hparams):
        super().__init__()
        # self.name = 'MNIST'
        self.hparams = hparams
        self.data_root = Path(hparams.data_root)
        self.out_features = self.hparams.out_channels
        self.input_channels = self.hparams.in_channels
        self.batch_size = self.hparams.batch_size
        self.data = {}
        self.input_height = 28
        self.input_width = 28

    def prepare_data(self):

        for split in ['train', 'val', 'test']:

            if self.hparams.augmentation_mode == 'none':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=0,
                                                           tilting_prob=0,
                                                           deformation_prob=0,
                                                           m_reflection_prob=0,
                                                           noise_prob=0,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'tilting':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=1,
                                                           tilting_prob=self.hparams.tilting_prob,
                                                           deformation_prob=0,
                                                           m_reflection_prob=0,
                                                           noise_prob=0,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'deformation':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=1,
                                                           tilting_prob=0,
                                                           deformation_prob=self.hparams.deformation_prob,
                                                           m_reflection_prob=0,
                                                           noise_prob=0,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'm_reflection':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=1,
                                                           tilting_prob=0,
                                                           deformation_prob=0,
                                                           m_reflection_prob=self.hparams.m_reflection_prob,
                                                           noise_prob=0,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'noise':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=1,
                                                           tilting_prob=0,
                                                           deformation_prob=0,
                                                           m_reflection_prob=0,
                                                           noise_prob=self.hparams.noise_prob,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'all':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=1,
                                                           tilting_prob=self.hparams.tilting_prob,
                                                           deformation_prob=self.hparams.deformation_prob,
                                                           m_reflection_prob=self.hparams.m_reflection_prob,
                                                           noise_prob=self.hparams.noise_prob,
                                                           classical_prob=0)
            elif self.hparams.augmentation_mode == 'classical':
                self.data[split] = BoneSegmentationDataset(root_dir=self.hparams.data_root,
                                                           split=split,
                                                           augmentation_prob=0,
                                                           tilting_prob=0,
                                                           deformation_prob=0,
                                                           m_reflection_prob=0,
                                                           noise_prob=0,
                                                           classical_prob=self.hparams.classical_prob,
                                                           classical_augmentation=True)

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

    def set_num_channels(self, num_channels):
        """
        Set the number of channels of the input
        :param num_channels:
        """
        self.input_channels = num_channels

    @staticmethod
    def add_dataset_specific_args(parser):
        """
        Parameters you define here will be available to your model through self.hparams
        :param parser:
        """
        # TOD:
        mnist_specific_args = get_argparser_group(title='Dataset options', parser=parser)
        mnist_specific_args.add_argument('--dataset_disable_train_transform', action='store_true')
        mnist_specific_args.add_argument('--input_height', default=28, type=int)
        mnist_specific_args.add_argument('--input_width', default=28, type=int)
        mnist_specific_args.add_argument('--augmentation_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--tilting_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--deformation_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--m_reflection_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--noise_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--classical_prob', default=0.5, type=float)
        mnist_specific_args.add_argument('--augmentation_mode', default='none', type=str)
        mnist_specific_args.add_argument('--gaussian_kernel', default=15, type=int)
        mnist_specific_args.add_argument('--gaussian_sigma', default=5, type=int)
        return parser


class BoneSegmentationDataset(Dataset):
    """BoneSegmentationDataset dataset."""

    def __init__(self, root_dir, split, augmentation_prob=0.5, tilting_prob=0, deformation_prob=0, m_reflection_prob=0,
                 noise_prob=0, classical_prob=0.0, classical_augmentation=False, pix2pix_augmentation=False):
        self.split = split
        self.root_dir = os.path.join(root_dir, split)
        self.data_list = os.listdir(os.path.join(self.root_dir, "images"))
        self.data_list = [item for item in self.data_list if ".png" in item]
        self.n_non_augmented_images = len(self.data_list)

        self.deformation = us.Deformation()
        self.tilting = us.ProbeTilting(plot_result=False)
        self.multiple_reflection = us.MultipleReflections(plot_result=False)
        self.noise = us.NoiseAugmentation(plot_result=False)
        self.augmentation_probability = augmentation_prob
        self.tilting_prob = tilting_prob
        self.deformation_prob = deformation_prob
        self.m_reflection_prob = m_reflection_prob
        self.noise_prob = noise_prob
        self.classical_prob = classical_prob
        self.classical_augmentation = classical_augmentation

        self.transform = transforms.ToTensor()  # converts to tensor and normalize between 0 and 1

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, "images", self.data_list[idx])
        label_name = os.path.join(self.root_dir, "labels", self.data_list[idx].split(".")[0] + "_label.png")

        # reading images and labels
        image = io.imread(img_name)
        label = io.imread(label_name)

        # applying augmentations if needed
        augmentation_string = ''
        if self.split != "test" and self.augmentation_probability > 0:
            if np.random.uniform(low=0.0, high=1.0) <= self.augmentation_probability:
                image, label, augmentation_string = self.apply_random_augmentation(image=image,
                                                                                   label=label,
                                                                                   filename=img_name)

        # dilating labels and get positive weights
        label = cv2.dilate(label*255, kernel=np.ones((5, 5), np.uint8))
        pos_weights = self.get_pos_weights(label)

        image = image.astype(np.float)/np.max(image)
        image = self.transform(image)
        label = self.transform(label)

        if self.split != "test" and self.classical_augmentation:
            image, label, augmentation_string = self.apply_classical_augmentation(image, label)

        # Making sure the label is binary (deformation might have blurred the label)
        label = torch.where(label >= 0.5, 1, 0)

        sample = {'image': image,
                  'label': label.float(),
                  'filename': self.data_list[idx],
                  'pos_weights': pos_weights,
                  'augmentations': augmentation_string}

        return sample

    @staticmethod
    def get_pos_weights(label):
        non_zeros = np.count_nonzero(label)
        zeros = label.size - non_zeros
        pos_weights = zeros / non_zeros
        return pos_weights

    def apply_classical_augmentation(self, image, label):

        augmentation_string = ''
        # Apply rotation
        if random.random() <= self.classical_prob:
            affine_params = transforms.RandomAffine.get_params(degrees=[-10, 10],
                                                               translate=[0.2, 0.2],
                                                               scale_ranges=[1, 1],
                                                               shears=[1, 1],
                                                               img_size=[503, 272])
            image = TF.affine(image,
                              angle=affine_params[0],
                              translate=affine_params[1],
                              scale=affine_params[2],
                              shear=affine_params[3])
            label = TF.affine(label,
                              angle=affine_params[0],
                              translate=affine_params[1],
                              scale=affine_params[2],
                              shear=affine_params[3])

            augmentation_string += " - rotation: {}-{}".format(affine_params[0], affine_params[1])

        # Apply color jitter
        if random.random() <= 0.3:
            color_jitter_transform = transforms.ColorJitter(brightness=0.2)
            image = color_jitter_transform(image)
            augmentation_string += " - jitter"

        # # Apply vertical flip
        # if random.random() <= 0.3:
        #     image = TF.vflip(image)
        #     label = TF.vflip(label)
        #     augmentation_string += " - vflip"

        if random.random() <= 0.3:
            image = TF.hflip(image)
            label = TF.hflip(label)
            augmentation_string += " - hflip"

        return image, label, augmentation_string

    def apply_random_augmentation(self, image, label, filename):

        augmentation_parameters = ""
        if np.random.uniform(low=0.0, high=1.0) <= self.deformation_prob:
            displacement = int(np.random.uniform(30, 100))
            augmentation_parameters += "displ_{}".format(displacement)
            try:
                image, label = self.deformation.execute(image, label, displacement=displacement)
            except:
                augmentation_parameters += "Failed"

        if np.random.uniform(low=0.0, high=1.0) <= self.tilting_prob:

            xy_probe = [int(np.random.uniform(0, 50)), 0]
            alpha_probe = math.pi * np.random.uniform(1, 45)
            augmentation_parameters += "-tilting_[{}, {}]_{}".format(xy_probe[0], xy_probe[1], alpha_probe)
            try:
                image, label = self.tilting.execute(image, label, xy_probe=xy_probe, alpha_probe=-alpha_probe)
            except:
                augmentation_parameters += "Failed"

        if np.random.uniform(low=0.0, high=1.0) <= self.m_reflection_prob:
            reflection_intensity = np.random.uniform(0.50, 0.90)
            augmentation_parameters += "-mreflection_{}".format(reflection_intensity)
            try:
                image, label = self.multiple_reflection.execute(image, label,
                                                                reflection_intensity=reflection_intensity)
            except:
                augmentation_parameters += "Failed"

                print("Multiple Reflection Augmentation failed: " + filename + " - reflection_intensity: " +
                      str(reflection_intensity))

        if np.random.uniform(low=0.0, high=1.0) <= self.noise_prob:
            path = pathlib.Path(filename)
            root = path.parent.parent.parent.parent
            local_energy_name = path.parts[-1].split(".")[0] + "_local_energy.mat"
            local_energy_path = os.path.join(root, "local_energy", local_energy_name)

            bone_signal = np.random.uniform(0.70, 1.40)
            bg_signal = np.random.uniform(0.70, 1.40)
            augmentation_parameters += "-noise_{}_{}".format(bone_signal, bg_signal)
            try:
                image, label = self.noise.execute(image, label, local_energy_path=local_energy_path, bone_signal=bone_signal,
                                                bg_signal=bg_signal)
            except:
                augmentation_parameters += "Failed"
                print("Noise Augmentation failed: " + filename + " local energy path: " + local_energy_path + " - bone signal: " +
                      str(bone_signal) + " - background signal: " + str(bg_signal))

        return image, label, augmentation_parameters
