from datasets.dataset_utils import *
from PIL import Image
from torch.utils.data import Dataset
import os
from random import shuffle

class BaseDataset(Dataset):

    def __len__(self):
        """Return the total number of images in the dataset."""
        return NotImplementedError

    def __getitem__(self, idx):
        return NotImplementedError

class BasePairedDataset(Dataset):
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
        self.AB_paths = [item for item in self.AB_paths if "label" not in os.path.split(item)[-1]]

        if data_structure == 'subject_based':
            self.AB_paths = get_split_subjects_data(self.AB_paths, subject_list)

        assert (self.hparams.load_size >= self.hparams.crop_size)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    def __getitem__(self, idx):
        raise NotImplementedError

class BaseUnpairedDataset(Dataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):

        self.hparams = hparams
        self.transform_A = get_transform(self.hparams, grayscale=(self.hparams.input_nc == 1))
        self.transform_B = get_transform(self.hparams, grayscale=(self.hparams.output_nc == 1))

        if data_list is not None:
            self.A_paths = data_list[0]
            self.B_paths = data_list[1]

        if data_structure == 'folder_based':
            self.AB_dir = os.path.join(self.hparams.data_root, split)  # create a path '/path/to/data/trainA'

            self.AB_paths = sorted(make_dataset(self.AB_dir, self.hparams.max_dataset_size))
            self.A_paths = [item for item in self.AB_paths if "label" in os.path.split(item)[-1]]
            self.B_paths = [item for item in self.AB_paths if "label" not in os.path.split(item)[-1]]

        elif data_structure == 'subject_based':

            self.AB_paths = sorted(make_dataset(self.hparams.data_root, self.hparams.max_dataset_size))  # get image paths

            self.A_paths = [item for item in self.AB_paths if "label" in os.path.split(item)[-1]]
            self.A_paths = get_split_subjects_data(self.A_paths, subject_list)

            self.B_paths = [item for item in self.AB_paths if "label" not in os.path.split(item)[-1]]
            self.B_paths = get_split_subjects_data(self.B_paths, subject_list)
            shuffle(self.B_paths)

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __len__(self):
        """Return the total number of images in the dataset."""
        return max(self.A_size, self.B_size)

    def __getitem__(self, idx):
        raise NotImplementedError


class USBonesPaired(BasePairedDataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):
        super().__init__(hparams, split, subject_list, data_list, data_structure)

    def __getitem__(self, idx):
        # read a image given a random integer index
        A_path = self.AB_paths[idx].replace(".png", "_label.png")  # label
        B_path = self.AB_paths[idx]  # image

        image_name = os.path.split(B_path)[-1].replace(".png", "")

        A = Image.open(A_path).convert('LA') if os.path.exists(A_path) else None
        B = Image.open(B_path).convert('LA') if os.path.exists(B_path) else None

        # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A) if A is not None else None
        B = B_transform(B) if B is not None else None

        return B, A, image_name  # return image, label, image_name


class USBonesUnpaired(BaseUnpairedDataset):
    def __init__(self, hparams, split=None, subject_list=None, data_list=None, data_structure='folder_based'):
        super().__init__(hparams, split, subject_list, data_list, data_structure)

    def __getitem__(self, idx):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[idx % self.A_size]  # make sure index is within then range
        if self.hparams.serial_batches:   # make sure index is within then range
            index_B = idx % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A_name = os.path.split(A_path)[-1]
        B_name = os.path.split(B_path)[-1]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        return A, B, A_name, B_name
