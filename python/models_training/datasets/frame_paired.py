from datasets.base_dataset import BasePairedDataset
from PIL import Image
import os
from datasets.dataset_utils import get_params, get_transform


class FramePaired(BasePairedDataset):
    def __init__(self, hparams, split, **kwargs):
        super().__init__(hparams, split, **kwargs)

    def __getitem__(self, idx):
        # read a image given a random integer index
        A_path = self.AB_paths[idx].replace(".png", "_label.png")  # label
        B_path = self.AB_paths[idx]  # image

        image_name = os.path.split(B_path)[-1].replace(".png", "")

        A = Image.open(A_path).convert('LA') if os.path.exists(A_path) else None  # condition
        B = Image.open(B_path).convert('LA') if os.path.exists(B_path) else None  # image

        # apply the same transform to both A and B
        transform_params = get_params(self.hparams, A.size)
        A_transform = get_transform(self.hparams, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.hparams, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A) if A is not None else None
        B = B_transform(B) if B is not None else None

        return {'Image': B,
                'Label': A,
                'ImageName': image_name}