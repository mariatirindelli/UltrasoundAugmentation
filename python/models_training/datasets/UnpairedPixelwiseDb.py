from utils.utils import get_argparser_group
from torch.utils.data import Dataset
from datasets.dataset_utils import *
from datasets.PairedPixelwiseDb import BaseDbModule, BasedModuleChildMeta
from abc import ABC
from utils.utils import str2bool

class UnpairedFolderSplitDb(BaseDbModule, ABC, metaclass=BasedModuleChildMeta):

    def __init__(self, hparams):
        super().__init__(hparams)

    def prepare_data(self):

        for split in ['train', 'val', 'test']:
            self.data[split] = UnalignedDataset(self.hparams, split)

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

        return parser

class UnalignedDataset(Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, hparams, split):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.hparams=hparams

        self.dir_A = os.path.join(self.hparams.data_root, split + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(self.hparams.data_root, split + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, self.hparams.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, self.hparams.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.transform_A = get_transform(self.hparams, grayscale=(self.hparams.input_nc == 1))
        self.transform_B = get_transform(self.hparams, grayscale=(self.hparams.output_nc == 1))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.hparams.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
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

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
