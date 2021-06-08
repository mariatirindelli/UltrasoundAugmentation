import unittest
from test_utils import create_parser, get_params_dict, val_type_dict, TestDbClass
from datasets.base_dataset import BasePairedDataset, BaseUnpairedDataset
import os


class TestBaseDbPaired(TestDbClass):

    def test_data_list_in_kwargs(self):

        params_dict = get_params_dict()
        params_dict['data_root'] = val_type_dict('data_root', str)

        parser = create_parser(params_dict)
        hparams = parser.parse_args()

        data_list = ['7_29_17.png',
                     '5_22_36.png',
                     '15_76_7.png',
                     '13_67_4.png',
                     '5_21_18.png',
                     '14_72_51.png',
                     '11_60_47.png',
                     '7_28_21.png',
                     '9_44_0.png']

        for split in ['train', 'val', 'test']:
            base_paired_db = BasePairedDataset(hparams, split=split, data_list=data_list)
            self.assetEqualList(data_list, base_paired_db.AB_paths)

    def test_subject_list_in_kwargs(self):

        data_root = 'test_data\\subject_db'

        params_dict = get_params_dict()
        params_dict['data_root'] = val_type_dict(data_root, str)

        parser = create_parser(params_dict)
        hparams = parser.parse_args()

        subject_list = self.train_subjects
        expected_AB_items = self.train_data_list
        expected_AB_paths = [os.path.join(data_root, item) for item in expected_AB_items]

        base_paired_db = BasePairedDataset(hparams, split='train', subject_list=subject_list)
        self.assetEqualList(expected_AB_paths, base_paired_db.AB_paths)

    def test_data_structure_in_kwargs(self):
        data_root = 'test_data\\folder_paired_db'

        params_dict = get_params_dict()
        params_dict['data_root'] = val_type_dict(data_root, str)

        parser = create_parser(params_dict)
        hparams = parser.parse_args()

        for split in ['train', 'val', 'test']:

            expected_AB_items = getattr(self, split + '_data_list')
            expected_AB_paths = [os.path.join(data_root, split, item) for item in expected_AB_items]

            base_paired_db = BasePairedDataset(hparams, split=split, data_structure='folder_based')
            self.assetEqualList(expected_AB_paths, base_paired_db.AB_paths)

class TestBaseDbUnpaired(TestDbClass):
    def test_data_list_in_kwargs(self):

        params_dict = get_params_dict()
        params_dict['data_root'] = val_type_dict('data_root', str)

        parser = create_parser(params_dict)
        hparams = parser.parse_args()

        data_list = [
            ['1_1_46.png',
             '8_37_26.png',
             '9_47_30.png',
             '14_72_67.png',
             '4_17_21.png',
             '14_72_23.png'],
            ['14_72_63.png',
             '3_10_11.png',
             '8_35_4.png',
             '11_60_9.png']]

        for split in ['train', 'val', 'test']:
            base_unpaired_db = BaseUnpairedDataset(hparams, split=split, data_list=data_list)
            self.assetEqualList(data_list[0], base_unpaired_db.A_paths)
            self.assetEqualList(data_list[1], base_unpaired_db.B_paths)

    def test_subject_list_in_kwargs(self):
        data_root = 'test_data\\subject_db'

        params_dict = get_params_dict()
        params_dict['data_root'] = val_type_dict(data_root, str)

        parser = create_parser(params_dict)
        hparams = parser.parse_args()

        subject_list = self.train_subjects
        expected_B_items = self.train_data_list
        expected_B_paths = [os.path.join(data_root, item) for item in expected_B_items]
        expected_A_paths = [item.replace(".png", "_label.png") for item in expected_B_paths]

        base_unpaired_db = BaseUnpairedDataset(hparams, split='train', subject_list=subject_list)

        self.assetEqualList(expected_A_paths, base_unpaired_db.A_paths)
        self.assetEqualList(expected_B_paths, base_unpaired_db.B_paths)


if __name__ == '__main__':

    unittest.main()


