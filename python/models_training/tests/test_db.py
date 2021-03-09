import unittest
import argparse
import datasets.PairedPixelwiseDb as dt
import os
import numpy as np


class MyTestCase(unittest.TestCase):

    @staticmethod
    def _create_parser():
        parser = argparse.ArgumentParser(...)
        parser.add_argument("--data_root", default='', type=str)
        parser.add_argument("--output_nc", default=1, type=int)
        parser.add_argument("--input_nc", default=1, type=int)
        parser.add_argument("--batch_size", default=1, type=int)

        parser.add_argument('--preprocess', default='resize_and_crop', type=str)
        parser.add_argument('--load_size', default=256, type=int)
        parser.add_argument('--crop_size', default=256, type=int)
        parser.add_argument('--max_dataset_size', default=np.inf, type=float)
        parser.add_argument("--train_subjects", default='', type=str)
        parser.add_argument("--val_subjects", default='', type=str)
        parser.add_argument("--test_subjects", default='', type=str)
        parser.add_argument("--random_split", default='80_10_10', type=str)
        # ...Create your parser as you like...
        return parser

    def setUp(self):
        self.parser = self._create_parser()
        test_filepath = os.path.abspath(__file__)
        self.root = os.path.join(os.path.split(test_filepath)[0], "test_data")

    def test_BoneDb_folder_split_train(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'folder_db')
        split = 'test'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(root_folder, split))
        expected_output = [os.path.join(root_folder, split, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            split=split,
                            data_structure='folder_based')

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_BoneDb_folder_split_val(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'folder_db')
        split = 'val'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(root_folder, split))
        expected_output = [os.path.join(root_folder, split, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            split=split,
                            data_structure='folder_based')

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_BoneDb_folder_split_test(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'folder_db')
        split = 'test'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(root_folder, split))
        expected_output = [os.path.join(root_folder, split, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            split=split,
                            data_structure='folder_based')

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_BoneDb_subject_split_train(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')
        train_subject_ids = ["2", "4", "5", "6", "8", "9", "10", "12", "13", "15", "16", "18"]
        val_subject_ids = ["14", "17", "11", "7"]
        split = 'train'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(self.root, 'folder_db', split))
        expected_output = [os.path.join(root_folder, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            data_structure='subject_based',
                            subject_list=train_subject_ids)

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_BoneDb_subject_split_val(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')
        val_subject_ids = ["14", "17", "11", "7"]
        split = 'val'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(self.root, 'folder_db', split))
        expected_output = [os.path.join(root_folder, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            data_structure='subject_based',
                            subject_list=val_subject_ids)

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_BoneDb_subject_split_test(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')
        test_subject_ids = ["1", "3"]
        split = 'test'
        input_params = self.parser.parse_args(['--data_root', root_folder])

        expected_output = os.listdir(os.path.join(self.root, 'folder_db', split))
        expected_output = [os.path.join(root_folder, item) for item in expected_output if "label" not in item]

        method = dt.USBones(hparams=input_params,
                            data_structure='subject_based',
                            subject_list=test_subject_ids)

        for item in method.AB_paths:
            self.assertTrue(item in expected_output)

        for item in expected_output:
            self.assertTrue(item in method.AB_paths)

    def test_SubjectSplitdDb(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')

        train_subject_ids = "2, 4, 5,6, 8, 9,  10, 12, 13, 15, 16, 18"
        val_subject_ids = "14, 17, 11, 7"
        test_subject_ids = "1, 3"

        input_params = self.parser.parse_args(['--data_root', root_folder,
                                               '--train_subjects', train_subject_ids,
                                               '--val_subjects', val_subject_ids,
                                               '--test_subjects', test_subject_ids])

        expected_train_db = os.listdir(os.path.join(self.root, 'folder_db', 'train'))
        expected_train_db = [os.path.join(root_folder, item) for item in expected_train_db if "label" not in item]

        expected_val_db = os.listdir(os.path.join(self.root, 'folder_db', 'val'))
        expected_val_db = [os.path.join(root_folder, item) for item in expected_val_db if "label" not in item]

        expected_test_db = os.listdir(os.path.join(self.root, 'folder_db', 'test'))
        expected_test_db = [os.path.join(root_folder, item) for item in expected_test_db if "label" not in item]

        method = dt.SubjectSplitdDb(hparams=input_params)
        method.prepare_data()

        for split, expected_output in zip(['train', 'val', 'test'],
                                          [expected_train_db, expected_val_db, expected_test_db] ):

            for item in method.data[split].AB_paths:
                self.assertTrue(item in expected_output)

            for item in expected_output:
                self.assertTrue(item in method.data[split].AB_paths)



if __name__ == '__main__':
    unittest.main()
