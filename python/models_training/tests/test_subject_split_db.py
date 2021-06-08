import unittest
from test_utils import create_parser, get_params_dict, val_type_dict, TestDbClass
import os
import datasets.dataset_utils as dt_utils

# todo fix this
class MyTestCase(TestDbClass):

    def setUp(self):
        test_filepath = os.path.abspath(__file__)
        self.root = os.path.join(os.path.split(test_filepath)[0], "test_data")

    def test_SubjectSplitdDb_allSetGiven(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        root_folder = os.path.join(self.root, 'subject_db')

        train_subject_ids = "2, 4, 5,6, 8, 9,  10, 12, 13, 15, 16, 18"
        val_subject_ids = "14, 17, 11, 7"
        test_subject_ids = "1, 3"

        input_params = self.parser.parse_args(['--data_root', root_folder,
                                               '--train_subjects', train_subject_ids,
                                               '--val_subjects', val_subject_ids,
                                               '--test_subjects', test_subject_ids,
                                               '--paired_db', 'True'])

        expected_train_db, expected_val_db, expected_test_db = self._get_expected_file_lists(root_folder)

        method = dt.SubjectSplitDb(hparams=input_params)
        method.prepare_data()

        for split, expected_output in zip(['train', 'val', 'test'],
                                          [expected_train_db, expected_val_db, expected_test_db]):

            self.assertTrue(self.are_list_equal(method.data[split].AB_paths, expected_output))

    def test_SubjectSplitdDb_testValGiven(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')

        val_subject_ids = "14, 17, 11, 7"
        test_subject_ids = "1, 3"

        input_params = self.parser.parse_args(['--data_root', root_folder,
                                               '--val_subjects', val_subject_ids,
                                               '--test_subjects', test_subject_ids,
                                               '--paired_db', 'True'])

        expected_train_db, expected_val_db, expected_test_db = self._get_expected_file_lists(root_folder)

        method = dt.SubjectSplitDb(hparams=input_params)
        method.prepare_data()

        for split, expected_output in zip(['train', 'val', 'test'],
                                          [expected_train_db, expected_val_db, expected_test_db]):

            self.assertTrue(self.are_list_equal(method.data[split].AB_paths, expected_output))

    def test_SubjectSplitdDb_testGiven(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')
        test_subject_ids = "1, 3"

        input_params = self.parser.parse_args(['--data_root', root_folder,
                                               '--test_subjects', test_subject_ids,
                                               '--random_split', '80_20',
                                               '--paired_db', 'True'])

        _, _, expected_test_db = self._get_expected_file_lists(root_folder)

        # as we have 16 subjects in the train_val set - training: 80% of 16 = 12.8 = 13 - val = 16 - 13 = 3
        # 13/16 * 100 - 3/16 * 100%
        expected_train_percentage = 13/16 * 100
        expected_val_percentage =  3/16 * 100

        method = dt.SubjectSplitDb(hparams=input_params)
        method.prepare_data()

        self.assertTrue(self.are_list_equal(method.data['test'].AB_paths, expected_test_db))

        train_sub = dt_utils.get_subject_ids_from_data(method.data['train'].AB_paths)
        val_sub = dt_utils.get_subject_ids_from_data(method.data['val'].AB_paths)

        # checking no subject overlap between val and train
        for item in method.data['val'].AB_paths:
            self.assertFalse(dt_utils.get_id_from_filename(item) in train_sub)
            self.assertFalse(dt_utils.get_id_from_filename(item) in test_subject_ids)

        for item in method.data['train'].AB_paths:
            self.assertFalse(dt_utils.get_id_from_filename(item) in val_sub)
            self.assertFalse(dt_utils.get_id_from_filename(item) in test_subject_ids)

        val_percentage = len(val_sub) / (len(val_sub) + len(train_sub)) * 100
        train_percentage = len(train_sub) / (len(val_sub) + len(train_sub)) * 100

        self.assertTrue(expected_val_percentage == val_percentage)
        self.assertTrue(expected_train_percentage == train_percentage)

    def test_SubjectSplitdDb_noTestGiven(self):
        """
        Testing BoneDb implementation with folder_db, train split
        Returns:
        """
        self.setUp()

        root_folder = os.path.join(self.root, 'subject_db')

        input_params = self.parser.parse_args(['--data_root', root_folder,
                                               '--random_split', '80_10_10',
                                               '--paired_db', 'True'])

        # as we have 18 subjects in the train_val_test set - training: 80% of 18 = 14.4 = 14 - val = 80% of 18 = 1.8 = 2
        # - test = 18 - 14 - 2 = 2
        # 14/18 * 100= 77.77% - 2/18 * 100 - 2/18 * 100%
        expected_train_percentage = 14/18 * 100
        expected_val_percentage = 2/18 * 100
        expected_test_percentage = 2/18 * 100

        method = dt.SubjectSplitDb(hparams=input_params)
        method.prepare_data()

        train_sub = dt_utils.get_subject_ids_from_data(method.data['train'].AB_paths)
        val_sub = dt_utils.get_subject_ids_from_data(method.data['val'].AB_paths)
        test_sub = dt_utils.get_subject_ids_from_data(method.data['test'].AB_paths)

        # checking no subject overlap between val and train
        for item in method.data['val'].AB_paths:
            self.assertFalse(dt_utils.get_id_from_filename(item) in train_sub)
            self.assertFalse(dt_utils.get_id_from_filename(item) in test_sub)

        for item in method.data['train'].AB_paths:
            self.assertFalse(dt_utils.get_id_from_filename(item) in val_sub)
            self.assertFalse(dt_utils.get_id_from_filename(item) in test_sub)

        val_percentage = len(val_sub) / (len(val_sub) + len(train_sub) + len(test_sub)) * 100
        train_percentage = len(train_sub) / (len(val_sub) + len(train_sub) + len(test_sub)) * 100
        test_percentage = len(test_sub) / (len(val_sub) + len(train_sub) + len(test_sub)) * 100

        print(val_percentage, " ", train_percentage, "  ", test_percentage)

        self.assertTrue(expected_val_percentage == val_percentage)
        self.assertTrue(expected_train_percentage == train_percentage)
        self.assertTrue(expected_test_percentage == test_percentage)


if __name__ == '__main__':
    unittest.main()
