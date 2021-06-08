import argparse
import numpy as np
from BaseTest import BaseTest

def val_type_dict(val, data_type):
    return {'val':val, 'type':data_type}

def get_params_dict():

    params_dict = {
        'output_nc': val_type_dict(1, int),
        'input_nc': val_type_dict(1, int),
        'preprocess': val_type_dict('resize_and_crop', str),
        'load_size': val_type_dict(256, int),
        'crop_size': val_type_dict(256, int),
        'max_dataset_size': val_type_dict(np.inf, float),
    }
    return params_dict

def create_parser(data_dict):
    parser = argparse.ArgumentParser()

    for key in data_dict:
        parser.add_argument('--' + key, default=data_dict[key]['val'], type=data_dict[key]['type'])

    return parser


class TestDbClass(BaseTest):
    def __init__(self, *args, **kwargs):
        super(TestDbClass, self).__init__(*args, **kwargs)

        self.train_subjects = ['2', '4', '5', '6', '8', '9', '10', '12', '13', '15', '16', '18']
        self.train_data_list = [
            '2_7_0.png', '2_9_4.png', '2_9_6.png',
            '4_15_41.png', '4_15_8.png', '4_16_23.png', '4_16_97.png',
            '5_21_12.png', '6_23_15.png', '6_23_17.png', '6_23_3.png',
            '6_23_48.png', '6_27_184.png',
            '8_34_22.png', '8_35_2.png', '9_43_46.png',
            '9_43_50.png',
            '10_48_11.png', '10_48_17.png', '10_48_7.png', '10_49_11.png', '10_49_12.png', '10_49_3.png',
            '12_61_0.png', '12_62_13.png',
            '13_66_100.png', '13_66_99.png', '13_67_3.png', '13_67_4.png',
            '15_76_1.png', '15_76_6.png', '15_76_7.png',
            '16_77_13.png', '16_77_14.png', '16_77_5.png',
            '18_84_0.png', '18_84_2.png']

        self.val_subjects = ['7', '11', '14', '17']
        self.val_data_list = [
            '7_28_21.png', '7_28_3.png', '7_29_14.png', '7_29_17.png',
            '11_59_48.png', '11_60_9.png',
            '14_71_39.png', '14_71_7.png', '14_72_23.png', '14_72_36.png', '14_72_51.png', '14_72_63.png',
            '17_80_2.png']

        self.test_subjects = ['1', '3']
        self.test_data_list = [
            '1_1_0.png', '1_1_10.png', '1_1_12.png', '1_1_25.png', '1_1_9.png',
            '3_10_11.png', '3_10_14.png', '3_10_5.png', '3_10_9.png']

