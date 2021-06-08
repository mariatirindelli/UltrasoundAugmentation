import plotly.graph_objs as go
import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R
import os
from random import sample
import argparse
from pre_post_processing_scripts.fix_pythonpath import *
import imfusion
imfusion.init()


def pre_process_sweep(data):
    imfusion.executeAlgorithm('Ultrasound;Remove Duplicate Frames', [data])
    spacing = data[0].spacing
    sweep = np.squeeze(np.array(data))

    meta_data = {'spacing': spacing}
    return sweep, meta_data


def process_subject(data_path, sub_id, save_path):

    imf_data = imfusion.open(data_path)

    for i, data in enumerate(imf_data):
        sweep_id = str(i)
        sweep, meta_data = pre_process_sweep(data)

        # transpose and rotate to fit the way CT are saved
        sitk_array = np.transpose(sweep, [2, 1, 0])
        sitk_array = np.rot90(sitk_array, axes=(1, 2), k=2)

        sitk_image = sitk.GetImageFromArray(sitk_array)
        sitk_image.SetSpacing([meta_data['spacing'][2], meta_data['spacing'][0], meta_data['spacing'][1]])

        sitk.WriteImage(sitk_image, os.path.join(save_path, sub_id + "_" + sweep_id + ".mhd"))


def main(params):
    data_list = os.listdir(params.data_root)
    data_list = [item for item in data_list if ".imf" in item]

    for i, filename in enumerate(data_list):
        process_subject(data_path=os.path.join(params.data_root, filename),
                        sub_id=str(i),
                        save_path=params.save_dir)


if __name__ == '__main__':
    """
    example usage: 
    name.py --data_root=E:\\NAS\\exported_verse_db_6.4.21
            --save_dir=E:\\NAS\\data\\sweep_wise\\CT_sweep
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--save_dir', type=str)
    args = parser.parse_args()

    for item in imfusion.availableAlgorithms():
        print(item)

    main(args)
