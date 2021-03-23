import pandas as pd
import os
import numpy as np
import argparse

def prepare_data_list(input_db_path, output_save_path, logger):

    # 1. Read data_list.txt - the .txt file saved by ImFusionLabels in the export folder
    imf_data_list_path = os.path.join(input_db_path, "data_list.txt")
    pd_frame = pd.read_csv(imf_data_list_path, sep="\t")

    return_list = []
    for i in range(len(pd_frame.index)):

        input_image_path = os.path.join(input_db_path, pd_frame['#dataPath'][i])
        input_label_path = os.path.join(input_db_path, pd_frame['labelPath'][i])
        subject_name = os.path.split(pd_frame['originalDataPath'][i])[-1]

        return_list.append([input_image_path, input_label_path, subject_name])

    return return_list


def get_sub_id_from_filename(filename):
    """
    Assuming that the filename has the form subjectId_imageId.fmt, it extracts and reutnrs the subjectId
    Args:
        filename:

    Returns:
    """

    # if the input id a file path, only keep the filename
    filename = os.path.split(filename)[-1]
    sub_id = filename.split("_")[0]
    return sub_id

def get_sweep_id_from_filename(filename):
    """
    Assuming that the filename has the form subjectId_imageId.fmt, it extracts and reutnrs the subjectId
    Args:
        filename:

    Returns:
    """

    # if the input id a file path, only keep the filename
    filename = os.path.split(filename)[-1]
    sweep_id = filename.split("_")[1]
    return sweep_id

def get_image_id_from_filename(filename):
    # if the input id a file path, only keep the filename
    filename = os.path.split(filename)[-1]
    image_id = filename.split("_")[2]
    return image_id

def sort_images(image_names_list):
    image_id_list = [int(get_image_id_from_filename(item)) for item in image_names_list]
    sort_idx = np.argsort(image_id_list)

    sorted_image_list = [image_names_list[item] for item in sort_idx]
    return sorted_image_list

def images2sweep(file_list):

    sub_ids = list(set([get_sub_id_from_filename(item) for item in file_list]))

    data_dict = dict()

    for sub_id in sub_ids:
        subject_files = [item for item in file_list if get_sub_id_from_filename(item) == sub_id]

        sweeps_ids = list(set([get_sweep_id_from_filename(item) for item in subject_files]))
        data_dict[sub_id] = dict()

        for sweep_id in sweeps_ids:
            sweep_files = [item for item in subject_files if get_sweep_id_from_filename(item) == sweep_id]
            sorted_sweep_files = sort_images(sweep_files)
            data_dict[sub_id][sweep_id] = sorted_sweep_files

    return data_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

