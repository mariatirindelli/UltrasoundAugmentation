import pandas as pd
import os
import numpy as np
import argparse
from PIL import Image
import imageio
from models_training.datasets.dataset_utils import get_subject_ids_from_data, get_subject_based_random_split, \
    get_split_subjects_data

def prepare_imfusion_db_list(input_db_path):
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

def prepare_data_list(input_db_path, db_kind='imfusion_label'):

    if db_kind == 'imfusion_label':
        db_list = prepare_imfusion_db_list(input_db_path)

    return db_list


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

def save_data(data, save_path):
    fmt = save_path.split(".")[-1]

    if fmt == 'tiff':
        imageio.mimwrite(save_path, data)
    else:
        raise NotImplementedError

def apply_mask(image, mask):
    masked_image = image.copy()

    if image.shape != mask.shape:
        mask = np.tile(mask, (image.shape[0], 1, 1))

    masked_image[mask == 0] = 0

    return masked_image

def get_split_paths(data_root):

    data_list = os.listdir(data_root)
    data_list = [item for item in data_list if 'label' not in item]
    sub_ids = get_subject_ids_from_data(data_list)
    train_ids, val_ids, test_ids = get_subject_based_random_split(sub_ids)

    print(train_ids, " ", val_ids, " ", test_ids)

    train_data = get_split_subjects_data(data_list, train_ids)
    val_data = get_split_subjects_data(data_list, val_ids)
    test_data = get_split_subjects_data(data_list, test_ids)

    return {'train': train_data, 'val': val_data, 'test': test_data}

def load_image(image_path, vflip=True):
    if os.path.exists(image_path):
        data_mask = Image.open(image_path)
        if vflip:
            data_mask = data_mask.transpose(Image.FLIP_TOP_BOTTOM)
        data_mask = np.array(data_mask)  # flipping mask as images are saved flipped
    else:
        data_mask=None

    return data_mask


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

