import numpy as np
from PIL import Image
from fix_pythonpath import *
import logging
from db_generation_utils import prepare_data_list
import argparse
import imfusion
import matplotlib.pyplot as plt
imfusion.init()

def get_name_id_dict(subject_name_list, start_sub_id=0):
    name_id_dict = dict()

    for i, subject in enumerate(subject_name_list):
        name_id_dict[subject] = str(i + start_sub_id)

    return name_id_dict


def log_dic(dictionary, file_logger):
    for key in dictionary.keys():
        file_logger.info("subject name: {} - subject id: {}".format(key, dictionary[key]))

def save_png(data_array, path):
    im = Image.fromarray(data_array)
    im = im.transpose(Image.FLIP_TOP_BOTTOM )
    im.save(path)

def save_db(save_path, imfusion_exported_data_path, start_sub_id=0, rescale_labels=False, data_mask=None,
            file_logger=None):

    data_list = prepare_data_list(input_db_path=imfusion_exported_data_path,
                                  output_save_path=save_path,
                                  logger=logger)

    subjects_name_list = list(set([item[-1] for item in data_list]))
    name_id_dict = get_name_id_dict(subjects_name_list, start_sub_id=start_sub_id)

    log_dic(name_id_dict, file_logger)

    for (sweep_path, label_path, subject_name) in data_list:

        if not os.path.exists(sweep_path) or not os.path.exists(label_path):
            print("File {} or Label: {} does not exits".format(sweep_path, label_path))
            continue

        subject_id = name_id_dict[subject_name]

        sweep = imfusion.open(sweep_path)[0]
        labels = imfusion.open(label_path)[0]

        sweeep_id = os.path.split(sweep_path)[-1].strip(".imf")

        print(sweep_path, "  ", label_path)

        for iterator, (image, label) in enumerate(zip(sweep, labels)):

            image_array = np.array(image).astype(np.uint8)
            label_array = np.array(label).astype(np.uint8)

            if rescale_labels:
                label_array[label_array == 1] = 250
                label_array[label_array == 0] = 250

            if np.sum(label_array) == 0:
                continue

            if data_mask is not None:
                image_array[data_mask == 0] = 0
                label_array[data_mask == 0] = 0

            image_save_path = os.path.join(save_path, subject_id + "_" + sweeep_id + "_" +
                                           str(iterator) + ".png")
            label_save_path = os.path.join(save_path,  subject_id + "_" + sweeep_id + "_" +
                                           str(iterator) + "_label.png")

            print(image_save_path, " ", label_save_path)
            save_png(np.squeeze(image_array), image_save_path)
            save_png(np.squeeze(label_array), label_save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--imfusion_db_path', type=str, help='path to the imfusion db folder where data were exported')
    parser.add_argument('--save_path', type=str, help='Folder where the db will be saved')
    parser.add_argument('--start_sub_id', type=int, default=0,
                        help='The first sub_id. e.g. if start_sub_id=5, all sub_id will be higher than 5')
    parser.add_argument('--data_mask', type=str, default="",
                        help='The first sub_id. e.g. if start_sub_id=5, all sub_id will be higher than 5')

    args = parser.parse_args()

    logger = logging.getLogger('prepare_segmentation')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(args.save_path, "log_info.log"))
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    if os.path.exists(args.data_mask):
        data_mask = Image.open(args.data_mask)
        data_mask = data_mask.transpose(Image.FLIP_TOP_BOTTOM)
        data_mask = np.array(data_mask)  # flipping mask as images are saved flipped
    else:
        data_mask=None

    save_db(save_path=args.save_path,
            imfusion_exported_data_path=args.imfusion_db_path,
            file_logger=logger,
            start_sub_id=args.start_sub_id,
            data_mask=data_mask)

    logger.handlers.clear()
