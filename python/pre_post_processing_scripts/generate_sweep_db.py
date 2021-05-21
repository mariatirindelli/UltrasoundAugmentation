from fix_pythonpath import *
import logging
from proc_utils import prepare_data_list, apply_mask, save_data, load_image
import argparse
import imfusion
import data_loader as dl
imfusion.init()

def get_name_id_dict(subject_name_list, start_sub_id=0):
    name_id_dict = dict()

    for i, subject in enumerate(subject_name_list):
        name_id_dict[subject] = str(i + start_sub_id)

    return name_id_dict

def log_dic(dictionary, file_logger):
    for key in dictionary.keys():
        file_logger.info("subject name: {} - subject id: {}".format(key, dictionary[key]))

def save_db(param, mask, file_logger):

    data_list = prepare_data_list(input_db_path=param.data_path,
                                  db_kind=param.db_kind)

    subjects_name_list = list(set([item[-1] for item in data_list]))
    name_id_dict = get_name_id_dict(subjects_name_list)

    log_dic(name_id_dict, file_logger)

    for (sweep_path, label_path, subject_name) in data_list:

        if not os.path.exists(sweep_path) or not os.path.exists(label_path):
            print("File {} or Label: {} does not exits".format(sweep_path, label_path))
            continue

        subject_id = name_id_dict[subject_name]
        sweep_id = os.path.split(sweep_path)[-1].strip(".imf")

        sweep, _ = dl.load_data(subject_id)
        labels, _ = dl.load_data(label_path)

        sweep = apply_mask(sweep, mask)
        labels = apply_mask(labels, mask)

        save_data(data=sweep,
                  save_path=os.path.join(param.save_path,
                                         subject_id + "_" + sweep_id + "." + param.fmt))

        save_data(data=labels,
                  save_path=os.path.join(param.save_path,
                                         subject_id + "_" + sweep_id + "_label." + param.fmt))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Converts any kind of database in a sequence of images')
    parser.add_argument('--imfusion_db_path', type=str, help='path to the imfusion db folder where data were exported')
    parser.add_argument('--save_path', type=str, help='Folder where the db will be saved')
    parser.add_argument('--db_kind', type=str)
    parser.add_argument('--fmt', default='tiff', type=str)
    parser.add_argument('--data_mask', type=str, default="")

    args = parser.parse_args()

    logger = logging.getLogger('prepare_segmentation')
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(args.save_path, "log_info.log"))
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    data_mask = load_image(args.data_mask)

    save_db(param=args,
            mask=data_mask,
            file_logger=logger)

    logger.handlers.clear()
