from shutil import copy2
import logging
import argparse
import os


def save_db(src_dir, dst_dir):

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    src_files = os.listdir(src_dir)

    for file in src_files:
        src_filepath = os.path.join(src_dir, file)

        if "manual" in file:
            file = file.replace("manual", "label")

        dst_filepath = os.path.join(dst_dir, file)
        copy2(src_filepath, dst_filepath)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--src_path', type=str, help='path to the imfusion db folder where data were exported')
    parser.add_argument('--dst_path', type=str, help='Folder where the db will be saved')

    args = parser.parse_args()

    logger = logging.getLogger('prepare_segmentation')
    logger.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)

    save_db(src_dir=args.src_path,
            dst_dir=args.dst_path)

    logger.handlers.clear()
