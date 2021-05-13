import argparse
import os
import shutil
import datetime


def copy_files(data_path, save_path, save_prefix):
    a_ids = os.listdir(data_path) if os.path.exists(data_path) else []

    for item in a_ids:
        src_dir = os.path.join(data_path, item)
        dst_dir = os.path.join(save_path, save_prefix + item)
        shutil.copy(src_dir, dst_dir)

def main(params):

    if not os.path.exists(params.save_folder):
        os.mkdir(params.save_folder)

    for data_path, save_prefix in zip([params.a_path, params.b_path], [params.a_prefix, params.b_prefix]):
        copy_files(data_path, params.save_folder, save_prefix)

if __name__ == '__main__':

    """
    example usage: 
    name.py --root=D:\\NAS\\exported_verse_db_6.4.21 
            --save_path=D:\\NAS\\CT_label_db
            --target_size=516, 544
            --target_spacing=0.145228, 0.145228
            --data_mask=D:\\Maria\\DataBases\\SpineIFL\\curvilinear_mask_7cm.png
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--a_path', type=str)
    parser.add_argument('--b_path', type=str)
    parser.add_argument('--a_prefix', default='', type=str)
    parser.add_argument('--b_prefix', default='', type=str)
    parser.add_argument('--save_folder', type=str)

    args = parser.parse_args()

    main(args)