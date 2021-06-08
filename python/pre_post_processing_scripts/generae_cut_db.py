import argparse
import os
import shutil
from pre_post_processing_scripts.proc_utils import get_split_paths

def main(params):

    trainA_dir = os.path.join(params.save_path, "trainA")
    trainB_dir = os.path.join(params.save_path, "trainB")

    valA_dir = os.path.join(params.save_path, "valA")
    valB_dir = os.path.join(params.save_path, "valB")

    testA_dir = os.path.join(params.save_path, "testA")
    testB_dir = os.path.join(params.save_path, "testB")

    for dir_path in [trainA_dir, trainB_dir, valA_dir, valB_dir, testA_dir, testB_dir]:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        os.mkdir(dir_path)

    for group_id, data_root in zip(['A', 'B'], [params.A_root, params.B_root]):
        split_dict = get_split_paths(data_root)

        for split in ['train', 'val', 'test']:
            data_paths = [os.path.join(data_root, item) for item in split_dict[split]]
            dest_paths = [os.path.join(params.save_path, split + group_id, item) for item in split_dict[split]]

            for (src_path, dst_path) in zip(data_paths, dest_paths):

                # todo this can be removed
                if "label" in src_path:
                    continue

                shutil.copy(src=src_path,
                            dst=dst_path)


def str2tuple(input_string):
    split_string = input_string.split("-")
    return tuple([float(item) for item in split_string])


if __name__ == '__main__':

    """
    example usage: 
    name.py --A_root=D:\\NAS\\exported_verse_db_6.4.21 
            --B_root=D:\\NAS\\exported_verse_db_6.4.21 
            --save_path=D:\\NAS\\CT_label_db
            --split=80-10-10
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--A_root', type=str)
    parser.add_argument('--B_root', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--split', type=str2tuple)

    args = parser.parse_args()

    main(args)