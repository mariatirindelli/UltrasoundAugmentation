import os
import pandas as pd
from shutil import copy2
import random

database_path = "E:\\Chrissi\\spine\\NAS\\ImFusionClassification\\exported_data"
save_dir = "E:\\Chrissi\\spine\\NAS\\ImFusionClassification\\training_db"
old_db_path = "E:\\Chrissi\\spine\\NAS\\BoneClassification"
corrected_db_path = "E:\\Chrissi\\spine\\NAS\\BoneClassificationCorrected"

generate_cross_val_folders = False
create_imfusion_db = False
split_data = False
delete_unlabelled_data = False
check_db = True

# test_subjects_id = ["0", "1"]
#
# if create_imfusion_db:
#     fid = open( "E:\\Chrissi\\spine\\NAS\\ImFusionClassification\\batch_file.txt", 'w')
#
#     item_list = os.listdir(database_path)
#     item_list = [item.split(".")[0] for item in item_list if ".imf" in item and "labels" not in item]
#
#     fid.write("INPUTPATH;OUTPUTPATH")
#     for file_name in item_list:
#         input_path = os.path.join(database_path, file_name + ".imf")
#         output_path = os.path.join(database_path, file_name + ".png")
#         fid.write("\n" + input_path + ";" + output_path)
#
#     fid.close()
#
#
# if split_data:
#     bone_list_file = os.path.join(database_path, "data_list_training_1.txt")
#
#     pd_frame = pd.read_csv(bone_list_file, delimiter="\t")
#
#     data_paths = pd_frame.iloc[:, 0]
#     original_data_paths = pd_frame.iloc[:, 2]
#     tags_bone = pd_frame.iloc[:, 3]
#     tags_no_bone = pd_frame.iloc[:, 3]
#
#     for data_path, original_data_path, tag_bone, tag_no_bone in zip(data_paths, original_data_paths, tags_bone, tags_no_bone):
#         original_data_id = os.path.split(original_data_path)[-1].strip(".png")  # e.g. 0_33 (-> subjectId_number)
#         data_class = "bone" if tag_bone else "no_bone"
#         imf_data_id = os.path.split(data_path)[-1].strip(".imf")  # e.g. 0
#
#         png_data_path = os.path.join(database_path, imf_data_id + ".png")
#         save_path = os.path.join(save_dir, data_class, original_data_id + ".png")
#
#         print(png_data_path, "  ", save_path)
#
#         copy2(png_data_path, save_path)
#
#
# def get_split_subjects(path, split):
#     all_files = [item for item in os.listdir(os.path.join(path, split, "bone"))]
#     all_files.extend([item for item in os.listdir(os.path.join(path, split, "no_bone"))])
#
#     sub_list = list(set([item.split("_")[0] for item in all_files]))
#     return sub_list
#
# def print_unlabelled_data(data_path, split):
#     labeled_data = os.listdir( os.path.join(data_path, "label"))
#     bone_data = os.listdir( os.path.join(data_path, "bone"))
#
#     missing = 0
#     total = 0
#     for item in bone_data:
#         total += 1
#         if item not in labeled_data:
#             # print("Missing label for: {} - {}".format(split, item))
#             missing += 1
#
#     print("--------------- {}/{} missing in {}".format(missing, total, split))
#
#
# # generate_cross_val_folders
# if generate_cross_val_folders:
#
#     for i in range(5):
#         old_cross_val_dir = os.path.join(old_db_path, "cross_val" + str(i))
#         current_crossval_dir = os.path.join(corrected_db_path, "cross_val" + str(i))
#
#         print(current_crossval_dir)
#
#         for split in ["train", "val"]:
#             split_sub = get_split_subjects(old_cross_val_dir, split)
#             # print(split_sub)
#
#             # bones
#             for file_name in os.listdir( os.path.join(save_dir, "bone")):
#                 sub_id = file_name.split("_")[0]
#                 if sub_id not in split_sub:
#                     continue
#
#                 if sub_id == '3':
#                     continue
#
#                 src_path = os.path.join(save_dir, "bone", file_name)
#                 dst_path = os.path.join(current_crossval_dir, split, "bone", file_name)
#
#                 #print(src_path, "  ", dst_path)
#                 copy2(src_path, dst_path)
#
#             print_unlabelled_data(os.path.join(current_crossval_dir, split), split)
#
#
#             # no bones
#             for file_name in os.listdir(os.path.join(save_dir, "no_bone")):
#                 sub_id = file_name.split("_")[0]
#                 if sub_id not in split_sub:
#                     continue
#                 if sub_id == '3':
#                     continue
#
#                 src_path = os.path.join(save_dir, "no_bone", file_name)
#                 dst_path = os.path.join(current_crossval_dir, split, "no_bone", file_name)
#
#                 # print(src_path, "  ", dst_path)
#                 copy2(src_path, dst_path)
#
#
# if delete_unlabelled_data:
#
#     for i in range(5):
#         current_crossval_dir = os.path.join(corrected_db_path, "cross_val" + str(i))
#
#         print(current_crossval_dir)
#
#         for split in ["train", "val", "test"]:
#             current_path = os.path.join(current_crossval_dir, split)
#
#             labeled_data = os.listdir(os.path.join(current_path, "label"))
#             bone_data = os.listdir(os.path.join(current_path, "bone"))
#
#             for item in bone_data:
#
#                 if item not in labeled_data:
#                     file_to_remove = os.path.join(current_path, "bone", item)
#                     print("Removing: {}".format(file_to_remove))
#                     os.remove(file_to_remove)
#

if check_db:
    for i in range(5):

        current_crossval_dir = os.path.join(corrected_db_path, "cross_val" + str(i))
        print("-----------------Cross Val: {}".format(current_crossval_dir))

        for split in ["train", "val", "test"]:
            current_path = os.path.join(current_crossval_dir, split)
            labeled_data = os.listdir(os.path.join(current_path, "label"))

            bone_data = os.listdir(os.path.join(current_path, "bone"))
            non_bone_data = os.listdir(os.path.join(current_path, "no_bone"))

            for item in bone_data:
                if item not in labeled_data:
                    missing_file = os.path.join(current_path, "bone", item)
                    print("Missing file: {}".format(missing_file))

            print("training (bone/non_bone): {} - {}".format(len(bone_data), len(non_bone_data)))






