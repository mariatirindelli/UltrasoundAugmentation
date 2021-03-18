import pandas as pd
import os


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
