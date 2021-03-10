import os
import numpy as np
import logging
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
import utils
import pandas as pd
from sklearn.metrics import f1_score



def print_group_parameters(cross_val_group_id, cross_val_group_dict):
    print("---------------------------------------------------")
    print("Group: {}".format(cross_val_group_id))
    for param in cross_val_group_dict.keys():
        if param == "CrossValGroup":
            continue
        print("\t{} : {}".format(param, cross_val_group_dict[param]))
    print("---------------------------------------------------")

def evaluate_experiment(root, experiment_id):

    if not os.path.exists(os.path.join(root, experiment_id, "metrics.csv")):
        return []
    result_df = pd.read_csv(os.path.join(root, experiment_id, "metrics.csv"))

    full_res = pd.read_csv(os.path.join(root, experiment_id, "test_results.csv"))
    y_pred = full_res["y_pred"]
    y_true = full_res["y_true"]

    F1 = f1_score(y_true, y_pred, pos_label=0)

    return result_df["accuracy"], F1


def evaluate_group_results(group_path, group_experiments):

    experiment_result_accuracy = []
    experiment_result_f1 = []

    for i, experiment in enumerate(group_experiments):
        result_accuracy, f1 = evaluate_experiment(group_path, experiment)
        if len(result_accuracy) == 0:
            print("No results for experiment: {}".format(experiment))
            continue
        mean_accuracy = np.mean(result_accuracy)
        mean_f1 = np.mean(f1)
        experiment_result_accuracy.append(mean_accuracy)
        experiment_result_f1.append(mean_f1)

    print("5-fold cross validation accuracy - Mean: ", np.mean(experiment_result_accuracy), " Std: ", np.std(experiment_result_accuracy))
    logging.info("5-fold cross accuracy - Mean: ", np.mean(experiment_result_accuracy), " Std: ", np.std(experiment_result_accuracy))

    print("5-fold cross validation F1 - Mean: ", np.mean(experiment_result_f1), " Std: ", np.std(experiment_result_f1))
    logging.info("5-fold cross F1 Dice - Mean: ", np.mean(experiment_result_f1), " Std: ", np.std(experiment_result_f1))

    return np.mean(experiment_result_accuracy), np.std(experiment_result_accuracy)

def main(params):

    cross_val_groups = utils.get_cross_val_groups(params.group_path)

    # create the dataframe
    for i, cross_val_group_id in enumerate(cross_val_groups.keys()):

        current_group = cross_val_groups[cross_val_group_id]
        print_group_parameters(cross_val_group_id, current_group)

        mean_accuracy, std_accuracy = evaluate_group_results(group_path=params.group_path,
                                                     group_experiments=current_group["cross_val_folders_ids"])

        current_group["meanAccuracy"] = mean_accuracy
        current_group["stdAccuracy"] = std_accuracy

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--group_path', default='', type=str)
    parser.add_argument('--metric', default='acc', type=str)

    args = parser.parse_args()

    main(args)
