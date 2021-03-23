import os
import numpy as np
import logging
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
import utils
import pandas as pd
from scipy.spatial.distance import directed_hausdorff

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def image_with_colorbar(fig, ax, image, cmap = None):

    if cmap is None:
        pos0 = ax.imshow(image)
    else:
        pos0 = ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    ax.set_title('Image')
    divider = make_axes_locatable(ax)
    cax0 = divider.append_axes("right", size="5%", pad=0.05)
    tick_list = np.linspace(np.min(image), np.max(image), 5)
    cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
    cbar0.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar

def generate_plot(image_list, cmaps):
    fig, axs = plt.subplots(1, len(image_list), constrained_layout=True)

    for i, _ in enumerate(image_list):
        image_with_colorbar(fig, axs[i], image_list[i], cmaps[i])

    fig.tight_layout()

def dice_score(gt, pred):

    intersection = pred[gt > 0]

    numerator = 2 * np.count_nonzero(intersection)
    denominator = np.count_nonzero(gt) + np.count_nonzero(pred)

    if denominator == 0:
        return np.nan

    dice = numerator / denominator
    return dice

def hausdorff_score(gt, pred):
    gt_coordinates = np.argwhere(gt > 0)
    pred_coordinates = np.argwhere(pred > 0)

    hd = max(directed_hausdorff(gt_coordinates, pred_coordinates)[0], directed_hausdorff(pred_coordinates, gt_coordinates)[0])
    return hd


def evaluate_experiment(root, experiment_id, metric):

    # dice metric for each image is already saved in the csv file after testing, no need to re-compute it
    if metric == 'dice':
        if not os.path.exists(os.path.join(root, experiment_id, "test_results.csv")):
            return []
        result_df = pd.read_csv(os.path.join(root, experiment_id, "test_results.csv"))
        return result_df["dice_scores"]

    experiment_path = os.path.join(root, experiment_id)

    file_list = os.listdir(experiment_path)
    file_list = [item for item in file_list if ".npy" in item and "image" in item]

    dice_list = []
    hd_list = []
    filename_list = []

    num_files = len(file_list)
    for i, image_name in enumerate(file_list):

        image_path = os.path.join(experiment_path, image_name)
        label_path = os.path.join(experiment_path, image_name.replace("image", "gt"))
        pred_path = os.path.join(experiment_path, image_name.replace("image", "pred"))

        image = np.load(image_path)
        label = np.load(label_path)
        pred = sigmoid(np.load(pred_path))
        binary_pred = np.where(pred > 0.5, 1, 0)
        dice = dice_score(label, binary_pred)
        dice_list.append(dice)

        hd = hausdorff_score(label, binary_pred)
        hd_list.append(hd)

        filename_list.append(image_name.replace("_image.npy", ""))
        print("\r Progress: {0:.1f}%".format(i/num_files * 100), end="")

    df = pd.DataFrame({'filenames': filename_list, 'dice_scores':dice_list, 'hd_scores': hd})
    df.to_csv(os.path.join("D:\\NAS\\output\\1988", experiment_id,  "test_results_rec.csv"))
    print("\n")
    return dice_list, hd_list

def print_group_parameters(cross_val_group_id, cross_val_group_dict):
    print("---------------------------------------------------")
    print("Group: {}".format(cross_val_group_id))
    for param in cross_val_group_dict.keys():
        if param == "CrossValGroup":
            continue
        print("\t{} : {}".format(param, cross_val_group_dict[param]))
    print("---------------------------------------------------")

def evaluate_group_results(group_path, group_experiments, metric):

    experiments_result_dice = []
    experiments_result_hd = []
    for i, experiment in enumerate(group_experiments):

        result_dice, result_hd = evaluate_experiment(group_path, experiment, metric)
        if len(result_dice) == 0:
            print("No results for experiment: {}".format(experiment))
            continue
        if isinstance(result_dice[0], str):
            result_list = [item.strip("tensor(") for item in result_dice]
            result_list = [float(item.split(",")[0]) for item in result_list]
            mean_dice = np.mean(result_list)
        else:
            mean_dice = np.mean(result_dice)
            mean_hd = np.mean(result_hd)
        experiments_result_dice.append(mean_dice)
        experiments_result_hd.append(mean_hd)

    print("5-fold cross validation Dice - Mean: ", np.mean(experiments_result_dice), " Std: ", np.std(experiments_result_dice))
    logging.info("5-fold cross validation Dice - Mean: ", np.mean(experiments_result_dice), " Std: ", np.std(experiments_result_dice))

    print("5-fold cross validation Hd - Mean: ", np.mean(experiments_result_hd), " Std: ", np.std(experiments_result_hd))
    logging.info("5-fold cross validation Hd - Mean: ", np.mean(experiments_result_hd), " Std: ", np.std(experiments_result_hd))

    return np.mean(experiments_result_dice), np.std(experiments_result_dice), np.mean(experiments_result_hd), np.std(experiments_result_hd)

def main(params):

    cross_val_groups = utils.get_cross_val_groups(params.group_path)

    # create the dataframe
    pd_data = dict()
    row_labels = []
    for i, cross_val_group_id in enumerate(cross_val_groups.keys()):

        current_group = cross_val_groups[cross_val_group_id]
        print_group_parameters(cross_val_group_id, current_group)
        mean_dice, std_dice, mean_hd, std_hd = evaluate_group_results(group_path=params.group_path,
                                                     group_experiments=current_group["cross_val_folders_ids"],
                                                     metric=params.metric)

        current_group["meandDice"] = mean_dice
        current_group["stdDice"] = std_dice

        current_group["meandHd"] = mean_hd
        current_group["stdHd"] = std_hd

        # saving parameters and results for the group in the dataframe:
        row_labels.append("Experiment: " + str(i))
        for param in current_group.keys():
            if param not in pd_data.keys():
                pd_data[param] = [current_group[param]]
            else:
                pd_data[param].append(current_group[param])

    df = pd.DataFrame(data=pd_data, index=row_labels)

    if not os.path.exists(params.group_path + "_results"):
        os.mkdir(params.group_path + "_results")
    df.to_csv(os.path.join(params.group_path + "_results", "metrics.csv"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--group_path', default='', type=str)
    parser.add_argument('--metric', default='dice-rec', type=str)

    args = parser.parse_args()

    main(args)
