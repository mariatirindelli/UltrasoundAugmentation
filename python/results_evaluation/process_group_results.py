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

def get_label_pred_couples(label_c, pred_c):

    distance_matrix = np.zeros([len(label_c), len(pred_c)])
    for row, label_centroid in enumerate(label_c):
        for col, pred_centroid in enumerate(pred_c):
            distance_matrix[row, col] = \
                np.sqrt((label_centroid[0] - pred_centroid[0])**2 + (label_centroid[1] - pred_centroid[1])**2)

    couples_list = []

    for _ in range(len(pred_c)):

        min_idx = np.argmin(distance_matrix)
        best_couple = np.unravel_index(min_idx, shape=distance_matrix.shape)

        if distance_matrix[best_couple[0], best_couple[1]] == 100000:

            return couples_list

        couples_list.append( [best_couple[0] + 1, best_couple[1] + 1] )   # label id, pred id

        distance_matrix[best_couple[0], :] = 100000
        distance_matrix[:, best_couple[1]] = 100000

    return couples_list  # label id, pred id


def hausdorff_score(gt, pred):
    _, gt_connected_components = cv2.connectedComponents(gt.astype(np.uint8))
    _, pred_connected_components = cv2.connectedComponents(pred.astype(np.uint8))

    if np.sum(pred) == 0:
        #print("returning none")
        return None

    # getting connected components centroids
    pred_centroids = [np.mean(np.argwhere(pred_connected_components == i), axis=0) for i in range(1, np.max(pred_connected_components) + 1)]
    gt_centroids = [np.mean(np.argwhere(gt_connected_components == i), axis=0) for i in
                    range(1, np.max(gt_connected_components) + 1)]
    label_pred_couples = get_label_pred_couples(gt_centroids, pred_centroids)

    hd_list = []
    for item in label_pred_couples:
        label_connected_component = np.zeros(gt.shape)
        label_connected_component[gt_connected_components == item[0]] = 1

        pred_connected_component = np.zeros(pred.shape)
        pred_connected_component[pred_connected_components == item[1]] = 1
        gt_coordinates = np.argwhere(label_connected_component > 0)
        pred_coordinates = np.argwhere(pred_connected_component > 0)

        hd = max(directed_hausdorff(gt_coordinates, pred_coordinates)[0],
                 directed_hausdorff(pred_coordinates, gt_coordinates)[0])

        hd_list.append(hd)

        # if hd > 100:
        #     plt.subplot(2, 2, 1)
        #     plt.imshow(label_connected_component)
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(pred_connected_component)
        #
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(gt)
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(pred)
        #     plt.show()


    return np.mean(hd_list)


def evaluate_experiment(root, experiment_id, metric):

    # dice metric for each image is already saved in the csv file after testing, no need to re-compute it

    if metric == 'dice':
        if not os.path.exists(os.path.join(root, experiment_id, "test_results_rec.csv")):
            return []
        result_df = pd.read_csv(os.path.join(root, experiment_id, "test_results_rec.csv"))
        return result_df["dice_scores"], []

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
        if hd is not None:
            hd_list.append(hd)
        #hd_list.append(-1)

        #print(hd)
        filename_list.append(image_name.replace("_image.npy", ""))
        print("\r Progress: {0:.1f}%".format(i/num_files * 100), end="")

    df = pd.DataFrame({'filenames': filename_list, 'dice_scores':dice_list})
    df.to_csv(os.path.join("E:\\NAS_Maria\\output_chrissi\\2105", experiment_id,  "test_results_rec.csv"))
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

        print("experiment id: {} - dice score {}".format(experiment, mean_dice))
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
