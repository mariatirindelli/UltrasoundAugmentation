import os
import numpy as np
import logging
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import cv2
import utils
import pandas as pd
from PIL import Image

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def image_with_colorbar(fig, ax, image, cmap = None, title=""):

    if cmap is None:
        pos0 = ax.imshow(image, clim=(0, 1))
    else:
        pos0 = ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(title)
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

    dice = numerator / denominator
    return dice

def get_data_to_plot(root, experiment_id_list, worst_k=10, best_k=10):
    files_to_plot = []
    for experiment_id in experiment_id_list:
        if not os.path.exists(os.path.join(root, experiment_id, "test_results.csv")):
            print("Not existing experiment: {}".format(experiment_id))
            continue

        result_df = pd.read_csv(os.path.join(root, experiment_id, "test_results.csv"))

        dice_scores = result_df["dice_scores"]
        if isinstance(dice_scores[0], str):
            result_list = [item.strip("tensor(") for item in dice_scores]
            dice_scores = [float(item.split(",")[0]) for item in result_list]

        worst_results = np.argsort(dice_scores)
        filenames_worst = result_df["filenames"][worst_results[0:worst_k]]
        if len(filenames_worst) > 0:
            files_to_plot.extend(filenames_worst)

        filenames_best = result_df["filenames"][worst_results[-best_k::]]
        if len(filenames_best) > 0:
            files_to_plot.extend(filenames_best)

    return list(set(files_to_plot))

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_data(root, data_name, experiments_configs, experiment_ids, params, savepath="", return_pil_image=False):

    num_subplots = len(experiment_ids) + 3  # + 3 for the label and image and binary predictions

    subplot_cols = np.ceil(num_subplots * 2 / 3).astype(int)
    subplot_rows = np.ceil(num_subplots/subplot_cols).astype(int)

    fig, axs = plt.subplots(subplot_rows, subplot_cols, figsize=(15,15))

    label = None
    image = None

    for i, experiment_id in enumerate(experiment_ids):
        config = experiments_configs[experiment_id]

        title = ""
        for param in params:
            title += param + ": \n" + str(config[param]) + "\n"

        data_path = os.path.join(root, experiment_id, data_name.split(".")[0])

        if image is None:
            image = np.load(data_path + "_image.npy")
            if subplot_cols == 1 or subplot_rows == 1:
                image_with_colorbar(fig, axs[0], image, cmap='gray', title="Image")
            else:
                image_with_colorbar(fig, axs[0][0], image, cmap='gray', title="Image")

        if label is None:
            label = np.load(data_path + "_gt.npy")
            if subplot_cols == 1 or subplot_rows == 1:
                image_with_colorbar(fig, axs[1], label, cmap=None, title="GT Label")
            else:
                image_with_colorbar(fig, axs[0][1], label, cmap=None, title="GT Label")

        predictions = sigmoid(np.load(data_path + "_pred.npy"))
        binary_pred = np.where(predictions >= 0.5, 1, 0)
        current_dice = dice_score(label, binary_pred)
        title += "\nDice: {:.2f}".format(current_dice)
        if subplot_cols == 1 or subplot_rows == 1:
            image_with_colorbar(fig, axs[i + 2], binary_pred, cmap=None, title=title)
        else:
            row, col = np.unravel_index(i+2, [subplot_rows, subplot_cols])
            image_with_colorbar(fig, axs[row][col], binary_pred, cmap=None, title=title)

    for k in range(len(experiment_ids), subplot_rows*subplot_cols):
        if subplot_cols == 1 or subplot_rows == 1:
            axs[k].set_axis_off()
        else:
            row, col = np.unravel_index(k, [subplot_rows, subplot_cols])
            axs[row][col].set_axis_off()

    fig.tight_layout()

    if not return_pil_image:
        fig.savefig(savepath)
        plt.close(fig)

    else:
        pil_image = fig2img(fig)
        plt.close(fig)
        return pil_image


def main(params):

    config_params, experiments_configs = utils.get_dicts_and_varying_params(params.group_path)
    dict_list = [experiments_configs[key] for key in experiments_configs.keys()]
    cross_val_folders = utils.get_cross_fold_path(dict_list)

    for cross_val_folder in cross_val_folders:

        save_folder = os.path.join(params.output_save_folder, os.path.split(cross_val_folder)[-1])
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        experiment_ids = [key for key in experiments_configs.keys()
                                        if experiments_configs[key]["data_root"] == cross_val_folder]

        # plotting worst and best data
        data_to_plot = get_data_to_plot(params.group_path, experiment_ids)

        for data in data_to_plot:
            save_path = os.path.join(save_folder, data)
            plot_data(params.group_path, data, experiments_configs, experiment_ids, config_params.keys(), save_path)

        # generating gif with all data
        data_to_plot = os.listdir(os.path.join(params.group_path, experiment_ids[0]))
        data_to_plot = [item for item in data_to_plot if "_image.npy" in item]
        data_to_plot = [item.replace("_image.npy", "") for item in data_to_plot]
        images = []

        for data in data_to_plot:
            save_path = os.path.join(save_folder, data)
            pil_image = plot_data(params.group_path, data, experiments_configs, experiment_ids,
                      config_params.keys(), save_path, return_pil_image=True)
            images.append(pil_image)
        print("saving gif ..." + save_folder)
        images[0].save(os.path.join(save_folder, 'image_results.gif'),
                       save_all=True, append_images=images[1:], optimize=False, duration=500, loop=0)
        print("saved")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--group_path', default='Z:\\outputs1\\mariatirindelli\\BonesSegmentation\\groups\\1943', type=str)
    parser.add_argument('--metric', default='dice', type=str)
    parser.add_argument('--output_save_folder', default='', type=str)

    args = parser.parse_args()
    if args.output_save_folder == '':
        args.output_save_folder = args.group_path + "_results"

    main(args)
