import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import imutils
from shutil import copy2

mouse_x, mouse_y = -1, -1
window_flag = True

def get_image_size(epochs_fold):
    return 100, 100

def get_id_list(epochs_fold):

    labels_names = os.listdir(epochs_fold)
    ids = [item.replace("_label.png", "") for item in labels_names if "label" in item]

    return ids

def pad_image(img, padding):
    return np.pad(img, ((padding, padding), (padding, padding)), constant_values=(255, 255))

def onmouse(event, x, y, flags, param):
    global mouse_x, mouse_y, window_flag
    if event == cv2.EVENT_LBUTTONUP:
        mouse_x, mouse_y = x, y
        window_flag=False
        cv2.destroyWindow("image")

def combine_images(images, rows, cols, padding=5):
    padded_image_list = [pad_image(item, padding=padding) for item in images]

    concatenated_rows = [np.concatenate(padded_image_list[i * cols:i * cols + cols], axis=1) for i in range(rows)]

    concatenated_image = np.concatenate(concatenated_rows, axis=0)
    return concatenated_image.astype(np.uint8)

def get_epoch_id(concatenated_mask, x, y):
    return int(round(concatenated_mask[y, x]))

def add_header(image, mask, header_images, padding=5):
    header_images = [pad_image(item, padding) for item in header_images]
    header = np.concatenate(header_images, axis=1)

    left_padding = (image.shape[1] - header.shape[1]) // 2
    right_padding = image.shape[1] - left_padding - header.shape[1]

    padded_header = np.pad(header, ((padding, padding), (left_padding, right_padding)), constant_values=(255, 255))
    padded_mask = np.ones(padded_header.shape)*250

    result_image = np.concatenate([padded_header, image], axis=0)
    result_mask = np.concatenate([padded_mask, mask], axis=0)

    return result_image, result_mask

def show_results(root, epoch, img_id):
    us_path = os.path.join(root, epoch, img_id + '_sim_us.png')
    ct_path = os.path.join(root, epoch, img_id + '_ct.png')
    label_path = os.path.join(root, epoch, img_id + '_label.png')

    us = cv2.imread(us_path, 0)
    ct = cv2.imread(ct_path, 0)
    label = cv2.imread(label_path, 0)

    plt.subplot(1, 3, 1)
    plt.imshow(label, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(ct, cmap='gray')
    plt.subplot(1, 3, 3)
    plt.imshow(us, cmap='gray')
    plt.show()


def main(args):
    # mouse callback function
    global window_flag, mouse_x, mouse_y

    epochs_folds = os.listdir(args.root)
    id_list = get_id_list(os.path.join(args.root, epochs_folds[0]))

    out_ids = [item.replace("_label", "") for item in os.listdir(args.out_path) if "_label" in item]

    h, w = (512, 512)

    for img_id in id_list:

        # do not look at images already selected in the output folder
        if img_id in out_ids:
            continue

        ct_header = cv2.imread(os.path.join(args.root, epochs_folds[0], img_id + '_ct.png'), 0)
        label_header = cv2.imread(os.path.join(args.root, epochs_folds[0], img_id + '_label.png'), 0)

        img_list = [cv2.imread(os.path.join(args.root, epoch, img_id + '_sim_us.png'), 0) for epoch in epochs_folds]
        epoch_mask = [np.ones([h, w])*epoch for epoch in range(len(epochs_folds))]

        extra_images = args.cols*args.rows - len(img_list)

        for i in range(extra_images):
            img_list.append(np.zeros([h, w]))
            epoch_mask.append(np.ones([h, w])*250)

        concatenated_image = combine_images(img_list, args.rows, args.cols, args.padding)
        concatenated_mask = combine_images(epoch_mask, args.rows, args.cols, args.padding)

        concatenated_image, concatenated_mask = add_header(concatenated_image, concatenated_mask, [ct_header, label_header], padding=5)

        concatenated_image = imutils.resize(concatenated_image, width=1280)
        concatenated_mask = imutils.resize(concatenated_mask, width=1280)

        window_flag = True
        cv2.namedWindow('image')
        cv2.setMouseCallback('image', onmouse, param={'concatenated_image':concatenated_image})

        while window_flag:
            cv2.imshow('image', concatenated_image)
            cv2.waitKey(delay=1)

        epoch_idx = get_epoch_id(concatenated_mask, mouse_x, mouse_y)

        if epoch_idx >= len(epochs_folds):
            print("Image discarded")
            continue
        chosen_epoch = epochs_folds[epoch_idx]

        print(chosen_epoch)

        copy2(src=os.path.join(args.root, chosen_epoch, img_id + '_sim_us.png'),
              dst=os.path.join(args.out_path, img_id + '_sim_us.png'))

        copy2(src=os.path.join(args.root, chosen_epoch, img_id + '_ct.png'),
              dst=os.path.join(args.out_path, img_id + '_ct.png'))

        copy2(src=os.path.join(args.root, chosen_epoch, img_id + '_label.png'),
              dst=os.path.join(args.out_path, img_id + '_label.png'))

        show_results(args.root, chosen_epoch, img_id)


if __name__ == '__main__':

    """
    example usage: 
    name.py --root=E:\\NAS\\outputs\\output_db\\output_db
            --out_path=E:\\Maria\\DataBases\\CT_simUS_2D
            --rows=5
            --cols=5
            --padding=5
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', type=str)
    parser.add_argument('--out_path', type=str)
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--padding', type=int, default=5)

    params = parser.parse_args()

    main(params)
