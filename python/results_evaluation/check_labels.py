import cv2
import numpy as np
from skimage import data, color, io, img_as_float
import os
import matplotlib.pyplot as plt

def apply_blur(label, kernel_size=15, sigma=10):

    blur_label = cv2.GaussianBlur(label * 255, (15, 15), 10)
    blur_label = blur_label / np.max(blur_label)

    return blur_label

def visualize(image, label):

    alpha = 0.6
    rows, cols = label.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    color_mask[..., 0] = label  # Red block

    img_color = np.dstack((image, image, image)).astype(np.float)

    img_color[..., 0] = label*254 * (3/5) + img_color[..., 0]
    img_color[img_color > 254] = 254

    img_color = img_color.astype(np.uint8)

    # Display the output
    f, (ax0, ax1, ax2) = plt.subplots(1, 3, subplot_kw={'xticks': [], 'yticks': []})

    ax0.imshow(image, cmap='gray')
    ax1.imshow(img_color)
    ax2.imshow(label)
    plt.show()


def main(root = "Z:\\data1\\BoneSegmentation\\linear_probe\\cross_val0\\train"):
    image_folder = os.listdir(os.path.join(root, "images"))
    for image in image_folder:
        image_path = os.path.join(root, "images", image)
        label_path = os.path.join(root, "labels", image.replace(".", "_label."))

        image = io.imread(image_path)
        label = io.imread(label_path)

        blur_label = apply_blur(label)

        visualize(image, blur_label)


main()
