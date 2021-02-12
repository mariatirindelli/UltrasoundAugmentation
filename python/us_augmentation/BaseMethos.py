import cv2
import numpy as np

class BaseMethod():
    def __init__(self):
        self.name = ""

    @staticmethod
    def _blur(image, kernel, sigma):

        if np.max(image) <= 1:
            image = image * 255

        blur_image = cv2.GaussianBlur(image * 255, (kernel, kernel), sigma)
        blur_image = blur_image / np.max(blur_image)
        return blur_image

    @staticmethod
    def _get_bone_edges(label):
        yx_bones = np.argwhere(label == 1)

        tl_idx = np.argmin(yx_bones[:, 1])
        x_tl = yx_bones[tl_idx, 1]
        y_tl = yx_bones[tl_idx, 0]

        tr_idx = np.argmax(yx_bones[:, 1])
        x_tr = yx_bones[tr_idx, 1]
        y_tr = yx_bones[tr_idx, 0]

        return [x_tl, y_tl], [x_tr, y_tr]

    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def execute(self, image, label):
        raise NotImplementedError

