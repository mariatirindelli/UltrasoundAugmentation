import python.us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
from scipy.interpolate import splprep, splev
import scipy.integrate as integrate
from scipy import signal
from scipy.ndimage import gaussian_filter
import math

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

class ProbeTilting(us.BaseMethod):
    def __init__(self, blur_kernel=45, blur_sigma=20, apply_blur=True):
        super(ProbeTilting, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur

    @staticmethod
    def distance_along_line(p, x1, x2):

        p_der = np.polyder(p)
        fun = lambda x:np.sqrt(1 + p_der(x) ** 2)

        return integrate.quad(fun, x1, x2)[0]



    def split_curve(self, x_list, p):

        x_equi = [x_list[0]]
        for i in range(1, len(x_list)):

            x_old = x_list[i - 1]
            x = x_list[i]

            distance = abs(self.distance_along_line(p, x_old, x))
            print("\n", distance)

            if abs(distance - self.target_distance) <= self.delta:
                x_equi.append(x)
                continue

            iterations = 0
            delta_update = 0.2*self.target_distance

            while iterations < self.max_iter:
                x = x - np.sign(distance - self.target_distance) * delta_update
                distance = abs(self.distance_along_line(p, x_old, x))

                if abs(distance - self.target_distance) <= self.delta:
                    x_equi.append(x)
                    print(distance, "  ", x)
                    break

                iterations+=1
                delta_update = delta_update*0.5

            x_equi.append(x)
            print("max iter reached: ", distance)

        return x_equi

    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def _get_shadow_mask(self, label, view_point):

        y_view_point = view_point[1]
        x_view_point = view_point[0]

        yx_bones = np.argwhere(label==1)

        tl_idx = np.argmin(yx_bones[:, 1])
        x_tl = yx_bones[tl_idx, 1]
        y_tl = yx_bones[tl_idx, 0]

        tr_idx = np.argmax(yx_bones[:, 1])
        x_tr = yx_bones[tr_idx, 1]
        y_tr = yx_bones[tr_idx, 0]

        m1 = (y_tl - y_view_point)/(x_tl - x_view_point)
        b1 = y_view_point - m1*x_view_point

        m2 = (y_tr - y_view_point) / (x_tr - x_view_point)
        b2 = y_view_point - m2 * x_view_point

        bone_shadow = np.zeros(label.shape)
        for y in range(0, label.shape[0]):
            x_l = int((y - b1)/m1)
            x_r = int((y - b2)/m2)

            bone_shadow[y, x_l:x_r] = 1

        for x in range(x_tl, x_tr):
            y_min = np.max(np.argwhere(label[:, x]>0))
            y_min = y_min+40
            bone_shadow[0:y_min, x] = 0

        for x in range(0, x_tl):
            bone_shadow[0:y_tl, x] = 0

        for x in range(x_tr, label.shape[1]):
            bone_shadow[0:y_tr, x] = 0

        # bone_shadow[label>0] = 2
        #
        # plt.imshow(bone_shadow)
        # plt.show()

        return bone_shadow

    def execute(self, image, gt):
        from scipy.interpolate import UnivariateSpline

        label = ProbeTilting._blur(gt, kernel=15, sigma=25)

        x_data = np.argwhere(label > 0)[:, 1]
        image_y_data = np.argwhere(label > 0)[:, 0]
        y_data = label.shape[0] - image_y_data

        regressionLineOrder = 8
        regressionLine = np.polyfit(x_data, y_data, regressionLineOrder)
        p = np.poly1d(regressionLine)

        sorted_idx = np.argsort(x_data)
        x_sorted = x_data[sorted_idx]
        y_sorted = y_data[sorted_idx]

        x_idx_diff = np.diff(x_sorted)
        non_multiple_idx = np.argwhere(x_idx_diff != 0)

        x_sorted = x_sorted[non_multiple_idx]
        y_sorted = y_sorted[non_multiple_idx]

        # pick a random idx
        x_gauss_center = x_sorted[20]

        gaussian_label = np.zeros(label.shape)
        avg_int = np.mean(image[label > 0])
        max_int = np.max(image[label > 0])
        min_int = np.min(image[label > 0])


        for x in x_sorted:
            s = self.distance_along_line(p, x_gauss_center, x)
            y_gauss = self.gaussian(s, 0, 20)

            label_pixels_in_x = np.argwhere(x_data == x)
            y_to_color = image_y_data[label_pixels_in_x]
            x_to_color = x_data[label_pixels_in_x]  # vector [x, x, x, ..]

            gaussian_label[y_to_color, x_to_color] = y_gauss

        blur_label = ProbeTilting._blur(gt, kernel=15, sigma=25)
        augmented_image = blur_label * (image*(gaussian_label + 0.4)) + (1 - blur_label) * image
        #augmented_image = blur_label * (image + gaussian_label*(max_int - min_int)) + (1 - blur_label)*image

        shadow_mask = self._get_shadow_mask(gt, [0, 0])

        shadow_mask = gaussian_filter(shadow_mask, [5, 10], order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

        augmented_image = augmented_image*(1 - shadow_mask) + shadow_mask

        plt.subplot(2, 2, 1)
        plt.imshow(label)
        plt.subplot(2, 2, 2)
        plt.imshow(gaussian_label)
        plt.subplot(2, 2, 3)
        plt.imshow(image, cmap='gray')
        plt.subplot(2, 2, 4)
        plt.imshow(augmented_image, cmap='gray')
        plt.show()


if __name__ == '__main__':

    root = "C:\\Users\\maria\\OneDrive\\Desktop\\us_augmentation"

    img = cv2.imread(os.path.join(root, "test.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(root, "test_label.png"), cv2.IMREAD_GRAYSCALE)

    method = ProbeTilting()
    #method._get_shadow_mask(gt, [0, 0])
    method.execute(img, gt)