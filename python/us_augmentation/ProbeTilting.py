import python.us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.integrate as integrate
from scipy.ndimage import gaussian_filter
import math


class ProbeTilting(us.BaseMethod):
    def __init__(self,
                 label_blur_kernel=45,
                 label_blur_sigma=20,
                 intensity_gaussian_sigma=20,
                 intensity_blur_sigmas=(5, 0.1),
                 intensity_rescaling_factor=2,
                 intensity_offset=0.4,
                 shadow_mask_blur_sigmas=(5, 10),
                 shadow_start=40,
                 angle_weight=0.7,
                 distance_weight=0.3,
                 dilation_kernel=(10, 10),
                 apply_blur=True,
                 mode='gaussian',
                 plot_result=True):

        super(ProbeTilting, self).__init__()
        self.apply_blur = apply_blur
        self.label_blur_kernel = label_blur_kernel
        self.label_blur_sigma = label_blur_sigma

        self.intensity_gaussian_sigma = intensity_gaussian_sigma

        self.intensity_blur_sigmas = [intensity_blur_sigmas[0], intensity_blur_sigmas[1]]
        self.intensity_rescaling_factor = intensity_rescaling_factor,
        self.intensity_offset = intensity_offset

        self.angle_weight = angle_weight
        self.distance_weight = distance_weight

        self.shadow_mask_blur_sigmas = [shadow_mask_blur_sigmas[0], shadow_mask_blur_sigmas[1]]
        self.shadow_start = shadow_start

        self.dilation_kernel = np.ones([dilation_kernel[0], dilation_kernel[1]])

        # For now not used parameters
        self.max_distance_error = 0.05
        self.mode = mode  # can be 'curve_gaussian', 'distance_angle' - it defines how intensities are rescaled on the
        # bone segmentation
        self.plot_result = plot_result

    def execute(self, image, label, xy_probe=(0, 0), alpha_probe=0):

        label = cv2.dilate(label, self.dilation_kernel, iterations=1)

        # 1. Fitting polynomial to label map
        p = self.fit_polynomial_to_label(label)

        # 2. Sorting bone pixels to remove multiple x entries and sort the x array in ascenfing order
        x_data = np.argwhere(label > 0)[:, 1]
        y_data = np.argwhere(label > 0)[:, 0]
        x_sorted, y_sorted = self._sort_label_idx(x_data, y_data, remove_duplicates=True)

        # For each label point find the distance from the probe and the angle between us waves and bone
        bone_probe_distance = self._get_bone2probe_distance(xy_probe, x_sorted, p)
        bone_probe_angles = self._get_bone2probe_angle(alpha_probe, x_sorted, p)

        intensity_rescaling_mask = np.zeros(label.shape)
        x_gauss_center = x_sorted[np.argmin(bone_probe_distance)]

        for (x, dist, angle) in zip(x_sorted, bone_probe_distance, bone_probe_angles):

            if self.mode == 'gaussian':
                s = self.distance_along_line(p, x_gauss_center, x)
                y_intensity = self.gaussian(s, 0, 40)

            elif self.mode == 'distance_angle':
                # TODO: check max dist better
                normalized_distance = abs(dist) / (math.sqrt(label.shape[0] ** 2 + label.shape[1] ** 2))
                normalized_angle = angle/(math.pi/2)

                y_intensity = self.distance_weight*normalized_distance + self.angle_weight*normalized_angle

            else:
                raise ValueError("Unknown mode")

            label_pixels_in_x = np.argwhere(x_data == x)
            y_to_color = y_data[label_pixels_in_x]
            x_to_color = x_data[label_pixels_in_x]  # vector [x, x, x, ..]
            intensity_rescaling_mask[y_to_color, x_to_color] = y_intensity

        intensity_rescaling_mask = gaussian_filter(intensity_rescaling_mask, sigma=self.intensity_blur_sigmas) \
                                   * self.intensity_rescaling_factor + self.intensity_offset

        blur_label = self._blur(gt, kernel=self.label_blur_kernel, sigma=self.label_blur_sigma)

        augmented_image = blur_label * (image*intensity_rescaling_mask) + (1 - blur_label) * image
        shadow_mask = self._get_shadow_mask(gt, xy_probe)
        shadow_mask = gaussian_filter(shadow_mask, sigma=self.shadow_mask_blur_sigmas)

        augmented_image = augmented_image*(1 - shadow_mask) + shadow_mask

        if self.plot_result:
            normalized_distance = np.abs(bone_probe_distance) / (math.sqrt(label.shape[0] ** 2 + label.shape[1] ** 2))
            normalized_angle = np.array(bone_probe_angles) / (math.pi / 2)

            y_intensity_vec = self.distance_weight * normalized_distance + self.angle_weight * normalized_angle

            plt.subplot(1, 3, 1)
            self._plot_intensity_maps(label, x_sorted, bone_probe_distance)
            plt.subplot(1, 3, 2)
            self._plot_intensity_maps(label, x_sorted, bone_probe_angles)
            plt.subplot(1, 3, 3)
            self._plot_intensity_maps(label, x_sorted, y_intensity_vec)

            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(label)
            plt.subplot(2, 3, 2)
            plt.imshow(intensity_rescaling_mask)
            plt.subplot(2, 3, 4)
            plt.imshow(shadow_mask)
            plt.subplot(2, 3, 5)
            plt.imshow(image, cmap='gray')
            plt.subplot(2, 3, 6)
            plt.imshow(augmented_image, cmap='gray')
            plt.show()

    def split_curve(self, x_list, p, target_distance, max_iter=1000):
        """
        Splits the curve in equidistant samples along the polynomial curve
        """

        x_equidistant = [x_list[0]]
        for (x_old, x) in zip(x_list[::-1], x_list[1::]):

            iterations = 0

            while iterations < max_iter:
                distance = abs(self.distance_along_line(p, x_old, x))
                if abs(distance - target_distance) <= self.max_distance_error:
                    break

                delta_update = 0.2 * distance
                x = x - np.sign(distance - target_distance) * delta_update
                iterations += 1

            x_equidistant.append(x)
            print("iterations: ", iterations)

        return x_equidistant

    def _get_shadow_mask(self, label, view_point):

        y_view_point = view_point[1]
        x_view_point = view_point[0]

        [x_tl, y_tl], [x_tr, y_tr] = self._get_bone_edges(label)

        m1 = (y_tl - y_view_point)/(x_tl - x_view_point)
        b1 = y_view_point - m1*x_view_point

        m2 = (y_tr - y_view_point) / (x_tr - x_view_point)
        b2 = y_view_point - m2 * x_view_point

        bone_shadow = np.zeros(label.shape)

        for y in np.arange(label.shape[0] - 1, 0, step=-1):

            x1 = max(int((y - b1)/m1), 0)
            x2 = min(int((y - b2)/m2), label.shape[1])

            if x1 == x2 == 0:
                continue
            if x1 == x2 == label.shape[1]:
                continue

            bone_shadow[y, x1:x2] = 1

        bone_shadow[0:y_tl, 0:x_tl] = 0
        bone_shadow[0:y_tr, x_tr::] = 0

        for x in range(x_tl, x_tr):
            y = np.max(np.argwhere(label[:, x] > 0))
            bone_shadow[0:y, x] = 0

        return bone_shadow

    @staticmethod
    def distance_along_line(p, x1, x2):
        """
        Computes the distance between x1 and x2 along the polynomial curve p
        """

        p_der = np.polyder(p)

        def fun(x):
            return np.sqrt(1 + p_der(x) ** 2)

        return integrate.quad(fun, x1, x2)[0]

    @staticmethod
    def fit_polynomial_to_label(label, polynomial_order=3):

        x_data = np.argwhere(label > 0)[:, 1]
        y_data = np.argwhere(label > 0)[:, 0]

        regressionLine = np.polyfit(x_data, y_data, polynomial_order)
        p = np.poly1d(regressionLine)
        return p

    @staticmethod
    def _plot_intensity_maps(label, x_bone, intensities):
        image_to_plot = np.zeros(label.shape)

        for x, intensity in zip(x_bone, intensities):
            y = np.argwhere(label[:, x] == 1)
            image_to_plot[y, x] = intensity

        plt.imshow(image_to_plot)

    @staticmethod
    def _sort_label_idx(x, y, remove_duplicates=True, y_reduction='first'):
        """
        Given a set of data x, y it sort it such that x is increasing. If remove_duplicates is True, multiple values in
        x are removed, to obtain a strictly increasing vector. If remove_duplicates is True, the y value in
        correspondence with duplicate is averaged over multiple x if y_reduction is averaged, or only the first y
        value is kept if y_reduction is 'first'.
        """
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx].flatten()
        y_sorted = y[sorted_idx].flatten()

        if not remove_duplicates:
            return x_sorted, y_sorted

        x_idx_diff = np.diff(x_sorted)
        x_idx_diff = np.insert(x_idx_diff, 0, 1, axis=0)  # Adding a one as the derivative is one element shifted
        non_multiple_idx = np.argwhere(x_idx_diff != 0)
        x_no_duplicates = x_sorted[non_multiple_idx].flatten()
        y_no_duplicates = y_sorted[non_multiple_idx].flatten()

        if y_reduction == 'None':
            return x_sorted, y_sorted

        multiple_x_values = x_sorted[x_idx_diff == 0]
        multiple_x_values = list(set(multiple_x_values.tolist()))

        y_no_duplicates = y_no_duplicates.astype(np.float)
        for x in multiple_x_values:
            x_no_duplicates_idx = np.argwhere(x_no_duplicates == x).flatten()
            x_all_duplicate_idx = np.argwhere(x_sorted == x).flatten()

            y = np.mean(y_sorted[x_all_duplicate_idx])
            y_no_duplicates[x_no_duplicates_idx] = y

        return x_no_duplicates, y_no_duplicates

    @staticmethod
    def _line2point_distance(tilting_angle, xy_probe, x_points, y_points):
        """
        """

        assert x_points.size == y_points.size, "x_points and y_points must have the same length"

        m_probe = math.tan(tilting_angle)  # the m slope of the line defined by the transducer elements
        b_probe = xy_probe[1] - m_probe*xy_probe[0]

        distance_list = []
        for (xp, yp) in zip(x_points, y_points):
            d_p2probe = abs(yp - (m_probe*xp + b_probe)) / math.sqrt(1 + m_probe**2)
            distance_list.append(d_p2probe)

        return distance_list

    @staticmethod
    def _point2point_distance(xy_p1, xy_p2):

        d = math.sqrt((xy_p2[0] - xy_p1[0])**2 + (xy_p2[1] - xy_p1[1])**2)

        return d

    @staticmethod
    def _get_bone2probe_distance(xy_probe, x_bone, p, plot=False):
        distance_list = []
        for xp in x_bone:
            yp = p(xp)
            d = ProbeTilting._point2point_distance(xy_probe, [xp, yp])
            distance_list.append(d)

        if plot:

            x1 = 0
            x2 = max(np.max(x_bone), xy_probe[0]) + 10

            y1 = 0
            y2 = max(np.max(p(x_bone)), xy_probe[1]) + 10

            image = np.zeros([int(y2 - y1), int(x2 - x1)])

            for x, d in zip(x_bone, distance_list):
                y_to_color = int(p(x))
                image[y_to_color - 3:y_to_color+3, x] = d

            image[xy_probe[1]:xy_probe[1] + 3, xy_probe[0]:xy_probe[0] + 3] = 255

            plt.imshow(image)
            plt.show()

        return distance_list

    @staticmethod
    def _get_bone2probe_angle(alpha_probe, x_bone, p, plot=False):

        dp = np.polyder(p)

        angle_list = []
        m_probe = math.tan(alpha_probe)

        if alpha_probe != 0:
            m_us_rays = -1/m_probe
        else:
            m_us_rays = math.inf
        for xp in x_bone:
            m_t = dp(xp)

            if alpha_probe != 0:
                tan_alpha = abs((m_us_rays - m_t) / (1 + m_us_rays * m_t))
                alpha = math.atan(tan_alpha)

            else:
                alpha = math.pi/2 - math.atan(abs(m_t))

            angle_list.append(alpha)

        if plot:

            count = 0
            image = np.zeros([int(np.max(p(x_bone)) + 10), int(np.max(x_bone) + 10)])
            for x, angle in zip(x_bone, angle_list):

                y_to_color = int(p(x))
                image[y_to_color - 3:y_to_color+3, x] = angle

                yb = 0
                if m_us_rays == math.inf:
                    xb = x

                else:
                    b_line = p(x) - m_us_rays*x
                    xb = (yb - b_line) / m_us_rays

                if count % 10 == 0:
                    cv2.line(image, (x, y_to_color), (int(xb), yb), 1, 1)
                count += 1

            plt.imshow(image)
            plt.show()

        return angle_list


if __name__ == '__main__':

    root = "C:\\Users\\maria\\OneDrive\\Desktop\\us_augmentation"

    img = cv2.imread(os.path.join(root, "test.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(root, "test_label.png"), cv2.IMREAD_GRAYSCALE)

    method = ProbeTilting(mode='distance_angle')
    method.execute(img, gt, xy_probe=[10, 0], alpha_probe=-math.pi * 0.30)
