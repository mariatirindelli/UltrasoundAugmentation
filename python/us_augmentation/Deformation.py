import python.us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

class Deformation(us.BaseMethod):
    def __init__(self,
                 blur_kernel=95,
                 blur_sigma=20,
                 apply_blur=True,
                 max_displacement=1,
                 dilation_kernel=(10, 10),
                 plot_result=True,
                 lateral_transition_sigma = 50):
        super(Deformation, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur
        self.max_displacement = max_displacement
        self.plot_result = plot_result
        self.dilation_kernel = np.ones([dilation_kernel[0], dilation_kernel[1]])
        self.lateral_transition_sigma = lateral_transition_sigma

    @staticmethod
    def _deform(image, deformation_field):

        mapx_base, mapy_base = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        map_x = mapx_base
        map_y = mapy_base + deformation_field

        deformed_image = cv2.remap(img, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_CUBIC)
        return deformed_image

    @staticmethod
    def _linear_decay(initial_value, final_value, num_samples):
        return np.linspace(initial_value, final_value, num_samples)

    @staticmethod
    def _constant_decay(constant_value, num_samples):
        return np.ones(num_samples) * constant_value

    def _gaussian_transition(self, intensity_left, intensity_right, direction, max_len, displacement_gradient):

        if direction == 'right':
            #x = np.linspace(0, max_len, max_len)
            x = np.linspace(0, 500, 500)
            a = intensity_right
            b = intensity_left
        else:
            x = np.linspace(-500 + 1, 0, 500, endpoint=True)
            a = intensity_left
            b = intensity_right

        g = self.gaussian(x, 0, self.lateral_transition_sigma)

        norm_g = (b - a) * (g - np.min(g)) / (np.max(g) - np.min(g)) + a

        if direction == 'right':
            x_start_idxs = np.argwhere(np.abs(norm_g - intensity_right) < displacement_gradient).flatten()
            if x_start_idxs.size == 0:
                x_start_idx = max_len
            else:
                x_start_idx = min(x_start_idxs[0], max_len)

            gaussian_profile = norm_g[0:x_start_idx]

        elif direction == 'left':
            x_start_idxs = np.argwhere(np.abs(norm_g - intensity_left) < displacement_gradient).flatten()
            if x_start_idxs.size == 0:
                x_start_idx = 0
            else:
                x_start_idx = max(x_start_idxs[0], 500 - max_len)
            gaussian_profile = norm_g[x_start_idx::]

        # else:
        #
        #
        #
        # if x_start_idxs.size == 0:
        #     if direction == 'right':
        #         x_start_idx = max_len
        #     else:
        #         x_start_idx = 0
        # else:
        #     if direction == 'right':
        #         x_start_idx = x_start_idxs[0] + 1
        #     else:
        #         x_start_idx = x_start_idxs[-1] + 1
        #
        # if direction == 'right':
        #     gaussian_profile = norm_g[0:x_start_idx]
        # else:
        #     gaussian_profile = norm_g[x_start_idx::]

        return gaussian_profile

    def _get_bone_profile(self, label, mode='upper'):

        [x_tl, _], [x_tr, _] = self._get_bone_edges(label)
        y_edge = []
        x_upper_edge = range(x_tl, x_tr)

        for x in x_upper_edge:
            if mode == 'upper':
                y_edge.append(np.min(np.argwhere(label[:, x] > 0)))
            elif mode == 'lower':
                y_edge.append(np.max(np.argwhere(label[:, x] > 0)))
            elif mode == 'middle':
                y_edge.append(np.mean(np.argwhere(label[:, x] > 0)))

        return x_upper_edge, y_edge

    def get_region_below_bone(self, label):

        x_upper_profile, y_upper_profile = self._get_bone_profile(label, mode='lower')
        under_bone_mask = np.zeros(label.shape)

        for (x, y) in zip(x_upper_profile, y_upper_profile):
            under_bone_mask[y::, x] = 1

        return under_bone_mask

    def get_region_above_bone(self, label):

        x_upper_profile, y_upper_profile = self._get_bone_profile(label, mode='upper')
        above_bone_mask = np.zeros(label.shape)

        for (x, y) in zip(x_upper_profile, y_upper_profile):
            above_bone_mask[0:y, x] = 1

        return above_bone_mask

    def add_displacement_above_bone(self, label, displacement_map, initial_displacement, final_displacement,
                                    displacement_decay_function='linear'):

        x_lower_profile_idx,  y_lower_profile_idx = self._get_bone_profile(label, mode='upper')
        for (x, y) in zip(x_lower_profile_idx, y_lower_profile_idx):

            if displacement_decay_function == 'linear':
                displacement_profile = self._linear_decay(initial_displacement, final_displacement, y)
            elif displacement_decay_function == 'constant':
                displacement_profile = self._constant_decay(initial_displacement, label.shape[0] - y)
            else:
                continue

            displacement_map[0:y, x] = displacement_profile

        return displacement_map

    def add_displacement_below_bone(self, label, displacement_map, initial_displacement, final_displacement,
                                    displacement_decay_function = 'linear'):

        x_lower_profile_idx,  y_lower_profile_idx = self._get_bone_profile(label, mode='lower')
        for (x, y) in zip(x_lower_profile_idx, y_lower_profile_idx):

            if displacement_decay_function == 'linear':
                displacement_profile = self._linear_decay(initial_displacement, final_displacement, label.shape[0] - y)

            elif displacement_decay_function == 'constant':
                displacement_profile = self._constant_decay(initial_displacement, label.shape[0] - y)
            else:
                continue

            displacement_map[y::, x] = displacement_profile

        return displacement_map

    # TODO: rename better
    def apply_smoothed_horizontal_transition(self, displacement_field, x_initial,
                                             transition_region_gradient=0.1,  direction='left', mode='linear'):

        if direction == 'left':
            sign_transition = np.sign(np.sum(displacement_field[:, x_initial]) - np.sum(displacement_field[:, x_initial - 1]))
        elif direction == 'right':
            sign_transition = np.sign(
                np.sum(displacement_field[:, x_initial + 1]) - np.sum(displacement_field[:, x_initial]))
        else:
            raise ValueError("direction can be either left or right")

        for y in range(displacement_field.shape[0]):

            if direction == 'left':

                x_right = x_initial
                max_transition_area = x_right
                value_right = displacement_field[y, x_right]

                if sign_transition > 0:
                    value_left = np.max(displacement_field[y, 0:x_right - 1])
                else:
                    value_left = np.min(displacement_field[y, 0:x_right - 1])

            else:
                x_left = x_initial
                max_transition_area = displacement_field.shape[1] - x_left
                value_left = displacement_field[y, x_left]

                if sign_transition > 0:
                    value_right = np.max(displacement_field[y, x_left + 1::])
                else:
                    value_right = np.min(displacement_field[y, x_left + 1::])

            if mode == 'linear':
                pass

            elif mode == 'gaussian':
                transition_profile = self._gaussian_transition(value_left,
                                                               value_right,
                                                               direction,
                                                               max_transition_area,
                                                               0.01)

            if len(transition_profile) == 0:
                continue

            if direction == 'left':
                displacement_field[y, x_initial-len(transition_profile) + 1:x_initial + 1] = transition_profile
            else:
                displacement_field[y, x_initial:x_initial+len(transition_profile)] = transition_profile

            print()

        return displacement_field


    def execute(self, image, label):

        [x_tl, y_tl], [x_tr, y_tr] = self._get_bone_edges(label)

        displacement_field = np.ones(label.shape) * self.max_displacement
        displacement_field[label == 1] = 0
        xc_bone = x_tl + (x_tr - x_tl)/2  # x coordinate of the bone center

        # everything below the bones is kept fixed (deformation = 0)
        displacement_field = self.add_displacement_below_bone(label, displacement_field,
                                                                   0, 0, displacement_decay_function='constant')
        displacement_field = self.add_displacement_above_bone(label, displacement_field, self.max_displacement, 0,
                                                                   displacement_decay_function='linear')

        displacement_field = self.max_displacement - displacement_field

        displacement_field = displacement_field*50

        displacement_field = self.apply_smoothed_horizontal_transition(displacement_field, x_tr-1,
                                                                       transition_region_gradient=0.1,
                                                                       direction='right',
                                                                       mode='gaussian')

        displacement_field = self.apply_smoothed_horizontal_transition(displacement_field, x_tl,
                                                                       transition_region_gradient=0.1,
                                                                       direction='left', mode='gaussian')

        plt.imshow(displacement_field)
        plt.show()

        # n = 20
        # displacement_field[y_tl-n:y_tl+n, x_tl-n:x_tl+n] = self.max_displacement
        # displacement_field[y_tr - n:y_tr + n, x_tr - n:x_tr + n] = self.max_displacement
        #
        # displacement_field = gaussian_filter(displacement_field, sigma=[1, 10])
        deformed_image = self._deform(image, displacement_field)

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(deformed_image, cmap='gray')
        plt.subplot(1, 3, 3)
        plt.imshow(displacement_field)
        plt.show()

        # n = 0
        # for x in range(x_tl - n, x_tr + n):
        #
        #     y_min = np.max(np.argwhere(label[:, x]>0))
        #
        #     hor_displacement = self.gaussian(x, xc_bone, 100)
        #
        #     # over_bone = np.linspace(0, ds, label.shape[0] - y_min)
        #     # #over_bone = np.flip(over_bone)
        #     # displacement_field[0:y_min, x] = 0
        #     # displacement_field[y_min::, x] = over_bone
        #
        #     over_bone_displacement = np.linspace(0, ds, label.shape[0])
        #     # over_bone = np.flip(over_bone)
        #     displacement_field[:, x] = over_bone_displacement

        # n = 40
        # for y in range(label.shape[0]):
        #
        #     lat_smothing = np.linspace(displacement_field[y, x_tl]*0.7, displacement_field[y, x_tl], n)
        #     gauss = self.gaussian(np.linspace(0, n, n), 0, 100)
        #     print(gauss[0])
        #     #lat_smothing = np.flip(lat_smothing)
        #     displacement_field[y, x_tl-n:x_tl] = lat_smothing


        # n = 100
        # for y in range(label.shape[0]):
        #
        #     lat_smothing = np.linspace(displacement_field[y, x_tr-1]*0.0, displacement_field[y, x_tr-1], n)
        #     lat_smothing = np.flip(lat_smothing)
        #     auss = self.gaussian(np.linspace(0, n, n), 0, 100)
        #     displacement_field[y, x_tr:x_tr+n] = lat_smothing
        #
        # n_gauss = len(displacement_field[300::, 0])
        # gauss_d = self.gaussian(np.linspace(0, n_gauss, n_gauss), 0, 15)
        #
        # for col in range(displacement_field.shape[1]):
        #     displacement_field[300::, col] = gauss_d*displacement_field[300::, col]
        #
        #
        # gauss = self.gaussian(np.linspace(0, label.shape[1], label.shape[1]), bone_center, 100)
        #
        #
        # for i in range(displacement_field.shape[0]):
        #     displacement_field[i, :] = np.multiply(displacement_field[i, :], gauss)
        #
        #
        # mapx_base, mapy_base = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        #
        # mapx = mapx_base
        # mapy = mapy_base + displacement_field*80
        #
        #
        # deformed_apple = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_CUBIC)


def command_iteration(method):
    if method.GetOptimizerIteration() == 0:
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


if __name__ == '__main__':

    root = "C:\\Users\\maria\\OneDrive\\Desktop\\us_augmentation"

    img = cv2.imread(os.path.join(root, "test.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)
    gt = cv2.imread(os.path.join(root, "test_label.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)

    img = cv2.imread(os.path.join(root, "5_243.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)
    gt = cv2.imread(os.path.join(root, "5_243_label.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)



    fixed_image = sitk.ReadImage(os.path.join(root, "image.mhd"), sitk.sitkFloat32)
    moving_image = sitk.ReadImage(os.path.join(root, "image.mhd"), sitk.sitkFloat32)

    method = Deformation()
    method.execute(img, gt)

