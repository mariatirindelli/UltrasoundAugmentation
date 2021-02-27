import us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os

class NoiseAugmentation(us.BaseMethod):
    def __init__(self, blur_kernel=45, blur_sigma=20, bone_signal=0.7, bg_signal=1.3, apply_blur=True, plot_result=False):
        super(NoiseAugmentation, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur
        self.plot_result = plot_result
        self.bone_signal = bone_signal
        self.bg_signal = bg_signal

    @staticmethod
    def local_energy(local_energy_contents):
        local_energy = local_energy_contents.get('local_energy')
        local_energy = cv2.normalize(local_energy, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return local_energy

    @staticmethod
    def local_phase(local_phase_contents):
        local_phase = local_phase_contents.get('LP')[:, :, 0, 0]
        local_phase = cv2.normalize(local_phase, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return local_phase

    @staticmethod
    def local_energy_noise(local_energy_contents, label, img, bone_signal, bg_signal):
        loc_energy = NoiseAugmentation.local_energy(local_energy_contents)
        loc_energy = cv2.normalize(loc_energy, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # label
        label = cv2.GaussianBlur(label.astype(np.float)*255, (25, 25), 25)
        label = label/np.max(label)

        norm_image_energy = img/(loc_energy + 0.000001)
        augm_image_energy = norm_image_energy.copy().astype(np.float)

        local_energy_bone = loc_energy*bone_signal*label
        local_energy_bg = loc_energy*bg_signal*(1 - label)

        loc_energy = local_energy_bone + local_energy_bg
        augm_image_energy = augm_image_energy*loc_energy

        return augm_image_energy

    @staticmethod
    def local_phase_noise(local_phase_contents, label, img, bone_signal, bg_signal):
        loc_phase = NoiseAugmentation.local_phase(local_phase_contents)
        loc_phase = cv2.normalize(loc_phase, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # label
        label = cv2.GaussianBlur(label.astype(np.float) * 255, (25, 25), 25)
        label = label / np.max(label)

        norm_image_phase = img / loc_phase
        augm_image_phase = norm_image_phase.copy().astype(np.float)

        local_phase_bone = loc_phase * bone_signal * label
        local_phase_bg = loc_phase * bg_signal * (1 - label)

        loc_phase = local_phase_bone + local_phase_bg
        augm_image_phase = augm_image_phase * loc_phase

        return augm_image_phase

    @staticmethod
    def _plot(image, label, augmented_image):
        plt.subplot(1, 3, 1)
        plt.imshow(label)
        plt.subplot(1, 3, 2)
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(augmented_image, cmap='gray')
        plt.show()

    def execute(self, image, label, local_energy_path, bone_signal, bg_signal):

        assert isinstance(image, np.ndarray)
        assert isinstance(label, np.ndarray)
        assert isinstance(local_energy_path, str)

        if label is None:
            return -1

        # 1. Blur the label
        if self.apply_blur:
            blur_label = self._blur(label, kernel=self.blur_kernel, sigma=self.blur_sigma)
        else:
            blur_label = label

        # 2. Get the local energy matrix from your folder
        local_energy_contents = scipy.io.loadmat(local_energy_path)

        # 3. Get augmented image
        augmented_image = NoiseAugmentation.local_energy_noise(local_energy_contents, label, image, bone_signal, bg_signal)

        if self.plot_result:
            self._plot(image, label, augmented_image)

        return augmented_image, label

if __name__ == '__main__':

    root = "C:\\Users\\chris\\Documents\\projects\\US_augmentation\\augmentation_data"

    img = cv2.imread(os.path.join(root, "test_US_2.bmp"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(root, "test_label_2.png"), cv2.IMREAD_GRAYSCALE)
    local_energy_path = "_"

    bone_signal = 0.7
    bg_signal = 1.3
    method = NoiseAugmentation(apply_blur=True, plot_result=True)
    method.execute(img, gt, local_energy_path, bone_signal, bg_signal)


