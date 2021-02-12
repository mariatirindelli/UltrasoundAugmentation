import python.us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


class MultipleReflections(us.BaseMethod):
    def __init__(self, blur_kernel=45, blur_sigma=20, apply_blur=True, reflection_intensity=0.65, plot_result=False):
        super(MultipleReflections, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur
        self.reflection_intensity = reflection_intensity
        self.plot_result = plot_result

    @staticmethod
    def _find_centroid(labelmap):
        yx_bones = np.argwhere(labelmap > 0)
        y_c = np.mean(yx_bones[:, 0])
        x_c = np.mean(yx_bones[:, 1])

        return x_c, y_c

    @staticmethod
    def translate_image_patch(src_image, src_mask, x_translation, y_translation):

        yx_mask = np.argwhere(src_mask > 0)
        yx_translated = yx_mask
        yx_translated[:, 0] += int(y_translation)
        yx_translated[:, 1] += int(x_translation)

        translated_mask = np.zeros(src_mask.shape)
        translated_mask[yx_translated[:, 0], yx_translated[:, 1]] = src_mask[src_mask > 0]

        translated_image_patch = np.zeros(src_image.shape)
        translated_image_patch[yx_translated[:, 0], yx_translated[:, 1]] = src_image[src_mask > 0]

        return translated_mask, translated_image_patch

    @staticmethod
    def _plot(image, label, augmented_image):
        plt.subplot(1, 3, 1)
        plt.imshow(label)
        plt.subplot(1, 3, 2)
        plt.imshow(image, cmap='gray')
        plt.subplot(1, 3, 2)
        plt.imshow(augmented_image, cmap='gray')
        plt.show()

    def execute(self, image, label=None):

        if label is None:
            return -1

        # 1. Blur the label
        if self.apply_blur:
            blur_label = self._blur(label, kernel=self.blur_kernel, sigma=self.blur_sigma)
        else:
            blur_label = label

        # 2. Find the y position of the bone
        _, y_c = self._find_centroid(label)

        # 3. Find coordinate of the 1st reflection, at position 2*d where d is the distance of the bone from the
        # transducer, i.e. yc
        y_reflection = y_c * 2

        # 4. Getting the translated blurred mask and the masked, translated image patched
        blur_reflection_mask, reflection_image_patch = \
            self.translate_image_patch(image, blur_label, 0, int(y_reflection - y_c))

        # 5. Getting the non blurred translated label
        reflection_mask, _ = self.translate_image_patch(image, label, 0, int(y_reflection - y_c))

        augmented_image = blur_reflection_mask * reflection_image_patch * self.reflection_intensity + \
                          (1 - blur_reflection_mask * self.reflection_intensity) * image * 2

        if self.plot_result:
            self._plot(image, label, augmented_image)

        return augmented_image


if __name__ == '__main__':

    root = "C:\\Users\\maria\\OneDrive\\Desktop\\us_augmentation"

    img = cv2.imread(os.path.join(root, "test.png"), cv2.IMREAD_GRAYSCALE)
    gt = cv2.imread(os.path.join(root, "test_label.png"), cv2.IMREAD_GRAYSCALE)

    method = MultipleReflections(apply_blur=True, plot_result=True)
    method.execute(img, gt)