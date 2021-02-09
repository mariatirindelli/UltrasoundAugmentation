import python.us_augmentation as us
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk


class Deformation(us.BaseMethod):
    def __init__(self, blur_kernel=45, blur_sigma=20, apply_blur=True):
        super(Deformation, self).__init__()
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.apply_blur = apply_blur

    @staticmethod
    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    def execute(self, image, label):

        ds = 2

        yx_bones = np.argwhere(label == 1)
        tl_idx = np.argmin(yx_bones[:, 1])
        x_tl = yx_bones[tl_idx, 1]
        y_tl = yx_bones[tl_idx, 0]

        tr_idx = np.argmax(yx_bones[:, 1])
        x_tr = yx_bones[tr_idx, 1]
        y_tr = yx_bones[tr_idx, 0]

        displacement_field = - np.zeros(label.shape) * ds

        bone_center = x_tl + (x_tr - x_tl)/2
        n = 0
        for x in range(x_tl - n, x_tr + n):
            y_min = np.max(np.argwhere(label[:, x]>0))

            hor_disp = self.gaussian(x, bone_center, 100)

            # over_bone = np.linspace(0, ds, label.shape[0] - y_min)
            # #over_bone = np.flip(over_bone)
            # displacement_field[0:y_min, x] = 0
            # displacement_field[y_min::, x] = over_bone

            over_bone = np.linspace(0, ds, label.shape[0])
            # over_bone = np.flip(over_bone)
            displacement_field[:, x] = over_bone

        n = 40
        for y in range(label.shape[0]):

            lat_smothing = np.linspace(displacement_field[y, x_tl]*0.7, displacement_field[y, x_tl], n)
            gauss = self.gaussian(np.linspace(0, n, n), 0, 100)
            print(gauss[0])
            #lat_smothing = np.flip(lat_smothing)
            displacement_field[y, x_tl-n:x_tl] = lat_smothing

        n = 70
        for y in range(label.shape[0]):

            lat_smothing = np.linspace(displacement_field[y, x_tr-1]*0.0, displacement_field[y, x_tr-1], n)
            lat_smothing = np.flip(lat_smothing)
            auss = self.gaussian(np.linspace(0, n, n), 0, 100)
            displacement_field[y, x_tr:x_tr+n] = lat_smothing

        n_gauss = len(displacement_field[300::, 0])
        gauss_d = self.gaussian(np.linspace(0, n_gauss, n_gauss), 0, 10)

        for col in range(displacement_field.shape[1]):
            displacement_field[300::, col] = gauss_d*displacement_field[300::, col]


        gauss = self.gaussian(np.linspace(0, label.shape[1], label.shape[1]), bone_center, 100)


        for i in range(displacement_field.shape[0]):
            displacement_field[i, :] = np.multiply(displacement_field[i, :], gauss)


        mapx_base, mapy_base = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))

        mapx = mapx_base
        mapy = mapy_base + displacement_field*80


        deformed_apple = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_CUBIC)

        plt.subplot(1, 2, 1)
        plt.imshow(deformed_apple, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(image, cmap='gray')
        plt.show()

def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")

if __name__ == '__main__':

    root = "C:\\Users\\maria\\OneDrive\\Desktop\\us_augmentation"

    img = cv2.imread(os.path.join(root, "test.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)
    gt = cv2.imread(os.path.join(root, "test_label.png"), cv2.IMREAD_GRAYSCALE).astype(np.float)

    fixed_image = sitk.ReadImage(os.path.join(root, "image.mhd"), sitk.sitkFloat32)
    moving_image = sitk.ReadImage(os.path.join(root, "image.mhd"), sitk.sitkFloat32)

    method = Deformation()
    method.execute(img, gt)

