import cv2
import os
import us_augmentation as us
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import torchvision.transforms.functional as TF
from PIL import Image
import torchvision.transforms as transforms

data_root = "D:\\Chrissi\\spine\\NAS\\header_image"
image_path = "C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\8_127.png"
label_path = "C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\8_127_label.png"
local_energy_path = "C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\8_127_local_energy.mat"
output_path = "D:\\Chrissi\\spine\\NAS\\header_images"


image = io.imread(image_path)
label = io.imread(label_path)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\5_058_label.png', label)

# #label = cv2.dilate(label*255, kernel=np.ones((5, 5), np.uint8))
# pil_image = Image.fromarray(image)
#
# # vertical flip
# image_vflip = pil_image.copy()
# image_vflip = TF.vflip(pil_image)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_vflip.png', image_vflip, cmap='gray')
#
# # horizontal flip
# image_hflip = pil_image.copy()
# image_hflip = TF.hflip(pil_image)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_hflip.png', image_hflip, cmap='gray')
#
# # rotation
# image_rotation = pil_image.copy()
# image_rotation = TF.rotate(image_rotation, 45)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_rotation.png', image_rotation, cmap='gray')
#
# # translation
# image_translation = pil_image.copy()
# image_translation = TF.affine(image_translation, translate=[60,60], angle=0, shear=1, scale=1)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_translation.png', image_translation, cmap='gray')
#
# # shearing
# image_shear = pil_image.copy()
# image_shear = TF.affine(image_shear, translate=[0,0], angle=0, shear=[10, 10], scale=1)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_shear.png', image_shear, cmap='gray')
#
# # scaling
# image_scale = pil_image.copy()
# image_scale = TF.affine(image_scale, translate=[0,0], angle=0, shear=0, scale=1.5)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_scale.png', image_scale, cmap='gray')
#
# # brightness
# image_brightness = pil_image.copy()
# image_brightness = TF.adjust_brightness(image_brightness, 1.7)
# plt.imsave('D:\\Chrissi\\spine\\NAS\\header_images\\11_168_brightness.png', image_brightness, cmap='gray')

# # deformation
# image_deformation = image.copy()
# image_deformation, label_def = us.Deformation().execute(image_deformation, label, 80)
# plt.imsave('C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\5_058_deformation.png', image_deformation, cmap='gray')
#
# # reverberation
# image_reverberation = image.copy()
# image_reverberation, label_rev = us.MultipleReflections().execute(image_reverberation, label, 0.9)
# plt.imsave('C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\5_058_reverberation.png', image_reverberation, cmap='gray')

# noise
image_noise = image.copy()
image_noise, label_noise = us.NoiseAugmentation().execute(image_noise, label, local_energy_path, 0.5, 1.2)
plt.imsave('C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\8_127_noise.png', image_noise, cmap='gray')


# # all
# image_all = image.copy()
# image_all, label_all = us.NoiseAugmentation().execute(image_noise, label, local_energy_path, 0.6, 1.4)
# image_all, label_all = us.Deformation().execute(image_all, label, 100)
# image_all, label_all = us.MultipleReflections().execute(image_all, label_all, 1.0)
# plt.imsave('C:\\Users\\chris\\Documents\\projects\\US_augmentation\\MICCAI21\\figure_images\\5_058_all.png', image_all, cmap='gray')