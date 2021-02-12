import cv2
import sys
#from scipy import io
import scipy.io
from us_augmentation import noise_augmentation

x = 12
path = r"C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_data\test_vertebrae_single\test_US_{}.bmp".format(
    x)

img = cv2.imread(path, 0)
if img is None:
    sys.exit("Could not read the image.")
local_energy_contents = scipy.io.loadmat \
    (r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\local_energy_US_{}.mat'.format(x))
local_phase_contents = scipy.io.loadmat \
    (r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\local_phase_US_{}.mat'.format(x))
feature_symmetry_contents = scipy.io.loadmat \
    (r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\feature_symmetry_US_{}.mat'.format(x))
#bone_probability, bone_probability_binary = bone_probability.bone_probability(img, local_energy_contents, local_phase_contents, feature_symmetry_contents)

label = cv2.imread(r"C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_data\test_vertebrae_single\test_label_{}.png".format(x), 0)

augment_img_local_energy = noise_augmentation.local_energy_noise(local_energy_contents, label, img, 0.7, 1.3)
cv2.imwrite(r'C:\Users\chris\Documents\projects\US_augmentation\augmentation_data\augment_image_energy.png', augment_img_local_energy)
augment_img_local_phase = noise_augmentation.local_phase_noise(local_phase_contents, label, img, 0.7, 1.3)
cv2.imwrite(r'C:\Users\chris\Documents\projects\US_augmentation\augmentation_data\augment_image_phase.png', augment_img_local_phase)

# plt.imshow(local_noise_1, 'gray')
# plt.show()
# plt.imshow(local_noise_2, 'gray')
# plt.show()