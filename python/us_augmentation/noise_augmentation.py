import cv2
import numpy as np
import matplotlib.pyplot as plt
#from scipy import io
from us_augmentation import bone_probability


def local_energy_noise(local_energy_contents, label, img, bone_signal, bg_signal):
    loc_energy = bone_probability.local_energy(local_energy_contents)
    loc_energy = cv2.normalize(loc_energy, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # label
    label = cv2.GaussianBlur(label.astype(np.float)*255, (25, 25), 25)
    label = label/np.max(label)

    norm_image_energy = img/loc_energy
    augm_image_energy = norm_image_energy.copy().astype(np.float)

    local_energy_bone = loc_energy*bone_signal*label
    local_energy_bg = loc_energy*bg_signal*(1 - label)

    loc_energy = local_energy_bone + local_energy_bg
    augm_image_energy = augm_image_energy*loc_energy

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('real image')
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.title('label + gaussian blur')
    plt.subplot(1, 3, 3)
    plt.imshow(augm_image_energy*255, cmap='gray')
    plt.title('augmented image')
    plt.show()

    return  augm_image_energy

def local_phase_noise(local_phase_contents, label, img, bone_signal, bg_signal):
    loc_phase = bone_probability.local_phase(local_phase_contents)
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

    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='gray')
    # plt.title('real image')
    # plt.subplot(1, 3, 2)
    # plt.imshow(label)
    # plt.title('label + gaussian blur')
    # plt.subplot(1, 3, 3)
    # plt.imshow(augm_image_phase*255, cmap='gray')
    # plt.title('augmented image')
    # plt.show()

    return  augm_image_phase


# # read images
# x = 1
# while x <= 12:
#     path = r"C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_data\test_vertebrae_single\test_US_{}.bmp".format(x)
#
#     img = cv2.imread(path, 0)
#     if img is None:
#         sys.exit("Could not read the image.")
#
#     # load monogenic signal data
#     local_energy_contents = scipy.io.loadmat(r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\local_energy_US_{}.mat'.format(x))
#     local_phase_contents = scipy.io.loadmat(r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\local_phase_US_{}.mat'.format(x))
#     feature_symmetry_contents = scipy.io.loadmat(r'C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_result\feature_symmetry_US_{}.mat'.format(x))
#     bone_probability_binary = bone_probability.bone_probability(img, local_energy_contents, local_phase_contents, feature_symmetry_contents)
#
#     # load label
#     label = cv2.imread(r"C:\Users\chris\Documents\UltrasoundAugmentation\matlab\test_data\test_vertebrae_single\test_label_{}.png".format(x), 0)
#
#     # compute local energy
#     local_energy = local_energy_contents.get('LE')[:, :, 0, 0]
#     local_energy = cv2.normalize(local_energy, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#     probability_label = cv2.GaussianBlur(bone_probability_binary.astype(np.float)*255, (15,15),5)
#     probability_label = probability_label/np.max(probability_label)
#
#     # local phase noise
#
#
#
#     #local_energy_bone = local_energy*0.7*probability_label
#     #local_energy_bg = local_energy*1.3*(1 - probability_label)
#
#
#
#
#
#     # compute local phase
#     local_phase = local_phase_contents.get('LP')[:, :, 0, 0]
#     local_phase = cv2.normalize(local_phase, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#     # local phase noise
#     norm_image_phase = img/local_phase
#     augm_image_phase = norm_image_phase.copy().astype(np.float)
#
#     local_phase_bone = local_phase*1.2*label
#     local_phase_bg = local_phase*1.6*(1 - label)
#
#     local_phase = local_phase_bone + local_phase_bg
#     augm_image_phase = augm_image_phase*local_phase
#
#     # plt.subplot(1, 3, 1)
#     # plt.imshow(img, cmap='gray')
#     # plt.title('real image')
#     # plt.subplot(1, 3, 2)
#     # plt.imshow(label)
#     # plt.title('label + gaussian blur')
#     # plt.subplot(1, 3, 3)
#     # plt.imshow(augm_image_phase, cmap='gray')
#     # plt.title('augmented image')
#     # plt.show()
#
#     # compute feature symmetry
#     feature_symmetry = feature_symmetry_contents.get('FS')
#     feature_symmetry = cv2.normalize(local_phase, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#
#     # local phase noise
#     norm_image_feature_symmetry = img/feature_symmetry
#     augm_image_feature_symmetry = norm_image_feature_symmetry.copy().astype(np.float)
#
#     feature_symmetry_bone = feature_symmetry*1.5*label
#     feature_symmetry_bg = feature_symmetry*0.5*(1 - label)
#
#     feature_symmetry = feature_symmetry_bone + feature_symmetry_bg
#     augm_image_feature_symmetry = augm_image_feature_symmetry*feature_symmetry
#
#     # plt.subplot(1, 3, 1)
#     # plt.imshow(img, cmap='gray')
#     # plt.title('real image')
#     # plt.subplot(1, 3, 2)
#     # plt.imshow(label)
#     # plt.title('label + gaussian blur')
#     # plt.subplot(1, 3, 3)
#     # plt.imshow(augm_image_phase, cmap='gray')
#     # plt.title('augmented image')
#     # plt.show()
#
#     x = x + 1

