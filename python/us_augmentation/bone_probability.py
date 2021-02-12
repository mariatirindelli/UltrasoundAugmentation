import cv2
import sys
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy.misc
import scipy.stats as stats
from skimage.filters import gabor
from skimage import data, io
import itertools
import scipy.io
import scipy.ndimage as nd
from sklearn.preprocessing import normalize

# computation of a binary mask, input image should float
def binary_mask(img):
    # binary mask
    binary_mask = cv2.normalize(img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #for x in range(rows):
    #    if x < 10:
    #        binary_mask[binary_mask>0.125]=0.0
    binary_mask[binary_mask>0.0]=1.0

    return binary_mask

# computation of the Laplacian of a Gaussian (LoG) of an image
def lapl_gaus(img):
    # Laplacian of a Gaussian filter
    lapl_img = nd.gaussian_laplace(img,2)
    lapl_img = cv2.normalize(lapl_img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return lapl_img

# defines a shadow map of an image
def shadow_map(img):
    shadow = np.zeros([503,272])
    rows, cols = img.shape

    def gaussian(x, mu, sig):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    for y in range(cols):
        for x in range(rows):
            gaussian_weight = gaussian(np.arange(0, rows - x)/rows, 0, 1)
            nom = np.sum(gaussian_weight * img[0: rows-x, y])
            den = np.sum(gaussian_weight)
            shadow[x,y] = nom / den
    shadow = cv2.normalize(shadow, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return shadow

def local_energy(local_energy_contents):
    local_energy = local_energy_contents.get('LE')[:,:,0,0]
    local_energy = cv2.normalize(local_energy, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return  local_energy

def local_phase(local_phase_contents):
    local_phase = local_phase_contents.get('LP')[:,:,0,0]
    local_phase = cv2.normalize(local_phase, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return local_phase

def feature_symmetry(feature_symmetry_contents):
    feature_symmetry = feature_symmetry_contents.get('FS')
    feature_symmetry = cv2.normalize(feature_symmetry, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return  feature_symmetry

# integrated backscatter energy
def Ibs(img):
    rows, cols = img.shape
    ibs = np.zeros([503,272])
    for x in range (cols):
        for y in range (rows):
            ibs[y,x] = np.sum(pow(img[0:y,x], 2))
    ibs = cv2.normalize(ibs, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return ibs

# compute bone probability map
def bone_probability(img, local_energy_contents, local_phase_contents, feature_symmetry_contents):
    # define values
    rows, cols = img.shape

    # normalize the image
    float_img = img.astype(np.float)
    norm_img = cv2.normalize(float_img, None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # smoothing with Gaussian kernel
    smoothed_img = cv2.GaussianBlur(norm_img,(7,7),6)

    lapl_img = lapl_gaus(img)
    binary = binary_mask(img)
    shadow = shadow_map(img)
    loc_energy = local_energy(local_energy_contents)
    loc_phase = local_phase(local_phase_contents)
    feat_symm = feature_symmetry(feature_symmetry_contents)
    ibs = Ibs(img)

    #bone probability map
    bone_probability = cv2.normalize((lapl_img * binary * shadow * ibs * loc_energy * loc_phase * feat_symm), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    #bone probability binary
    bone_probability_binary = np.zeros([503,272])
    for x in range(cols):
        for y in range(rows):
            if bone_probability[y,x] > 0.1:
                bone_probability_binary[y,x] = 1.0
            else:
                bone_probability_binary[y,x] = 0.0

    # show images
    # f, axarr = plt.subplots(3,4)
    # axarr[0,0].imshow(norm_img, 'gray')
    # axarr[0,0].set_title('Normalized')
    # axarr[0,1].imshow(smoothed_img, 'gray')
    # axarr[0,1].set_title('Smoothed')
    # axarr[0,2].imshow(binary_mask, 'gray')
    # axarr[0,2].set_title('Mask')
    # axarr[0,3].imshow(lapl_img, 'gray')
    # axarr[0,3].set_title('Laplacian')
    # axarr[1,0].imshow(shadow, 'gray')
    # axarr[1,0].set_title('Shadow')
    # axarr[1,1].imshow(local_energy, 'gray')
    # axarr[1,1].set_title('Local Energy')
    # axarr[1,2].imshow(local_phase, 'gray')
    # axarr[1,2].set_title('Local Phase')
    # axarr[1,3].imshow(feature_symmetry, 'gray')
    # axarr[1,3].set_title('Feature Symmetry')
    # axarr[2,0].imshow(ibs, 'gray')
    # axarr[2,0].set_title('Integrated Backscattering')
    # axarr[2,1].imshow(bone_probability, 'gray')
    # axarr[2,1].set_title('Bone Probability')
    # axarr[2,2].imshow(bone_probability_binary, 'gray')
    # axarr[2,2].set_title('Bone Probability Binary')
    # plt.show()

    return bone_probability, bone_probability_binary



