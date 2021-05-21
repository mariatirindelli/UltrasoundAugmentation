import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import math
import argparse
from pre_post_processing_scripts.fix_pythonpath import *
import imfusion
imfusion.init()

def rotate_image(image, center, angle):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result

def is_rotated(shared_image):
    r = shared_image.matrix[0:3, 0:3]

    # (np.sum(r - np.eye(3)))
    if np.abs(np.sum(r - np.eye(3))) < 1e-16:
        return False

    return True

def preprocess_volumes(image_set):

    # TODO: if identity only spacing

    properties_resampling = imfusion.Properties({'targetSpacing': [1, 1, 1],
                                                'resamplingMode': 2,
                                                 'interpolationMode': 1})

    result_set = imfusion.executeAlgorithm('Image Resampling', [image_set], properties_resampling)

    if is_rotated(result_set[0][0]):
        properties_transformation = imfusion.Properties({'nearestInterpolation': 0,
                                                         'adjustSize': 1,
                                                         'samplerWrapping': 0})

        result_set = imfusion.executeAlgorithm('Apply Transformation', result_set, properties_transformation)[0]
    return np.squeeze(np.array(result_set))


def is_roi_valid(image_size, rotation_center, rotation_angle, roi):
    """
    roi = [x_in, x_end, y_in, y_end]
    """
    [x_in, x_end, y_in, y_end] = roi
    rotated_mask = rotate_image(np.ones(image_size) * 255, rotation_center, rotation_angle)
    roi_mask = rotated_mask[y_in:y_end, x_in:x_end]

    if roi_mask.size - np.count_nonzero(roi_mask) > 30:
        return False

    return True

def reshaping(image, image_spacing=(1, 1), target_spacing=(1, 1)):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * image_spacing[0] / target_spacing[0])
    new_height = int(height * image_spacing[1] / target_spacing[1])

    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

def random_crop_from_slice(im, label, skin_label, crop_size, image_spacing=(1, 1), target_spacing=(1, 1)):

    # 1 Rescale the images to get to the target spacing
    im = reshaping(im, image_spacing=image_spacing, target_spacing=target_spacing)
    label = reshaping(label, image_spacing=image_spacing, target_spacing=target_spacing)
    skin_label = reshaping(skin_label, image_spacing=image_spacing, target_spacing=target_spacing)

    # 2. Extract skin points
    yx_skin = np.argwhere(skin_label > 0)  # shape  [n_points, 2]

    # 3. Randomly select center
    center_idx = random.randint(0, yx_skin.shape[0] - 1)
    y_c, x_c = yx_skin[center_idx, 0], yx_skin[center_idx, 1]

    # 4. Compute the ROI indexes
    roi_width, roi_height = crop_size[0], crop_size[1]
    y_in = max(y_c - int(roi_height/2), 0)
    y_end = y_in + int(roi_height)
    x_roi_end = x_c + 2
    x_roi_in = int(max(0, x_roi_end - roi_width))

    # 5. Fit a line on the skin points in the roi
    skin_label_roi = skin_label[y_in:y_end, :]
    contours, _ = cv2.findContours(skin_label_roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    [vx, vy, _, _] = cv2.fitLine(contours[0], cv2.DIST_L1, 0, 0.01, 0.01)
    # vx,vy are normalized vector collinear to the line and x0,y0 is a point on the line

    # 6. Compute the rotation angle between the line and the vertical direction of the image and rotate the image
    # suc that the skin in the ROI is (almost) parallel to the image vertical side

    # print(vx, " ", vy)

    if vy != 0:
        alpha = -np.degrees(math.atan(vx/vy))
        center_tuple = (x_c, y_c)
        if not is_roi_valid(label.shape, center_tuple, alpha, [x_roi_in, x_roi_end, y_in, y_end]):
            return None, None

        rotated_image = rotate_image(im, center_tuple, alpha)
        rotated_label = rotate_image(label, center_tuple, alpha)
        image_roi = rotated_image[y_in:y_end, x_roi_in:x_roi_end]
        label_roi = rotated_label[y_in:y_end, x_roi_in:x_roi_end]

    else:
        image_roi = im[y_in:y_end, x_roi_in:x_roi_end]
        label_roi = label[y_in:y_end, x_roi_in:x_roi_end]

    if np.sum(label_roi) == 0:
        return None, None

    return image_roi, label_roi

def rescale_intensity(image):
    rescaled_image = image - np.min(image)
    rescaled_image = rescaled_image / np.max(rescaled_image) * 255

    return rescaled_image.astype(np.uint8)

def process_ct_volume(ct, label, skin, roi_size, image_spacing, target_spacing, images_per_slice=10, save_path='',
                      subject_id='', data_mask = None):

    image_id = 0
    for i in range(ct.shape[-1]):

        # extract slice
        ct_slice = np.squeeze(ct[..., i])
        label_slice = np.squeeze(label[..., i])
        skin_slice = np.squeeze(skin[..., i])

        # flip on the vertical dimension
        ct_slice = np.flip(ct_slice, axis=0)
        label_slice = np.flip(label_slice, axis=0)
        skin_slice = np.flip(skin_slice, axis=0)

        if np.sum(label_slice) == 0 or np.sum(skin_slice) == 0:
            continue

        for _ in range(images_per_slice):
            image_roi, label_roi = random_crop_from_slice(im=ct_slice,
                                                          label=label_slice,
                                                          skin_label=skin_slice,
                                                          crop_size=roi_size,
                                                          image_spacing=image_spacing,
                                                          target_spacing=target_spacing)

            if image_roi is None or label_roi is None:
                continue

            if image_roi.shape != (544, 516) or label_roi.shape != (544, 516):
                print("size error")
                continue

            image_id += 1

            image_path = os.path.join(save_path, subject_id + "_" + str(image_id) + '.png')
            label_path = os.path.join(save_path, subject_id + "_" + str(image_id) + '_label.png')

            image_roi = rescale_intensity( np.rot90(image_roi) )
            label_roi = rescale_intensity( np.rot90(label_roi) )

            if data_mask is not None:
                image_roi[data_mask==0] = 0
                label_roi[data_mask==0] = 0

            cv2.imwrite(image_path, image_roi)
            cv2.imwrite(label_path, label_roi)


def main(params):
    if os.path.exists(args.data_mask):
        data_mask = cv2.imread(args.data_mask, 0)
    else:
        data_mask = None

    data_list = [item for item in os.listdir(params.root) if ".imf" in item and
                 "label" not in item and "skin" not in item]

    for data_path in data_list:

        subject_id = data_path.split(".")[0]
        sweep_path = os.path.join(params.root, data_path)
        label_path = os.path.join(params.root, data_path.replace(".", "-labels."))
        skin_path = os.path.join(params.root, data_path.replace(".", "-skin."))

        if not os.path.exists(sweep_path) or not os.path.exists(label_path) or not os.path.exists(skin_path):
            print("One of the needed files was not found: ")
            print("{} : {}".format(sweep_path, "True" if os.path.exists(sweep_path) else "False"))
            print("{} : {}".format(label_path, "True" if os.path.exists(label_path) else "False"))
            print("{} : {}".format(skin_path, "True" if os.path.exists(skin_path) else "False"))
            continue

        print(sweep_path)
        ct = imfusion.open(sweep_path)[0]
        label = imfusion.open(label_path)[0]
        skin = imfusion.open(skin_path)[0]

        ct_array = preprocess_volumes(ct)
        label_array = preprocess_volumes(label)
        skin_array = preprocess_volumes(skin)

        process_ct_volume(ct=ct_array,
                          label=label_array.astype(np.uint8),
                          skin=skin_array.astype(np.uint8),
                          roi_size=params.target_size,
                          image_spacing=(ct[0].spacing[0], ct[0].spacing[1]),
                          target_spacing=params.target_spacing,
                          images_per_slice=params.images_per_slice,
                          save_path=params.save_path,
                          subject_id=subject_id,
                          data_mask=data_mask)

def str2tuple(input_string):
    split_string = input_string.replace(" ", "")
    split_string = split_string.split(",")
    return tuple([float(item) for item in split_string])


if __name__ == '__main__':

    """
    example usage: 
    name.py --root=D:\\NAS\\exported_verse_db_6.4.21 
            --save_path=D:\\NAS\\CT_label_db
            --target_size=516, 544
            --target_spacing=0.145228, 0.145228
            --data_mask=D:\\Maria\\DataBases\\SpineIFL\\curvilinear_mask_7cm.png
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--root', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--target_size', type=str2tuple)
    parser.add_argument('--target_spacing', type=str2tuple)
    parser.add_argument('--images_per_slice', type=int, default=10)
    parser.add_argument('--data_mask', type=str, default="",
                        help='The first sub_id. e.g. if start_sub_id=5, all sub_id will be higher than 5')

    args = parser.parse_args()

    main(args)