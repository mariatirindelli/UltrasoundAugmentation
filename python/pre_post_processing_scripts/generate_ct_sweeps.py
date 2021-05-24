import plotly.graph_objs as go
import numpy as np
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R
import os
from random import sample
import argparse

def fit_plane(points):
    assert points.shape[1] == 3
    centroid = points.mean(axis=0)
    x = points - centroid[None, :]
    U, S, Vt = np.linalg.svd(x.T @ x)
    normal = U[:, -1]
    origin_distance = normal @ centroid
    rmse = np.sqrt(S[-1] / len(points))
    return normal, -origin_distance, centroid, rmse


def resample(image, transform):
    """
    This function resamples (updates) an image using a specified transform
    :param image: The sitk image we are trying to transform
    :param transform: An sitk transform (ex. resizing, rotation, etc.
    :return: The transformed sitk image
    """
    reference_image = image
    interpolator = sitk.sitkBSpline
    default_value = 0
    return sitk.Resample(image, reference_image, transform, interpolator, default_value)

def rotation3d(image, theta_x, theta_y, theta_z, center_x, center_y, center_z):
    """
    This function rotates an image across each of the x, y, z axes by theta_x, theta_y, and theta_z degrees
    respectively
    :param image: An sitk MRI image
    :param theta_x: The amount of degrees the user wants the image rotated around the x axis
    :param theta_y: The amount of degrees the user wants the image rotated around the y axis
    :param theta_z: The amount of degrees the user wants the image rotated around the z axis
    :param show: Boolean, whether or not the user wants to see the result of the rotation
    :return: The rotated image
    """
    theta_x = np.deg2rad(theta_x)
    theta_y = np.deg2rad(theta_y)
    theta_z = np.deg2rad(theta_z)

    fixed_center = (center_x, center_y, center_z)
    euler_transform = sitk.Euler3DTransform(fixed_center, theta_x, theta_y, theta_z, (0, 0, 0))

    euler_transform.SetCenter(fixed_center)
    euler_transform.SetRotation(theta_x, theta_y, theta_z)
    resampled_image = resample(image, euler_transform)

    return resampled_image


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def get_rotation(skin_mask):
    """
    The results are give in SimpleItk coordinates:
    Example
     - 0 = SI direction = z
     - 1 = LL direction = x
     - 2 = AP direction = y

    """

    point_cloud = np.argwhere(skin_mask > 0)

    # 1. Fitting the point with a plane
    normal, distance, centroid, rmse = fit_plane(point_cloud)

    # 2. Always the normal directed as the y axis
    if normal[1] < 0:
        normal = -normal

    # 3. Compute the rotation angles between the plane normal and the y axis
    rot_mat = rotation_matrix_from_vectors(normal, np.array([0.0, 1.0, 0.0]))
    rot_mat = R.from_matrix(rot_mat)
    thetas = rot_mat.as_euler('zyx', degrees=True)
    thetas = [int(thetas[0]), int(thetas[1]), int(thetas[2])]

    # 4. Sort x, y, z to fit the order expected by SimpleItk
    center_x, center_y, center_z = centroid[1], centroid[2], centroid[0]
    theta_z, theta_y, theta_x = thetas[1], thetas[2], thetas[0]

    return [theta_x, theta_y, theta_z], [center_x, center_y, center_z]

def get_crop_roi(label_mask, centroid, us_bmode_size=(512, 512)):
    in_0 = centroid[0] - us_bmode_size[0] // 2
    end_0 = in_0 + us_bmode_size[0]

    end_1 = centroid[1] - 5
    in_1 = end_1 - us_bmode_size[1]

    cropped_label = label_mask[in_0:end_0, in_1:end_1, :]

    reduced_label = np.sum(np.sum(cropped_label, axis=0, keepdims=True), axis=1)
    reduced_label = np.squeeze(reduced_label)
    in_2 = np.argwhere(reduced_label != 0)[0][0]
    end_2 = np.argwhere(reduced_label != 0)[-1][0]

    return [in_0, end_0], [in_1, end_1], [in_2, end_2]


def resample_2_us(sitk_image, us_spacing):
    """
    The us_spacing is in python axis coordinate
    Example
     - 0 = SI direction = z
     - 1 = LL direction = x
     - 2 = AP direction = y

    simpleItk expects the spacing to be passed as (z, x, y) = (us_spacing[2], us_spacing[0], us_spacing[1])
    """

    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator = sitk.sitkBSpline

    new_spacing = (us_spacing[2], us_spacing[0], us_spacing[1])
    resample_filter.SetOutputSpacing(new_spacing)

    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)
    orig_spacing = sitk_image.GetSpacing()

    new_size = [np.ceil(orig_size[i] * (orig_spacing[i] / new_spacing[i])) for i in range(len(orig_size))]
    new_size = [int(s) for s in new_size]
    resample_filter.SetSize(new_size)

    resampled_image = resample_filter.Execute(sitk_image)

    return resampled_image


def process_volume(vol_path, label_path, skin_mask_path, random_centroid, us_bmode_size=(512, 512),
                   us_spacing=(0.14, 0.14, 1), roi_tolerance=10):
    """
    The random centroid is given in python coordinate - axis = [0, 1, 2]

    """
    sitk_vol = sitk.ReadImage(vol_path)
    sitk_label = sitk.ReadImage(label_path)

    sitk_skin_mask = sitk.ReadImage(skin_mask_path)
    skin_mask = sitk.GetArrayFromImage(sitk_skin_mask)

    # 1. Keeping the skin mask only around the centroid as we want the "local plane"
    img_spacing = sitk_skin_mask.GetSpacing()
    roi_height = int(us_bmode_size[0] * us_spacing[0] / img_spacing[0])

    in_0 = random_centroid[0] - roi_height // 2 - roi_tolerance
    end_0 = random_centroid[0] + roi_height // 2 + roi_tolerance

    if in_0 < 0:
        in_0 = 0
        end_0 = in_0 + roi_height + 2*roi_tolerance

    if end_0 >= skin_mask.shape[0]:
        end_0 = skin_mask.shape[0] - 1
        in_0 = end_0 - roi_height - 2*roi_tolerance

    skin_mask[0:in_0] = 0
    skin_mask[end_0::] = 0

    # 2. Computing the rotation (in SimpleItk coordinates)
    [theta_x, theta_y, theta_z], [center_x, center_y, center_z] = get_rotation(skin_mask=skin_mask)

    # 3. rotate the volume and labels
    rotated_volume = rotation3d(sitk_vol, theta_x, theta_y, theta_z, center_x, center_y, center_z)
    resampled_volume = resample_2_us(rotated_volume, us_spacing)
    rotated_volume_array = sitk.GetArrayFromImage(resampled_volume)

    rotated_label = rotation3d(sitk_label, theta_x, theta_y, theta_z, center_x, center_y, center_z)
    resampled_label = resample_2_us(rotated_label, us_spacing)
    rotated_label_array = sitk.GetArrayFromImage(resampled_label)

    spacing = resampled_label.GetSpacing()

    # 3. Crop the volume and labels to generate the sweeps

    # In simple itk coordinates
    # (us_spacing_z, us_spacing_x, us_spacing_y) = (us_spacing[0], us_spacing[1], us_spacing[2])
    center_1, center_2, center_0 = int(center_x * img_spacing[1] / us_spacing[1]), \
                                   int(center_y * img_spacing[2] / us_spacing[2]), \
                                   int(center_z * img_spacing[0] / us_spacing[0])

    [in_0, end_0], [in_1, end_1], [in_2, end_2] = get_crop_roi(label_mask=rotated_label_array,
                                                               centroid=[center_0, center_1, center_2],
                                                               us_bmode_size=us_bmode_size)

    cropped_volume = rotated_volume_array[in_0:end_0, in_1:end_1, in_2:end_2]
    cropped_label = rotated_label_array[in_0:end_0, in_1:end_1, in_2:end_2]

    return cropped_volume, cropped_label, spacing

def get_random_centroid(skin_mask):

    pts_0 = np.argwhere(skin_mask > 0)[:, 0]

    centroid_0 = sample(pts_0.tolist(), 1)[0]

    if centroid_0 - 5 < 0:
        centroid_0 = centroid_0 + 5

    local_roi = skin_mask.copy()
    local_roi[0:centroid_0 - 5, :, :] = 0
    local_roi[centroid_0 + 5 ::, :, :] = 0

    centroid_1 = np.mean(np.argwhere(local_roi > 0)[:, 1])
    centroid_2 = np.mean(np.argwhere(local_roi > 0)[:, 2])

    return [int(centroid_0), int(centroid_1), int(centroid_2)]

def main(params):

    volume_list = os.listdir(params.data_root)
    volume_list = [item for item in volume_list if "label" not in item and "skin" not in item]

    for volume_name in volume_list:
        subject_id = volume_name.replace(".imf", "")
        for i in range(params.iter_x_volume):
            sweep_id = i

            skin_mask_name = volume_name.replace(".mhd", "-skin.imf")
            label_name = volume_name.replace(".mhd", "-labels.imf")

            sitk_skin_mask = sitk.ReadImage(os.path.join(params.data_root, skin_mask_name))
            skin_mask = sitk.GetArrayFromImage(sitk_skin_mask)

            random_centroid = get_random_centroid(skin_mask)

            cropped_volume, cropped_label, spacing = process_volume(
                vol_path="test_file\\volume.mhd",
                label_path=os.path.join(params.data_root, label_name),
                skin_mask_path=os.path.join(params.data_root, skin_mask_name),
                random_centroid=random_centroid,
                us_bmode_size=(80, 80))

            cropped_volume_itk = sitk.GetImageFromArray(cropped_volume)
            cropped_label_itk = sitk.GetImageFromArray(cropped_label)

            cropped_volume_itk.SetSpacing(spacing)
            cropped_label_itk.SetSpacing(spacing)

            sitk.WriteImage(cropped_volume_itk, os.path.join(params.save_dir, subject_id + "_" + sweep_id + ".mhd"))
            sitk.WriteImage(cropped_label_itk, os.path.join(params.save_dir, subject_id + "_" + sweep_id + "_label.mhd"))


    # sitk_skin_mask = sitk.ReadImage("test_file\\skin_label.mhd")
    # skin_mask = sitk.GetArrayFromImage(sitk_skin_mask)
    #
    # random_centroid = get_random_centroid(skin_mask)
    # cropped_volume, cropped_label = process_volume(vol_path="test_file\\volume.mhd",
    #                                                label_path="test_file\\label.mhd",
    #                                                skin_mask_path="test_file\\skin_label.mhd",
    #                                                random_centroid=random_centroid,
    #                                                us_bmode_size=(512, 512))
    #
    # sitk.WriteImage(sitk.GetImageFromArray(cropped_volume), "cropped_volume.mhd")
    # sitk.WriteImage(sitk.GetImageFromArray(cropped_label), "cropped_label.mhd")

    #sitk_image = sitk.ReadImage("tmp_real.mhd")
    #
    # point_cloud2 = get_scatter_plot(point_cloud)
    #
    # vector_system = get_axis_plot(centroid, centroid+normal*50)
    # x_axis = get_axis_plot([0, 0, 0], [rotated_array.shape[0], 0, 0])
    # y_axis = get_axis_plot([0, 0, 0], [0, rotated_array.shape[1], 0])
    # z_axis = get_axis_plot([0, 0, 0], [0, 0, rotated_array.shape[2]])
    #
    # # data2 = [vold_data_rot, vector_system]
    # data2 = [point_cloud1, point_cloud2, vector_system, x_axis, y_axis, z_axis]
    # fig2 = go.Figure(data=data2)


if __name__ == '__main__':

    """
    example usage: 
    name.py --data_root=E:\\NAS\\exported_verse_db_6.4.21
            --save_dir=E:\\NAS\\data\\sweep_wise\\CT_sweep
            --iter_x_volume=10
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--iter_x_volume', type=int)
    args = parser.parse_args()

    main(args)


