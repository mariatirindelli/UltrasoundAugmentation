import plotly.graph_objs as go
from plotly.offline import plot
import numpy as np
from sklearn.decomposition import PCA
import SimpleITK as sitk
from scipy.spatial.transform import Rotation as R

def get_volume_plot(vol):
    X, Y, Z = np.mgrid[0:vol.shape[0], 0:vol.shape[1], 0:vol.shape[2]]

    vol_plot = go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=vol.flatten(),
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=21,  # needs to be a large number for good volume rendering
    )
    return vol_plot


def get_axis_plot(p1, p2):
    vector = go.Scatter3d(x=[p1[0], p2[0]],
                          y=[p1[1], p2[1]],
                          z=[p1[2], p2[2]],
                          marker=dict(size=1,
                                      color="rgb(84,48,5)"),
                          line=dict(color="rgb(84,48,5)",
                                    width=6)
                          )

    return vector

def get_scatter_plot(pts):

    vector = go.Scatter3d(x=pts[:, 0],
                          y=pts[:, 1],
                          z=pts[:, 2]
                          )
    return vector

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
  return sitk.Resample(image, reference_image, transform,
                     interpolator, default_value)


def get_center(img):
  """
  This function returns the physical center point of a 3d sitk image
  :param img: The sitk image we are trying to find the center of
  :return: The physical center point of the image
  """
  width, height, depth = img.GetSize()
  return img.TransformIndexToPhysicalPoint((int(np.ceil(width/2)),
                                          int(np.ceil(height/2)),
                                          int(np.ceil(depth/2))))


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

  image_center_tmp = get_center(image)
  image_center = (center_x, center_y, center_z)
  euler_transform = sitk.Euler3DTransform(image_center, theta_x, theta_y, theta_z, (0, 0, 0))

  euler_transform.SetCenter(image_center)
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


sitk_image = sitk.ReadImage("tmp.mhd")
vol = sitk.GetArrayFromImage(sitk_image)
point_cloud = np.argwhere(vol>0)

normal, _, centroid, _ = fit_plane(point_cloud)

center_x, center_y, center_z = centroid[1], centroid[2], centroid[0]

rotated_volume = rotation3d(sitk_image, 45, 0, 0, center_x, center_y, center_z)
rotated_array = sitk.GetArrayFromImage(rotated_volume)

rotated_points = np.argwhere(rotated_array > 10)

point_cloud1 = get_scatter_plot(point_cloud)
point_cloud2 = get_scatter_plot(rotated_points)

x_axis = get_axis_plot([0, 0, 0], [rotated_array.shape[0], 0, 0])
y_axis = get_axis_plot([0, 0, 0], [0, rotated_array.shape[1], 0])
z_axis = get_axis_plot([0, 0, 0], [0, 0, rotated_array.shape[2]])

# data2 = [vold_data_rot, vector_system]
vector_system = get_axis_plot(centroid, centroid+normal*50)
data2 = [point_cloud1, point_cloud2, vector_system, x_axis, y_axis, z_axis]
fig2 = go.Figure(data=data2)

fig2.show()



# rotated_volume = rotation3d(sitk_image, theta_x, theta_y, theta_z, center_x, center_y, center_z)
# rotated_array = sitk.GetArrayFromImage(rotated_volume)
#
# print(np.max(rotated_array))
#
# sitk.WriteImage(sitk.GetImageFromArray(rotated_array), "rotated_real.mhd")
# sitk.WriteImage(sitk.GetImageFromArray(vol), "non_rotated_real.mhd")
#
#
# point_cloud = get_scatter_plot(point_cloud)
# vold_data_rot = get_volume_plot(vol)
# vector_system = get_axis_plot(centroid, centroid+normal*50)
#
# x_axis = get_axis_plot([0, 0, 0], [vol.shape[0], 0, 0])
# y_axis = get_axis_plot([0, 0, 0], [0, vol.shape[1], 0])
# z_axis = get_axis_plot([0, 0, 0], [0, 0, vol.shape[2]])
#
# # data2 = [vold_data_rot, vector_system]
# data2 = [point_cloud, vector_system, x_axis, y_axis, z_axis]
# fig2 = go.Figure(data=data2)
#
#
#
#
# fig2.show()


