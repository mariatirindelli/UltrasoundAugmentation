import SimpleITK as sitk
from pre_post_processing_scripts.generate_ct_sweeps import get_random_centroid
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'png'


def show_centroid(image, centroid):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    cv2.circle(img=img_rgb, center=(centroid[1], centroid[0]), radius=5, color=(255, 0, 0), thickness=2)
    return img_rgb

def main():
    skin_itk = sitk.ReadImage("test_file/skin_label.mhd")
    skin = sitk.GetArrayFromImage(skin_itk) * 255

    centroid = get_random_centroid(skin)

    # roi = skin[centroid[0] - 30:centroid[0] + 30, centroid[1] - 30:centroid[1] + 30, centroid[2] - 30:centroid[2] + 30]

    centroid_0 = centroid[0]

    data = np.zeros([skin.shape[2], skin.shape[0], skin.shape[1], 3])
    for i_3 in range(skin.shape[-1]):
        image_roi = np.squeeze(skin[:, :, i_3])
        centroid_1 = np.argwhere(image_roi[centroid_0, :] > 0)[-1][0]

        print(i_3)

        img = show_centroid(image_roi, [centroid_0, centroid_1])
        data[i_3, :, :, :] = img

    fig = px.imshow(data, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))
    fig.show()


    input("")



main()