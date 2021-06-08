import numpy as np
import SimpleITK as sitk
import os
import argparse
from PIL import Image

def main(params):

    image_list = os.listdir(params.data_root)
    image_list = [item for item in image_list if ".mhd" in item]

    out_files = os.listdir(params.save_dir)

    for image_name in image_list:

        print(image_name)

        if image_name in out_files:
            continue

        image = sitk.ReadImage(os.path.join(params.data_root, image_name))
        image_array = sitk.GetArrayFromImage(image)

        if image_array.shape[0] != 512 or image_array.shape[1] != 512:
            continue

        image_array = image_array - np.min(image_array)
        image_array = image_array/np.max(image_array) * 255
        image_array = image_array.astype(np.uint8)

        mask_volume = np.tile(np.expand_dims(params.mask, -1), (1, 1, image_array.shape[-1]))

        image_array[mask_volume == 0] = 0

        save_image = sitk.GetImageFromArray(image_array)
        save_image.SetSpacing(image.GetSpacing())

        sitk.WriteImage(save_image, os.path.join(params.save_dir, image_name))

if __name__ == '__main__':

    """
    example usage: 
    name.py     --data_root=E:\\NAS\\data\\sweep_wise\\CT_sweep_mhd
                --save_dir=E:\\NAS\\data\\sweep_wise\\CT_sweep_mhd_masked           
                --mask=E:\\Maria\\DataBases\\SpineIFL\\curvilinear_mask_7cm.png
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--mask', type=str)
    args = parser.parse_args()

    mask = Image.open(args.mask)
    resized_maks = mask.resize( (512, 512))
    mask_array = np.array(resized_maks)
    args.mask = np.rot90(mask_array, axes=(0, 1), k=3).astype(np.uint8)

    main(args)
