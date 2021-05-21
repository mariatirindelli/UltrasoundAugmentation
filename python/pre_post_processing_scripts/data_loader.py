from fix_pythonpath import *
import imfusion
import numpy as np


def load_imfusion_sweep(sweep_path):
    sweep = imfusion.open(sweep_path)[0]
    image_list = [np.array(item).astype(np.uint8) for item in sweep]

    data_array = np.stack(image_list, axis=0)

    meta_data = {'timestamps': [item.matrix for item in sweep]}

    return data_array, meta_data

def load_imfusion_data(sweep_path):
    sweep = imfusion.open(sweep_path)[0]
    if isinstance(sweep, imfusion.UltrasoundSweep):
        return load_imfusion_sweep(sweep_path)
    else:
        raise NotImplementedError


def load_data(data_path):
    """
    Returns the data in data_path as (channels, height, width) and associated metadata
    """

    fmt = data_path.split(".")[-1]

    if fmt == ".imf":
        imfusion.init()
        return load_imfusion_data(data_path)

    else:
        raise NotImplementedError
