import numpy as np
from PIL import Image
from fix_pythonpath import *
import logging
import argparse
import imfusion
from pre_post_processing_scripts.proc_utils import str2bool, images2sweep
from PIL import Image
from joblib import Parallel, delayed

imfusion.init()


def load_png(image_path_list, rescale=False, rescale_size=(255, 255), hflip=False,
                  vflip=False):

    image_list = []
    for image_path in image_path_list:
        image = Image.open(image_path)
        imf_image = imfusion.SharedImage(image)
        image_list.append(imf_image)

    return image_list

def rescale_image(np_image, shape, method=Image.BICUBIC):
    img = Image.fromarray(np_image)

    resized_image = img.resize(shape, method)
    print(resized_image.size)

    return np.array(resized_image)

def parallelized_load_png(image_path_list, rescale=False, rescale_size=(255, 255), hflip=False,
                  vflip=False):

    images = Parallel(n_jobs=4, verbose=5)(
        delayed(load_png)(f) for f in image_path_list
    )
    return images

def load_npy(image_path, rescale=False, rescale_size=(255, 255), hflip=False,
                  vflip=False):

    image = np.load(image_path)
    resize_image = rescale_image(image, [544, 516])

    if len(image.shape) == 2:
        image = np.expand_dims(resize_image, axis=-1)

    return image

def parallelized_load_npy(image_path_list, rescale=False, rescale_size=(255, 255), hflip=False,
                  vflip=False):

    images = Parallel(n_jobs=4, verbose=5)(
        delayed(load_npy)(f) for f in image_path_list
    )

    np_images = np.stack(images)
    return np_images

def print_algs():
    for item in imfusion.availableAlgorithms():
        print(item)


def create_sweeps(hparams):
    print_algs()

    image_list = [item for item in os.listdir(hparams.data_path) if hparams.postfix in item]
    subject_data = images2sweep(image_list)

    for subject_id in subject_data.keys():
        for sweep_id in subject_data[subject_id].keys():

            data_list = [os.path.join(hparams.data_path, item) for item in subject_data[subject_id][sweep_id]]
            np_image_sweep = parallelized_load_npy(data_list)

            image_set = imfusion.SharedImageSet(np.zeros(np_image_sweep.shape, dtype='float32'))
            image_set.assignArray(np_image_sweep)

            imfusion.executeAlgorithm('Set Spacing', [image_set], imfusion.Properties({'spacing': hparams.spacing}))

            save_location = os.path.join(hparams.output_path, subject_id + "_" + sweep_id + "_" +
                                         hparams.postfix + ".imf")
            print("Saving result in: {}".format(save_location))

            imfusion.executeAlgorithm('IO;ImFusionFile', [image_set], imfusion.Properties({'location': save_location}))


def parse_args(arg_parser):
    args = arg_parser.parse_args()

    if not args.rescale:
        args.rescale_size = None
    else:
        args.rescale_size = [int(item) for item in args.rescale_size.split(",")]

    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--spacing', type=str, default='0.14, 0.14, 1')
    parser.add_argument('--hflip', type=str2bool, default=True)
    parser.add_argument('--vflip', type=str2bool, default=False)
    parser.add_argument('--rescale', type=str, default=False)
    parser.add_argument('--rescale_size', type=str, default="544,516")
    parser.add_argument('--clockwise_rotate', type=float, default=0)
    parser.add_argument('--postfix', type=str, default='fake')

    # todo: divide by postfix

    parsed_args = parse_args(parser)

    for postfix in ['fake', 'label']:
        parsed_args.postfix = postfix

        create_sweeps(parsed_args)

