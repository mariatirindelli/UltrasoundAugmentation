from shutil import copy2
import os
import random
from PIL import Image
import numpy as np

datadir = "C:\\Users\\maria\\Downloads\\bones_bmode\\bones_bmode\\images"
labeldir = "C:\\Users\\maria\\Downloads\\bones_bmode\\bones_bmode\\labels"
dst_dir = "C:\\GitRepo\\UltrasoundAugmentation\\python\\models_training\\datasets\\bones"

data_list = os.listdir(datadir)
val = ["14", "17", "11", "7"]
test = ["1", "3"]

train_list = [item for item in data_list if val[0] != item.split("_")[0] and
              val[1] != item.split("_")[0] and
              val[2] != item.split("_")[0] and
              val[3] != item.split("_")[0] and
              test[0] != item.split("_")[0] and
              test[1] != item.split("_")[0]]

val_list = [item for item in data_list if val[0] == item.split("_")[0] or
            val[1] == item.split("_")[0] or
            val[2] == item.split("_")[0] or
            val[3] == item.split("_")[0]]

test_list = [item for item in data_list if test[0] == item.split("_")[0] or
            test[1] == item.split("_")[0]]

train_data = random.sample(train_list, 400)
val_data = random.sample(val_list, 100)
test_data = random.sample(test_list, 100)

for item in train_data:
    copy2(os.path.join(datadir, item), os.path.join(dst_dir, "train"))
    inputdir = os.path.join(labeldir, item.replace(".png", "_label.png"))
    outputdir = os.path.join(dst_dir, "train", item.replace(".png", "_label.png"))

    im = Image.open(inputdir)
    data = np.array(im)  # "data" is a height x width x 4 numpy array
    data[data == 0] = 100
    data[data == 1] = 250

    im2 = Image.fromarray(data)
    im2.save(outputdir)

for item in val_data:
    copy2(os.path.join(datadir, item), os.path.join(dst_dir, "val"))

    inputdir = os.path.join(labeldir, item.replace(".png", "_label.png"))
    outputdir = os.path.join(dst_dir, "val", item.replace(".png", "_label.png"))

    im = Image.open(inputdir)
    data = np.array(im)  # "data" is a height x width x 4 numpy array
    data[data == 0] = 100
    data[data == 1] = 250

    im2 = Image.fromarray(data)
    im2.save(outputdir)

for item in test_data:
    copy2(os.path.join(datadir, item), os.path.join(dst_dir, "test"))

    inputdir = os.path.join(labeldir, item.replace(".png", "_label.png"))
    outputdir = os.path.join(dst_dir, "test", item.replace(".png", "_label.png"))

    im = Image.open(inputdir)
    data = np.array(im)  # "data" is a height x width x 4 numpy array
    data[data == 0] = 100
    data[data == 1] = 250

    im2 = Image.fromarray(data)
    im2.save(outputdir)
