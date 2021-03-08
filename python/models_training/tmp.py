from shutil import copy2
import os
import random
from PIL import Image
import numpy as np

datadir = "C:\\Users\\maria\\Downloads\\bones_bmode\\bones_bmode\\images"
labeldir = "C:\\Users\\maria\\Downloads\\bones_bmode\\bones_bmode\\labels"
dst_dir = "C:\\Users\\maria\\OneDrive\\Desktop\\bones"

data_list = os.listdir(datadir)
val = ["14", "17", "11", "7"]

train_list = [item for item in data_list if val[0] not in item and
              val[1] not in item and
              val[2] not in item and
              val[3] not in item]

val_list = [item for item in data_list if val[0] in item or
              val[1] in item or
              val[2] in item or
              val[3] in item]

train_data = random.sample(train_list, 400)
val_data = random.sample(train_list, 100)

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
