import wandb


# testing sliding image as https://plotly.com/python/imshow/

#wandb.init(project="plot tests")

import plotly.express as px
from skimage import io

data = io.imread("https://github.com/scikit-image/skimage-tutorials/raw/main/images/cells.tif")
img = data[25:40]
fig = px.imshow(img, animation_frame=0, binary_string=True, labels=dict(animation_frame="slice"))

wandb.log({"my_chart": fig})