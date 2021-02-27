import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def get_angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def image_with_colorbar(fig, ax, image, cmap=None, title=""):

    if cmap is None:
        pos0 = ax.imshow(image, clim=(0, 1))
    else:
        pos0 = ax.imshow(image, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax0 = divider.append_axes("right", size="5%", pad=0.05)
    tick_list = np.linspace(np.min(image), np.max(image), 5)
    cbar0 = fig.colorbar(pos0, cax=cax0, ticks=tick_list, fraction=0.001, pad=0.05)
    cbar0.ax.set_yticklabels(["{:.2f}".format(item) for item in tick_list])  # vertically oriented colorbar
