from copy import deepcopy
import numpy as np
import matplotlib.figure
from skimage import (
    color,
    filters,
    io,
    measure,
    morphology,
    segmentation
)
from scipy import ndimage


def _segmentation(source: str, ax: matplotlib.figure.Figure):
    """Attempts to identify the number of pips on a dice

    :param source: path to the input image
    :param ax: ax for result plotting
    :return:
    """
    image_color = io.imread(source)
    if image_color.shape[-1] == 4:
        image_color = color.rgba2rgb(image_color)

    image_color = image_color / 255
    image_grayscale = color.rgb2gray(image_color)

    markers = np.full_like(image_grayscale, 1.0)
    markers[
        (image_color[..., 0] < 0.3) &
        (image_color[..., 1] < image_color[..., 0] + 0.3) &
        (image_color[..., 1] > image_color[..., 0] - 0.3) &
        (image_color[..., 2] < image_color[..., 0] + 0.3) &
        (image_color[..., 2] > image_color[..., 0] - 0.3)
    ] = 2.0

    elevation_map = filters.sobel(image_grayscale)
    seg = segmentation.watershed(elevation_map, markers)
    seg = morphology.dilation(seg, selem=morphology.disk(2))
    seg = ndimage.binary_fill_holes(seg - 1)
    seg = morphology.erosion(seg)
    contours = measure.find_contours(seg)

    ax.imshow(image_color)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_axis_off()
