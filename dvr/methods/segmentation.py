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
    image_grayscale = color.rgb2gray(image_color)

    # Segmentacja nr 1
    elevation_map = filters.sobel(image_grayscale)
    markers = np.zeros_like(image_grayscale)
    markers[image_grayscale > 0.8] = 1
    markers[image_grayscale < 0.3] = 2
    seg = segmentation.watershed(elevation_map, markers)

    # Segmentacja nr 2
    # elevation_map_2 = filters.scharr(image_grayscale)
    # markers_2 = np.zeros_like(image_grayscale)
    # markers_2[image_grayscale > 0.6] = 1
    # markers_2[image_grayscale < 0.3] = 2
    # seg_2 = segmentation.watershed(elevation_map_2, markers_2)

    # Łączenie segmentacji
    # seg = segmentation.join_segmentations(seg, seg2)

    seg = ndimage.binary_fill_holes(seg - 1)
    seg = morphology.dilation(seg, selem=morphology.disk(2))
    contours = measure.find_contours(seg)

    # labeled, _ = ndimage.label(seg)

    ax.imshow(image_color)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_title(f"{len(contours)}")
