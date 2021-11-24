from copy import deepcopy
import numpy as np
import matplotlib.figure
from skimage import (
    color,
    exposure,
    filters,
    io,
    measure,
    morphology,
    segmentation
)
from scipy import ndimage, signal


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

    # image = deepcopy(image_color)
    # r_average = image_color[..., 0].mean() - 0.3
    # g_average = image_color[..., 1].mean() - 0.1
    # b_average = image_color[..., 2].mean() - 0.1
    # image[image[..., 0] > r_average, 0] = r_average
    # image[image[..., 1] > g_average, 1] = g_average
    # image[image[..., 2] > b_average, 2] = b_average
    #
    # ax.imshow(image)

    image_grayscale = color.rgb2gray(image_color)

    image = image_grayscale

    # image = image_grayscale
    # image = morphology.opening(image, selem=morphology.disk(1))
    # image = morphology.closing(image, selem=morphology.disk(1))

    # image = morphology.erosion(image, selem=morphology.disk(1))

    # ax.imshow(image, cmap="gray")

    hist, bins = np.histogram(image, bins=256)

    # ax.hist(image.flat, bins=256)

    # peaks, _ = signal.find_peaks(hist, distance=15, plateau_size=(1, 1))
    # threshold = min(15, peaks[1] - peaks[0])
    # left = max(0, peaks[0] - threshold)
    # right = min(hist.size - 1, peaks[0] + threshold)
    # left = bins[left]
    # right = bins[right + 1]
    # image_binned = np.digitize(image, bins=[0.0, left, right, 1.0])
    markers = np.full_like(image, 1.0)
    # markers[image_binned == 2] = 2.0

    # markers[image] = 1.0

    markers[
        (image_color[..., 0] < 0.3) &
        (image_color[..., 1] < image_color[..., 0] + 0.3) &
        (image_color[..., 1] > image_color[..., 0] - 0.3) &
        (image_color[..., 2] < image_color[..., 0] + 0.3) &
        (image_color[..., 2] > image_color[..., 0] - 0.3)
    ] = 2.0

    # common = np.where(hist == hist.max())[0][0]
    # common = np.where(peaks >= common)[0][0]
    # threshold = abs(min(128, peaks[common] - peaks[common - 1]))
    # left = max(0, peaks[common] - threshold)
    # right = min(hist.size - 1, peaks[common] + threshold)
    # left = bins[left]
    # right = bins[right + 1]
    # image_binned = np.digitize(image, bins=[0.0, left, right, 1.0])
    # markers[image_binned == 2] = 1.0

    # ax.imshow(markers, cmap="gray")

    elevation_map = filters.sobel(image)
    seg = segmentation.watershed(elevation_map, markers)
    seg = morphology.dilation(seg, selem=morphology.disk(2))
    seg = ndimage.binary_fill_holes(seg - 1)
    seg = morphology.erosion(seg)
    contours = measure.find_contours(seg)

    ax.imshow(image_color)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.set_axis_off()
    # ax.set_title(f"{len(contours)}")
