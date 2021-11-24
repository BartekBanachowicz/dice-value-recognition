import math

import numpy
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
    # Wczytywanie i konwersja kolorów
    image_color = io.imread(source)
    if image_color.shape[-1] == 4:
        image_color = color.rgba2rgb(image_color)

    image_color = image_color / 255
    image_grayscale = color.rgb2gray(image_color)

    # Wyznaczanie markerów
    markers = np.full_like(image_grayscale, 1.0)
    markers[
        (image_color[..., 0] < 0.3) &
        (image_color[..., 1] < image_color[..., 0] + 0.3) &
        (image_color[..., 1] > image_color[..., 0] - 0.3) &
        (image_color[..., 2] < image_color[..., 0] + 0.3) &
        (image_color[..., 2] > image_color[..., 0] - 0.3)
    ] = 2.0

    # Segmentacja
    elevation_map = filters.sobel(image_grayscale)
    seg = segmentation.watershed(elevation_map, markers)
    seg = morphology.dilation(seg, selem=morphology.disk(2))
    seg = ndimage.binary_fill_holes(seg - 1)
    seg = morphology.erosion(seg, selem=morphology.disk(3))
    # seg = morphology.erosion(seg, selem=morphology.disk(2))
    contours = measure.find_contours(seg)

    # Odczytanie średnich pól oczek
    labeled, _ = ndimage.label(seg)
    properties = measure.regionprops(labeled)
    areas = numpy.array([i.area for i in properties])
    centers = numpy.array([i.centroid for i in properties])
    area = numpy.median(areas)

    # Max odległośc między oczkami - przekątna kostki
    r = math.sqrt(area / math.pi)
    max_distance = 8 * r * math.sqrt(2)

    # Wykrywanie czy dane oczka należą do tej samej kostki
    dices = numpy.array([i for i in range(1, len(properties) + 1)])
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            dist = math.sqrt((centers[i][0] - centers[j][0]) ** 2 + (centers[i][1] - centers[j][1]) ** 2)
            if dist <= max_distance:
                dices[j] = dices[i]

    # liczenie kropek na tych samych kostkach
    dice_numbers, bins = numpy.histogram(dices, bins=[i for i in range(1, dices.max() + 2)])

    ax.imshow(image_color)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    print(dice_numbers)
    # ax.set_title(f"Liczba oczek: {' '.join(map(str, dice_numbers))}")
    ax.set_axis_off()
