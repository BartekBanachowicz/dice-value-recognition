import math
import pylab
import numpy
import numpy as np
import matplotlib.figure
import skimage
from skimage import (
    color,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    exposure
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

    # Zmiana intensywności

    percentile_low, percentile_high = np.percentile(image_grayscale, (0, 60))
    image_grayscale = exposure.rescale_intensity(image_grayscale, out_range=(percentile_low, percentile_high))

    # Wyznaczanie markerów
    markers = np.full_like(image_grayscale, 1.0)
    markers[
        (image_color[..., 0] < 0.3) &
        (image_color[..., 1] < 0.3) &
        (image_color[..., 1] < image_color[..., 0] + 0.3) &
        (image_color[..., 1] > image_color[..., 0] - 0.3) &
        (image_color[..., 2] < 0.3) &
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

    # Odczytanie średnich pól oczek
    labeled, _ = ndimage.label(seg)
    properties = measure.regionprops(labeled)
    areas = numpy.array([i.area for i in properties])
    centers = numpy.array([i.centroid for i in properties])
    area = numpy.median(areas)

    # Max odległośc między oczkami - przekątna kostki
    r = math.sqrt(area / math.pi)
    max_distance = 6 * r * math.sqrt(2)

    # Wykrywanie czy dane oczka należą do tej samej kostki
    dices = numpy.array([i for i in range(1, len(properties) + 1)])
    for i in range(len(properties)):
        for j in range(i + 1, len(properties)):
            dist = math.sqrt((centers[i][0] - centers[j][0])**2 + (centers[i][1] - centers[j][1])**2)
            if dist <= max_distance:
                dices[j] = dices[i]

    # Wyświetlanie i kolorowanie
    dice_numbers, counts = np.array(np.unique(dices, return_counts=True))

    cm = pylab.get_cmap("gist_rainbow")
    cm = {j : cm(i / dice_numbers.size) for i, j in enumerate(dice_numbers)}

    image_mask = numpy.zeros((image_grayscale.shape[0], image_grayscale.shape[1], 4), dtype=np.float64)
    for i in range(len(properties)):
        image_mask[properties[i].coords[:, 0], properties[i].coords[:, 1], :] = cm[dices[i]]

    ax.imshow(image_color)
    ax.imshow(image_mask)

    print()
    print('\n'.join([f"{counts[i]}: {cm[j]}" for i, j in enumerate(dice_numbers)]))

    ax.set_axis_off()
