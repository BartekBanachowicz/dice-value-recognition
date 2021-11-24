import math
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
    labeled, _ = ndimage.label(seg)

    # odczytanie średnich pól oczek
    areas = []
    centers = []
    for props in measure.regionprops(labeled):
        (y, x) = props.centroid
        centers.append(props.centroid)
        areas.append(props.area)
        ax.plot(x, y, marker='o', markersize=2, color='white')
    avg_area = sum(areas)/len(areas)

    # max odległośc między oczkami: ~przekątna kostki
    r = math.sqrt(avg_area/math.pi)
    max_distance = 8*r*math.sqrt(2)

    # wykrywanie czy dane oczka należą do tej samej kostki
    same_dice = np.zeros((len(centers), len(centers)))
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = math.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
            if dist <= max_distance:
                same_dice[i][j] = 1

    # liczenie kropek na tych samych kostkach
    i = 0
    dice_numbers = []
    rows_deleted = []
    while i < len(same_dice):
        dice_numbers.append(1)
        rows_deleted.append(0)
        j = i + 1 + rows_deleted[-1]
        while j < len(same_dice[i]):
            if same_dice[i][j] == 1:
                dice_numbers[-1] += 1
                same_dice = np.delete(same_dice, j - sum(rows_deleted), 0)
                rows_deleted[-1] += 1
            j += 1
        i += 1

    print("dice numbers:", dice_numbers)

    ax.imshow(image_color)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax_title = "Liczba oczek: "
    for num in dice_numbers:
        ax_title += "{} ".format(num)
    ax.set_title(ax_title)
    ax.set_axis_off()
