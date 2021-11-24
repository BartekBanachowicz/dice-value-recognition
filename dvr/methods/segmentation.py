import numpy as np
import matplotlib.figure
from matplotlib import pyplot as plt
from skimage import (
    color,
    filters,
    io,
    exposure,
    measure,
    morphology,
    segmentation
)
from scipy import (
    ndimage
)
import math

def _segmentation(source: str, ax: matplotlib.figure.Figure):
    """Attempts to identify the number of pips on a dice

    :param source: path to the input image
    :param ax: ax for result plotting
    :return:
    """
    image_color = io.imread(source)
    image_grayscale = color.rgb2gray(image_color)

    # wyznaczenie markerów
    hist, bin_centers = exposure.histogram(image_grayscale, nbins=3)
    black_limit = bin_centers[0]
    white_limit = bin_centers[1]
    # limity działają w sytuacji, w której zdjęcie zawiera 3 główne kolory:
    # ciemne kropki, jasne tło i ciało kostki w odcieniach szarości

    # Segmentacja nr 1
    elevation_map = filters.sobel(image_grayscale)
    markers = np.zeros_like(image_grayscale)
    markers[image_grayscale > white_limit] = 1
    markers[image_grayscale < black_limit] = 2
    seg = segmentation.watershed(elevation_map, markers)

    seg = ndimage.binary_fill_holes(seg - 1)
    seg = morphology.dilation(seg, selem=morphology.disk(2))
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