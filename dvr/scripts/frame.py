import os
import dvr
import click
import matplotlib.cm as cm
import numpy as np
from skimage import (
    io,
    segmentation,
    filters,
    measure
)
from scipy import ndimage
from matplotlib import pyplot as plt

@click.command()
def main():
    path = os.path.join(dvr.__root__, "images/sets")
    fig, axes = plt.subplots(5, 2, figsize=(15, 15))

    for i in range(1, 11):
        image = io.imread(os.path.join(path, f"{i}.jpg"), as_gray=True)
        image_color = io.imread(os.path.join(path, f"{i}.jpg"))
        color_array = iter(cm.rainbow(np.linspace(0, 1, 20)))

        # SEGMENTACJA 01

        elevation_map = filters.sobel(image)
        markers = np.zeros_like(image)
        markers[image > 0.8] = 1
        markers[image < 0.2] = 2
        seg = segmentation.watershed(elevation_map, markers)

        # SEGMENTACJA_02

        elevation_map2 = filters.scharr(image)
        markers2 = np.zeros_like(image)
        markers2[image > 0.6] = 1
        markers2[image < 0.3] = 2
        seg2 = segmentation.watershed(elevation_map2, markers2)

        seg = segmentation.join_segmentations(seg, seg2)

        seg = ndimage.binary_fill_holes(seg - 1)
        labeled, _ = ndimage.label(seg)

        contours = measure.find_contours(labeled, 0.8)

        for props in measure.regionprops(labeled):
            y, x = props.centroid
            axes[int(i / 2), i % 2].plot(x, y, marker='o', markersize=2, color='white')
        axes[int(i / 2), i % 2].imshow(image_color)
        for contour in contours:
            axes[int(i / 2), i % 2].plot(contour[:, 1], contour[:, 0], linewidth=2)
        axes[int(i / 2), i % 2].axis('off')

    plt.tight_layout()
    fig.savefig('Dices.pdf')
    plt.show()
