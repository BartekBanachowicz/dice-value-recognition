import os
import dvr
import click
import matplotlib.cm as cm
import numpy as np
from skimage import (
    io,
    segmentation,
    filters,
    measure,
    morphology
)
from scipy import ndimage
from matplotlib import pyplot as plt

@click.command()
def main():
    path = os.path.join(dvr.__root__, "images/classic_d6")
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    dices = [i for i in range(1, 7)]

    for i in range(0, len(dices)):
        image = io.imread(os.path.join(path, f"{dices[i]}.jpg"), as_gray=True)
        image_color = io.imread(os.path.join(path, f"{dices[i]}.jpg"))
        color_array = iter(cm.rainbow(np.linspace(0, 1, 20)))

        # SEGMENTACJA 01

        elevation_map = filters.sobel(image)
        markers = np.zeros_like(image)
        markers[image > 0.8] = 1
        markers[image < 0.3] = 2
        seg = segmentation.watershed(elevation_map, markers)

        # SEGMENTACJA_02

        elevation_map2 = filters.scharr(image)
        # markers2 = np.zeros_like(image)
        # markers2[image > 0.6] = 1
        # markers2[image < 0.3] = 2
        # seg2 = segmentation.watershed(elevation_map2, markers2)

        # seg = segmentation.join_segmentations(seg, seg2)

        seg = ndimage.binary_fill_holes(seg - 1)
        seg = morphology.dilation(seg, selem=morphology.disk(2))
        labeled, _ = ndimage.label(seg)
        contours = measure.find_contours(seg)

        axes[int(i / 2), i % 2].imshow(image_color)
        for contour in contours:
            axes[int(i / 2), i % 2].plot(contour[:, 1], contour[:, 0], linewidth=2)
        axes[int(i / 2), i % 2].axis('off')
        axes[int(i / 2), i % 2].set_title(f"{len(contours)}")

    plt.tight_layout()
    fig.savefig('Dices.pdf')
    plt.show()
