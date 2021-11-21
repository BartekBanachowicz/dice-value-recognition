import os
import itertools
import dvr
import click
from matplotlib import pyplot as plt


@click.command()
def main():
    path = os.path.join(dvr.__root__, "images/classic_d6")
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [x.path for x in os.scandir(path) if x.is_file()]
    elif files == []:
        print(f"No files found at {path}")
        return

    print("Processing...", end='\r')

    w_size = 0

    for i in itertools.count(1):
        if len(files) <= i**2:
            w_size = i
            break

    h_size = len(files) // w_size + min(1, len(files) % w_size)

    fig= plt.figure(figsize=(10, 10), frameon=False, tight_layout=True)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)

    for i, file in enumerate(files, start=1):
        ax = fig.add_subplot(h_size, w_size, i)
        dvr.methods.segmentation(file, ax)
        ax.set_axis_off()

    fig.savefig('Dices.pdf')
    print(f"{len(files)} files processed, results saved to Dices.pdf")
