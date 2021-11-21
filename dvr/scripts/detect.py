import os
import itertools
import dvr
import click
from matplotlib import pyplot as plt


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "interactive", "-i", "--interactive", is_flag=True,
    help="whether matplotlib backend should be used to display the results"
)
@click.option(
    "output", "-o", "--output", type=click.Path(), default="result.pdf",
    help="output filename"
)
@click.argument("mode", type=click.Choice(["segmentation"], case_sensitive=False))
@click.argument("input_",  metavar="INPUT", type=click.Path(exists=True))
def main(interactive: bool, output: str, mode: str, input_: str):
    """A tool used for detecting dice values

    \b
    MODE should be one of the available processing modes
    INPUT should be an image file or a directory containing multiple images
    """
    files = []
    if os.path.isfile(input_):
        files = [input_]
    elif os.path.isdir(input_):
        files = [x.path for x in os.scandir(input_) if x.is_file()]
    elif files == []:
        print(f"No files found at {input_}")
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
        dvr.methods.segmentation(file, ax) if mode == "segmentation" else None
        ax.set_axis_off()

    fig.savefig(output)
    plt.show() if interactive else None
    print(f"{len(files)} files processed, results saved to {output}")
