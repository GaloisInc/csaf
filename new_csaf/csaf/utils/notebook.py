from csaf import System
import typing
import pathlib
import subprocess
import sys


def open_image(path):
    """open an image in a OSs default image viewer"""
    imageViewerFromCommandLine = {'linux':'xdg-open',
                                  'win32':'explorer',
                                  'darwin':'open'}[sys.platform]
    subprocess.run([imageViewerFromCommandLine, path])


def view_block_diagram(system: typing.Union[System, typing.Type[System]],
                       ipython_notebook=False):
    """convenience function to quickly view a system composition"""
    plot_fname = f".pub-sub-plot.png"
    # plot configuration pub/sub diagram as a file -- proj specicies a dot executbale and -Gdpi is a valid dot
    # argument to change the image resolution
    system = system() if isinstance(system, type) else system
    system.plot_config(fname=pathlib.Path(plot_fname).resolve(), prog=["dot", "-Gdpi=400"])

    if ipython_notebook:
        from IPython.display import Image
        return Image(plot_fname, height=800, width=800)
    else:
        open_image(plot_fname)
