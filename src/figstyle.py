"""Shared print style for the figures.

Text is set by LaTeX in Times (the manuscript's font) through Matplotlib's PGF backend when pdflatex is available;
otherwise the STIX fonts, which match Times closely, are used. Figures are drawn at the manuscript's text width so
that they are included at their natural size.
"""
import shutil

import matplotlib

WIDTH = 6.30  # inches: 160 mm, the text width of the A4 manuscript
TEX = shutil.which("pdflatex") is not None


def setup():
    if TEX:
        matplotlib.use("pgf")
        fonts = {"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "font.family": "serif",
                 "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{mathptmx}"}
    else:
        matplotlib.use("Agg")
        fonts = {"font.family": "STIXGeneral", "mathtext.fontset": "stix"}
    import matplotlib.pyplot as plt

    plt.rcParams.update(fonts)
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.5, "axes.edgecolor": "black", "axes.labelpad": 3, "axes.titlepad": 4,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
        "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "lines.linewidth": 0.9, "lines.markersize": 3.2,
        "legend.frameon": False, "legend.handlelength": 2.2, "legend.columnspacing": 2.0,
        "pdf.fonttype": 42, "savefig.dpi": 300, "savefig.facecolor": "white",
    })
    return plt


def tt(name):
    """A condition name in the typewriter face used in the text."""
    s = name.replace("_", r"\_")
    return rf"\texttt{{{s}}}" if TEX else rf"$\mathtt{{{s}}}$"
