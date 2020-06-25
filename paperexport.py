import os
import sys
import argparse
import functools
import itertools
import string
import bisect
import tempfile

import numpy as np
import numpy.core.defchararray as npstr
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import matplotlib.path as mpath
import matplotlib.ticker as ticker
from astropy.table import Table, vstack, unique
from astropy.io import ascii, fits
import astropy.units as u
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.special import erf, erfc
from scipy import stats
from astroquery.gaia import Gaia
import emcee
import corner

# Until I figure out how to use Matplotlib styles. Use this instead.
mpl.rcParams["axes.titlesize"] = 24
mpl.rcParams["axes.labelsize"] = 26
mpl.rcParams["lines.linewidth"] = 5
mpl.rcParams["lines.markersize"] = 10
mpl.rcParams["xtick.labelsize"] = 16
mpl.rcParams["ytick.labelsize"] = 16
mpl.rcParams["legend.fontsize"] = 14

sys.path.append(os.path.join(os.environ["RESEARCH"], "Binaries", "scripts"))
import observations as obs
import path_config as paths
import read_catalog as catin
import hrplots as hr
import astropy_util as au
import catalog
import sed
import data_splitting as split
import biovis_colors as bc
import aspcap_corrections as aspcor
import data_cache as cache
import rotation_consistency as rot
import sample_characterization as samp
import mist
import dsep
import baraffe
import models
import yrec
import extinction
import jenboundary as jen
import binarycalcs as bincalc

PAPER_PATH = paths.HOME_DIR / "Documents" / "Papers" / "rvvar20"
TABLE_PATH = PAPER_PATH / "tables"
FIGURE_PATH = PAPER_PATH / "fig"
PLOT_SUFFIX = "pdf"
PLOT_PATH = paths.HEAD_DIR / "plots"

figsize=(9, 9)

Protstr = r"$P_{\mathrm{rot}}$"
vsinistr = r"$v \sin i$"
kmsstr = r"km s$^{-1}$"
Teffstr = r"$T_{\mathrm{eff}}$"
MKstr = r"$M_{Ks}$"
fehstr = r"$[Fe/H]$"

def build_filepath(toplevel, filename, suffix=PLOT_SUFFIX):
    '''Generate a full path to save a filename.'''

    fullpath = toplevel / ".".join((filename, suffix))
    return str(fullpath)

def write_plot(filename, suffix=PLOT_SUFFIX, toplevel=FIGURE_PATH):
    '''Create a decorator that will write plots to a given file.'''
    def decorator(f): 
        @functools.wraps(f)
        def wrapper():
            plt.close("all")
            a = f()
            filepath = build_filepath(toplevel, filename, suffix=suffix)
            # Have a hidden file that is modified every time the figure is
            # written. This will work better with make.
#            touch(build_filepath(toplevel.parent, "."+filename, suffix="txt"))
            plt.savefig(filepath)
            return a
        return wrapper
    return decorator

def plot_MDM_targets():
    '''Plot the MDM targets on an HR diagram.'''
    splitter = split.MDMSplitter()
    split.initialize_mdm_sample(splitter)

    RVtargs = splitter.subsample(["MDM RV Targets"])
    dlsb = splitter.subsample(["MDM DLSB"])
    slsb = splitter.subsample(["MDM SLSB"])
    others = splitter.subsample(["Not Observed"])
    print(dlsb[["teff", "M_K", "D"]])

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        others["teff"], others["M_K"], color="grey", marker=".", ls="", 
        alpha=0.1, label="McQuillan", axis=ax)
    hr.absmag_teff_plot(
        RVtargs["teff"], RVtargs["M_K"], color=bc.red, marker="o", ls="", 
        label="MDM", axis=ax)
    hr.absmag_teff_plot(
        dlsb["teff"], dlsb["M_K"], color=bc.blue, marker="o", ls="", 
        label="DLSB", axis=ax)
    hr.absmag_teff_plot(
        slsb["teff"], slsb["M_K"], color=bc.pink, marker="o", ls="", 
        label="SLSB", axis=ax)
    plt.legend(loc="lower left")
    ax.set_xlabel(Teffstr)
    ax.set_ylabel(MKstr)
