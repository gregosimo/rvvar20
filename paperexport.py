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
    splitter = cache.mdm_targets()

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
    ax.set_title("MDM Targets with Gaia")

def APOGEE_tidsync_target_quality(
        speclook=False, berger_companions=False, vsini_check=False):
    '''Explore the APOGEE tidally synchronzied candidate quality.

    The type of exploration depends on the keyword flags enabled.

    Speclook will pull up the spectra for the targets in a web browser.'''
    splitter = cache.apogee_tidsync_targets()

    rv_variable_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "RV Variable"])
    rv_nonvariable_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "RV Nonvariable"])
    single_visit_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "Single Visit"])
    rv_variable_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "RV Variable"])
    rv_nonvariable_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "RV Nonvariable"])
    single_visit_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "Single Visit"])

    single_visit_chi2_tidsync = (
        (single_visit_tidsync["VHELIO_AVG"] - 
         single_visit_tidsync["radial_velocity"])**2 /
    (0.5**2 + single_visit_tidsync["radial_velocity_error"]**2))
    gaia_rv_variable_tidsync = au.multi_logical_and(
        ~single_visit_chi2_tidsync.mask, single_visit_chi2_tidsync > 5.5)
    gaia_rv_nonvariable_tidsync = au.multi_logical_and(
        ~single_visit_chi2_tidsync.mask, single_visit_chi2_tidsync <= 5.5)
    gaia_ambiguous_tidsync = single_visit_chi2_tidsync.mask

    single_visit_chi2_cool = (
        (single_visit_cool["VHELIO_AVG"] - 
         single_visit_cool["radial_velocity"])**2 /
    (0.5**2 + single_visit_cool["radial_velocity_error"]**2))
    gaia_rv_variable_cool = au.multi_logical_and(
        ~single_visit_chi2_cool.mask, single_visit_chi2_cool > 5.5)
    gaia_rv_nonvariable_cool = au.multi_logical_and(
        ~single_visit_chi2_cool.mask, single_visit_chi2_cool <= 5.5)
    gaia_ambiguous_cool = single_visit_chi2_cool.mask

    # First let's look at the APOGEE spectra
    if speclook:
        print("RV Variable targets:")
        # 2M19012311+3939550 looks like a normal spectrum.
        # 2M19502492+4629099 looks normal with spectrum. COLORTE_WARN is
        # triggered.
        # 2M19045950+5037063 may be an SB2. There does seem to be a template
        # mismatch for this target.
        for row in rv_variable_tidsync[["APOGEE_ID", "LOCATION_ID"]]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in single_visit_tidsync[["APOGEE_ID", "LOCATION_ID"]][gaia_rv_variable_tidsync]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in rv_variable_cool[["APOGEE_ID", "LOCATION_ID"]]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in single_visit_cool[["APOGEE_ID", "LOCATION_ID"]][gaia_rv_variable_cool]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        print("RV Nonvariable targets:")
        # 2M19351864+4214367 spectrum looks normal. There is a COLORTE_WARN flag.
        # 2M19330557+4619084 spectrum looks normal.
        # 2M19211671+4701074 spectrum looks normal.
        # 2M19273672+4940144 spectrum looks normal.
        # 2M19415749+4249137 spectrum looks normal. There is a COLORTE_WARN
        # flag.
        # 2M19235494+3834587 spectrum looks normal.
        # 2M18563342+4513481 spectrum looks normal. There is a COLORTE_WARN
        # flag.
        # 2M19384193+4551547 spectrum looks normal. There is a COLORTE_WARN
        # flag.
        for row in rv_nonvariable_tidsync[["APOGEE_ID", "LOCATION_ID"]]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in single_visit_tidsync[["APOGEE_ID", "LOCATION_ID"]][gaia_rv_nonvariable_tidsync]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in rv_nonvariable_cool[["APOGEE_ID", "LOCATION_ID"]]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
        for row in single_visit_cool[["APOGEE_ID", "LOCATION_ID"]][gaia_rv_nonvariable_cool]:
            print("APOGEE ID: {0:s}, LOCATION ID: {1:d}".format(
                row[0], row[1]))
    if berger_companions:
        print("RV Variable:")
        print(rv_variable_tidsync[["APOGEE_ID", "Ncomp"]])
        print(single_visit_tidsync[["APOGEE_ID", "Ncomp"]][gaia_rv_variable_tidsync])
        print(rv_variable_cool[["APOGEE_ID", "Ncomp"]])
        print(single_visit_cool[["APOGEE_ID", "Ncomp"]][gaia_rv_variable_cool])
        
        print("RV Nonvariable:")
        print(rv_nonvariable_tidsync[["APOGEE_ID", "Ncomp"]])
        print(single_visit_tidsync[["APOGEE_ID", "Ncomp"]][gaia_rv_nonvariable_tidsync])
        print(rv_nonvariable_cool[["APOGEE_ID", "Ncomp"]])
        print(single_visit_cool[["APOGEE_ID", "Ncomp"]][gaia_rv_nonvariable_cool])
    if vsini_check:
        





def plot_APOGEE_tidsync_targets():
    '''Plot the APOGEE tidally-synchronized targets on an HR diagram.'''
    splitter = cache.apogee_tidsync_targets()
    
    fullsample = splitter.subsample([])
    tidsync = splitter.subsample(["Fast McQuillan", "Tidsync"])
    rv_variable_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "RV Variable"])
    rv_nonvariable_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "RV Nonvariable"])
    single_visit_tidsync = splitter.subsample(
        ["Fast McQuillan", "Tidsync", "Single Visit"])
    rv_variable_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "RV Variable"])
    rv_nonvariable_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "RV Nonvariable"])
    single_visit_cool = splitter.subsample(
        ["Fast McQuillan", "Cool Rapid Dwarfs", "Single Visit"])

    single_visit_chi2_tidsync = (
        (single_visit_tidsync["VHELIO_AVG"] - 
         single_visit_tidsync["radial_velocity"])**2 /
    (0.5**2 + single_visit_tidsync["radial_velocity_error"]**2))
    gaia_rv_variable_tidsync = au.multi_logical_and(
        ~single_visit_chi2_tidsync.mask, single_visit_chi2_tidsync > 5.5)
    gaia_rv_nonvariable_tidsync = au.multi_logical_and(
        ~single_visit_chi2_tidsync.mask, single_visit_chi2_tidsync <= 5.5)
    gaia_ambiguous_tidsync = single_visit_chi2_tidsync.mask

    single_visit_chi2_cool = (
        (single_visit_cool["VHELIO_AVG"] - 
         single_visit_cool["radial_velocity"])**2 /
    (0.5**2 + single_visit_cool["radial_velocity_error"]**2))
    gaia_rv_variable_cool = au.multi_logical_and(
        ~single_visit_chi2_cool.mask, single_visit_chi2_cool > 5.5)
    gaia_rv_nonvariable_cool = au.multi_logical_and(
        ~single_visit_chi2_cool.mask, single_visit_chi2_cool <= 5.5)
    gaia_ambiguous_cool = single_visit_chi2_cool.mask

    f, ax = plt.subplots(1, 1, figsize=figsize)
    hr.absmag_teff_plot(
        fullsample["TEFF"], fullsample["M_K"], color="grey", marker=".", ls="",
        alpha=0.1, label="APOGEE", axis=ax)
    hr.absmag_teff_plot(
        rv_variable_tidsync["TEFF"], rv_variable_tidsync["M_K"], color="red", 
        marker="*", ls="", label="Rapid (RV Variable)", axis=ax)
    hr.absmag_teff_plot(
        single_visit_tidsync["TEFF"][gaia_rv_variable_tidsync], 
        single_visit_tidsync["M_K"][gaia_rv_variable_tidsync], color="red", 
        marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        rv_nonvariable_tidsync["TEFF"], rv_nonvariable_tidsync["M_K"], 
        color="blue", marker="*", ls="", label="Rapid (RV Nonvariable)", 
        axis=ax)
    hr.absmag_teff_plot(
        single_visit_tidsync["TEFF"][gaia_rv_nonvariable_tidsync], 
        single_visit_tidsync["M_K"][gaia_rv_nonvariable_tidsync], color="blue", 
        marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        single_visit_tidsync["TEFF"][gaia_ambiguous_tidsync], 
        single_visit_tidsync["M_K"][gaia_ambiguous_tidsync], color="black", 
        marker="*", ls="", label="Rapid (Single Epoch)", axis=ax)
    hr.absmag_teff_plot(
        rv_variable_cool["TEFF"], rv_variable_cool["M_K"], color="red", 
        marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        single_visit_cool["TEFF"][gaia_rv_variable_cool], 
        single_visit_cool["M_K"][gaia_rv_variable_cool], color="red", 
        marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        single_visit_cool["TEFF"][gaia_rv_nonvariable_cool], 
        single_visit_cool["M_K"][gaia_rv_nonvariable_cool], color="blue", 
        marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        rv_nonvariable_cool["TEFF"], rv_nonvariable_cool["M_K"], 
        color="blue", marker="*", ls="", label="", axis=ax)
    hr.absmag_teff_plot(
        single_visit_cool["TEFF"][gaia_ambiguous_cool], 
        single_visit_cool["M_K"][gaia_ambiguous_cool], color="black", 
        marker="*", ls="", label="", axis=ax)

    hr.absmag_teff_plot(
        [5600, 5600, 4850, 4850, 5600], [4.5, 2.5, 2.5, 4.5, 4.5],
        color=bc.purple, marker="", ls="--", label="MDM Selection region")

    solmetiso = mist.MISTIsochrone.isochrone_from_file(0.0)
    solmet_table = solmetiso.iso_table(5e8)
    young_table = solmetiso.iso_table(1.2e8)
    hr.absmag_teff_plot(
        10**solmet_table[solmetiso.logteff_col], 
        solmet_table[mist.band_translation["Ks"]], color=bc.pink,
        marker="", ls="-", label="MIST (500 Myr)", axis=ax, lw=2, zorder=5)
    hr.absmag_teff_plot(
        10**young_table[solmetiso.logteff_col], 
        young_table[mist.band_translation["Ks"]], color=bc.pink,
        marker="", ls="--", label="MIST (120 Myr)", axis=ax, lw=2, zorder=5)

    ax.legend()
    ax.set_title("RV Variability for 1 day < P < 3 day targets")
    ax.set_xlabel(Teffstr)
    ax.set_ylabel(MKstr)

def plot_APOGEE_Gaia_RV_differences():
    '''Plot the RV differences between APOGEE and Gaia RVs.

    This will be useful for using Gaia to distinguish between RV variable and
    nonvariable objects.
    '''
    splitter = cache.apogee_tidsync_targets()

    tidsync = splitter.subsample(["Tidsync"])
    cool_rapid_dwarfs = splitter.subsample(["Cool Rapid Dwarfs"])

    rvdiff_tidsync = tidsync["VHELIO_AVG"] - tidsync["radial_velocity"]
    rvdiff_cool = cool_rapid_dwarfs["VHELIO_AVG"] - cool_rapid_dwarfs["radial_velocity"]

    chisq_tidsync = rvdiff_tidsync ** 2 / (
        0.5**2 + tidsync["radial_velocity_error"]**2)
    chisq_cool = rvdiff_cool ** 2 / (
        0.5**2 + cool_rapid_dwarfs["radial_velocity_error"]**2)

    f, ax = plt.subplots(1, 1, figsize=figsize)
    ax.hist(chisq_cool, bins=400, density=True)

    chisq_array = np.linspace(0, 60, 200)
    chisq_vals = stats.chi2.pdf(chisq_array, 1)
    ax.plot(chisq_array, chisq_vals, 'k-')


