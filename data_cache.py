import string

import numpy as np
import numpy.core.defchararray as npstr
import astropy_util as au
from astropy.table import vstack, Table

import data_splitting as split
import sample_characterization as samp
import read_catalog as catin
import mist
import sed
import catalog
import baraffe
import aspcap_corrections as cors

@au.memoized
def mdm_targets():
    """A data splitter with the MDM targets."""
    a = split.MDMSplitter()
    split.initialize_mdm_sample(a)
    return a

@au.memoized
def apogee_tidsync_targets():
    """A data splitter to isolate APOGEE tidally-synchronized binaries."""
    apo = split.APOGEESplitter()
    combo = apo.join_with_McQuillan_periods()
    split.initialize_full_APOGEE(combo)
    split.initialize_RVvar_APOGEE(combo)
    clean = combo.split_subsample([
        "K Detection", "In Gaia", "APOGEE Valid Parameters"])
    clean.data["MIST K (sol)"] = samp.calc_model_mag_fixed_age_feh_alpha(
        clean.data["TEFF"], 0.0, "Ks", age=1e9, model="MIST v1.2")
    clean.data["MIST K Error"] = samp.calc_model_mag_err_fixed_age_feh_alpha(
        clean.data["TEFF"], 0.0, "Ks", teff_err=clean.data["TEFF_ERR"], age=1e9, 
        model="MIST v1.2")
    clean.data["K Excess"] = clean.data["M_K"] - clean.data["MIST K (sol)"] 
    clean.data["K Excess Error Down"] = np.sqrt(
        clean.data["M_K_err2"]**2 + clean.data["MIST K Error"]**2)
    clean.data["K Excess Error Up"] = np.sqrt(
        clean.data["M_K_err1"]**2 + clean.data["MIST K Error"]**2)
    clean.data["MIST R (APOGEE)"] = samp.calc_model_fixed_age_feh_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, 0.0, 1e9)
    apogee_logteff_err = (
        clean.data["TEFF_ERR"] / clean.data["TEFF"] / np.log(10))
    clean.data["MIST R Err (APOGEE)"] = samp.calc_model_err_fixed_age_feh_alpha(
        np.log10(clean.data["TEFF"]), mist.MISTIsochrone.logteff_col,
        mist.MISTIsochrone.radius_col, apogee_logteff_err, 0.0, age=1e9)
    return clean

