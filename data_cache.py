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
