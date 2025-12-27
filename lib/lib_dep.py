
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pyshtools as pysh
from pyshtools import constants
import numpy as np
import csv
from cartopy import crs as ccrs
import sys
from matplotlib.ticker import ScalarFormatter,FormatStrFormatter,MultipleLocator
from tqdm import tqdm
import time
import os
from typing import Literal
from scipy import integrate
from scipy.stats import norm
from scipy.optimize import curve_fit
import scipy.io
from PIL import Image
import random
import gc
import sklearn.metrics 
import tracemalloc
import subprocess
from matplotlib.patches import Patch
import skimage.metrics
import skimage.transform
import cv2
from scipy.ndimage import zoom
from scipy.interpolate import interp2d
from scipy.special import erf
import regex as re
import pickle
import cmath
import mpmath as mp

random.seed(42)
np.random.seed(42)

# SHTOOLs default figstyle settings
# pysh.utils.figstyle()

np.set_printoptions(formatter={'float_kind':'{:e}'.format}, precision=3)

# Enables LaTeX plot globally
plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "serif",
})
plt.rcParams['font.size'] = '12'


cmap = 'turbo'   # WARNING: not colorblind map   'jet','turbo'


G_const = 6.6743e-11   # m^3/(Kg*s^2)



##########################################################################################################################
##########################################################################################################################


# Utilities



from cycler import cycler

# Define MATLAB default colors in Hex (7 colors)
matlab_colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
plt.rcParams['axes.prop_cycle'] = cycler(color=matlab_colors)



# -----------------------------------------------------------------------------------------------------------------------

# Set precision for mpmath calculations

bits = 64 * 4               # same as Julia: 256 bits
mp.mp.prec = bits

digits = np.ceil(bits * np.log10(2))  # convert bits -> decimal digits
mp.mp.dps = digits         # set decimal precision (mp.dps ~ number of decimal digits)
