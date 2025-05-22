import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pyshtools as pysh
from pyshtools import constants
import numpy as np
import csv
from cartopy import crs as ccrs
import sys
from matplotlib.ticker import ScalarFormatter
import cartopy.crs as ccrs
from tqdm import tqdm
import time
import os
from typing import Literal
from scipy import integrate
from scipy.stats import norm
from scipy.optimize import curve_fit
import scipy.io
from Planets_ConfigFiles import *
from PIL import Image
import random
import gc
import sklearn.metrics 
import tracemalloc
import subprocess
from matplotlib.patches import Patch
import skimage.metrics
import skimage.transform
import numpy as np
import cv2
from scipy.ndimage import zoom
from scipy.interpolate import interp2d


random.seed(41)


# SHTOOLs default figstyle settings
# pysh.utils.figstyle()


# Enables LaTeX plot globally
plt.rcParams.update({
    "text.usetex": True,  
    "font.family": "serif",
})

cmap = 'jet'   # WARNING: not colorblind map


G_const = 6.6743e-11   # m^3/(Kg*s^2)





##########################################################################################################################
##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

# Utilities



from cycler import cycler

# Define MATLAB default colors in Hex (7 colors)
matlab_colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"]
plt.rcParams['axes.prop_cycle'] = cycler(color=matlab_colors)





# ------------------------------------------------------------------------------------------------------------------------
##########################################################################################################################
##########################################################################################################################
# ------------------------------------------------------------------------------------------------------------------------

# Functions





def DataReader(body: Literal["Mercury","Venus","Earth","Moon","Ganymede"], n_max, n_layers=None):

    """
    Usage
    ----------
    Reader for spherical harmonics coefficients (SHGravCoeffs/SHCoeffs) for gravity and topography data.
    see Planets_ConfigFiles.py for further details.


    Parameters
    ----------
    body            : str, option ["Mercury", "Earth", "Venus", "Moon"]
                      Planetary body implemented (see Planets_ConfigFiles.py)
    n_max           : int
                      The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    n_layers        : int, default = None
                      Number of internal layers (selecting the correspondent implemented interiors model).

    Output
    ----------
    param_bulk      : array,
                      Array containing bulk constants:
                        ref_radius      [km]
                        GM_const        [m^3/sec^2]
                        errGM_const     
                        ref_mass        [kg]
                        ref_rho         [kg/m^3]
                        ref_ang_vel     [rad/sec]
                        ref_MoI         (I/MR^2)   
                        err_MoI         
                        r_e_fact        
                        r_p_fact            
    param_body      : array,
                      Array containing body file data:
                        grav_file           [str] 
                        header_opt_grav     [bool]        
                        format_grav         (shtools, bshc)
                        top_file            [str]
                        topo_factor         (transforming to km)
                        header_opt_top      [bool] 
                        format_topo         (shtools, bshc)
                        rho_boug            [kg/m^3]
                        n_half              (cutting degree)
    param_int      : array,
                      Array containing interior parameters (as function of n_layers):
                        rho_layers          [kg/m^3]
                        radius_layers       [km]
                        interface_type      [string]
    coeffs_grav     : pyshtools.SHGravCoeffs
                      Gravitational coefficients
    coeffs_topo     : pyshtools.SHGravCoeffs, [km]
                      Topography coefficients
    """




# Read configuration file
    match body:

            case "Mercury":
                param_bulk = Mercury_ConfigFile.bulk()
                param_body = Mercury_ConfigFile.data()
                if n_layers is not None: param_int  = Mercury_ConfigFile.interiors(n_layers)
                    
            case "Venus":
                param_bulk = Venus_ConfigFile.bulk()
                param_body = Venus_ConfigFile.data()
                if n_layers is not None: param_int  = Venus_ConfigFile.interiors(n_layers)
                
            case "Earth":
                param_bulk = Earth_ConfigFile.bulk()
                param_body = Earth_ConfigFile.data()
                if n_layers is not None: param_int  = Earth_ConfigFile.interiors(n_layers)

            case "Moon":
                param_bulk = Moon_ConfigFile.bulk()
                param_body = Moon_ConfigFile.data()
                if n_layers is not None: param_int  = Moon_ConfigFile.interiors(n_layers)

            case "Ganymede":
                param_bulk = Ganymede_ConfigFile.bulk()
                param_body = Ganymede_ConfigFile.data()
                if n_layers is not None: param_int  = Ganymede_ConfigFile.interiors(n_layers)

            case _:
                print("Invalid body name")
                sys.exit()



    # Extracting parameters
    grav_file       = param_body[0]
    topo_file       = param_body[1]
    topo_factor     = param_body[2]
    header_opt_grav = param_body[3]
    format_grav     = param_body[4]
    header_opt_topo = param_body[5]
    format_topo     = param_body[6]

    ref_radius      = param_bulk[0]
    GM_const        = param_bulk[1]



    # Gravity data
    print('Gravity datafile:')

    if grav_file is not None:
        print(grav_file + '\n')
        if header_opt_grav:
            coeffs_grav = pysh.SHGravCoeffs.from_file(grav_file, format=format_grav, lmax=n_max, header=header_opt_grav)
        else:
            coeffs_grav = pysh.SHGravCoeffs.from_file(grav_file, format=format_grav, lmax=n_max, header=header_opt_grav,
                                                    r0=ref_radius*1e+3, gm=GM_const)
        coeffs_grav.name = grav_file.split('/')[-1].split('.')[0]
    else:
        print('No gravitational file (zero data) \n')
        coeffs_grav = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=GM_const, r0=ref_radius*1e+3)
        coeffs_grav.name = 'No Data'

    coeffs_grav.gm = GM_const                   # m^3/s^2
    coeffs_grav.r0 = ref_radius*1e+3            # m




    # Topography data (+ conversion into [km])
    print('Topography datafile: \n')

    if topo_file is not None:
        print(topo_file + '\n')
        coeffs_topo = pysh.SHCoeffs.from_file(topo_file, format=format_topo, lmax=n_max, units = 'km',header=header_opt_topo)
        if topo_factor != 1:
            coeffs_topo /= topo_factor
            if body == "Earth": coeffs_topo += ref_radius 
    else:
        print('Generating RANDOM surface topography coefficients: \n')
        degrees = np.arange(n_max+1, dtype=float)
        degrees[0] = np.inf
        coeffs_topo = pysh.SHCoeffs.from_random(degrees**(-2), seed=42*n_layers)
        coeffs_topo.set_coeffs(ref_radius,0,0)
        surf = coeffs_topo.expand(lmax=n_max,extend=True)
        x = float(input("Insert the DeltaH (max - min) for topography: "))
        a = x/(np.max(surf.data) - np.min(surf.data))
        coeffs_topo.coeffs *= a
        coeffs_topo.set_coeffs(ref_radius,0,0)




    if n_layers is not None: 
        return param_bulk,param_body,param_int, coeffs_grav, coeffs_topo
    else:
        return param_bulk,param_body, coeffs_grav, coeffs_topo

##########################################################################################################################
##########################################################################################################################



def Global_Analysis(coeffs_grav, coeffs_topo, n_max, r, rho_boug, i_max, saving_dir=None,plot_opt: Literal[None,'single','multiple'] = None, load_opt=False,n_min=3, proj_opt=ccrs.Mollweide(),verbose_opt=False):


    """
    Usage
    ----------
    Evaluate gravity and topography coefficients (SHGravCoeffs/SHCoeffs) in spherical harmonics and produce values 
    and maps for:
    - gravity potential U   [m^2/s^2]
    - Topography H          [km]
    - Free-Air anomalies    [mGal]
    - Bouguer anomalies     [mGal]

    Maps are in .pdf files
    Data are in .dat files

    Parameters
    ----------
    coeffs_grav         : pyshtools.SHGravCoeffs
                          Gravitational coefficients
    coeffs_topo         : pyshtools.SHGravCoeffs, [km]
                          Topography coefficients
    r                   : float, [km]
                          Radius where the maps are to be evaluated.
    n_max               : int
                          The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    rho_boug            : float, [kg/m^3]
                          Crust density, used both in the Bouguer correction  
    i_max               : int
                          The maximum order used in the Taylor-series expansion when calculating the potential coefficients (Bouguer correction).
    saving_dir          : str, default = None
                          Saving directory for the output files (data and images).
                          If None, no files are saved.
    plot_opt            : str, options = [None,'single','multiple'], default = 'multiple
                          None = no plotting
                          single = each product has its own map
                          multiple = all of the products are plotted in a single map together
    load_opt            : bool, default = False
                          If True, load previously calculated matrices
                          If False, evaluate the products and save/overwrite them
    n_min               : int, default=3
                          The minimum spherical harmonic degree of the output spherical harmonic coefficients (GRAVITY).
    proj_opt            : cartopy.crs, default = ccrs.Mollweide()
                          Map projection for plots 
    verbose_opt         : bool, default = False
                          Verbose option for progress and displaying sections


    Output
    ----------
    U_matrix        : pyshtools.SHGrid
                      Gravitational Potential matrix SHGrid
    topog_matrix    : pyshtools.SHGrid
                      Topography matrix SHGrid
    deltag_freeair  : pyshtools.SHGrid
                      Free-Air anomalies matrix SHGrid
    deltag_boug     : pyshtools.SHGrid
                      Bouguer anomalies matrix SHGrid
          
    """

    # Global analysis:

    if verbose_opt:
        print("# ------------------------------------------------------------------------------------------------------\n")
        print("\n")
        print("Global analysis:")
        print("\n")

    # ------------------------------------------------------------------------------------------------------


    # Compute Gravity (U - FreeAir)
    grav = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,r=r,extend=True)
    if n_min>=0: grav_min = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,lmax_calc=n_min,r=r,extend=True)


    # ------------------------------------------------------------------------------------------------------


    # Gravitational Potential  (m^2/s^2)
    if verbose_opt: print("Gravitational Potential:")

    if load_opt and os.path.isfile(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        U_matrix = np.loadtxt(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        U_matrix = pysh.SHGrid.from_array(U_matrix)
    else:
        if n_min>=0: 
            U_matrix = grav.pot - grav_min.pot
        else: U_matrix = grav.pot
        
        if saving_dir is not None: U_matrix.to_file(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")

    if plot_opt=='single':
        fig, ax = U_matrix.plot(colorbar='right',projection=proj_opt, cb_label='$m^2/s^2$')
        if saving_dir is not None: fig.savefig(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".pdf", dpi=600)

    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Topography (km)
    if verbose_opt: print("Topography:")

    if load_opt and os.path.isfile(saving_dir+"/topog_matrix.dat") is True:
        topog_matrix = np.loadtxt(saving_dir+"/topog_matrix.dat")
        topog_matrix = pysh.SHGrid.from_array(topog_matrix)
    else:
        topog_matrix = coeffs_topo.expand(lmax=n_max,extend=True)
        if n_min>=0:
            topog_matrix_min = coeffs_topo.expand(lmax=n_max,lmax_calc=0,extend=True)
            topog_matrix = topog_matrix-topog_matrix_min

        theta = topog_matrix.lats()
        phi = topog_matrix.lons()
        if saving_dir is not None:
            np.savetxt(saving_dir+"/theta.dat",theta)
            np.savetxt(saving_dir+"/phi.dat",phi)
            topog_matrix.to_file(saving_dir+"/topog_matrix.dat")

    if plot_opt=='single':
        fig, ax = topog_matrix.plot(colorbar='right',projection=proj_opt, cb_label='km')
        if saving_dir is not None: fig.savefig(saving_dir+"/topog_matrix.pdf", dpi=600)

    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Gravitational Accelleration (Free-Air anomalies)  (mGal)
    if verbose_opt: print("Free-Air anomalies:")


    if load_opt and os.path.isfile(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        deltag_freeair = np.loadtxt(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        deltag_freeair = pysh.SHGrid.from_array(deltag_freeair)
    else:
        if n_min>=0: 
            deltag_freeair = grav.total*1e+5  - grav_min.total*1e+5  
        else: deltag_freeair = grav.total*1e+5
         
        if saving_dir is not None: deltag_freeair.to_file(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")

    if plot_opt=='single':
        fig, ax = deltag_freeair.plot(colorbar='right',projection=proj_opt, cb_label='mGal')
        if saving_dir is not None: fig.savefig(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".pdf", dpi=600)


    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Bouguer anomalies
    if verbose_opt: print("Bouguer anomalies:")

    if load_opt and os.path.isfile(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        deltag_boug = np.loadtxt(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        deltag_boug = pysh.SHGrid.from_array(deltag_boug)
    else:
        # NB: topography coeff in m
        bouger_correction = pysh.SHGravCoeffs.from_shape(shape=coeffs_topo*1e+3, rho=rho_boug, gm=coeffs_grav.gm, lmax=n_max, nmax=i_max) 
        bouger_correction = bouger_correction.change_ref(r0=coeffs_grav.r0)

        bouguer_coeff = coeffs_grav - bouger_correction
        bouguer = bouguer_coeff.expand(lmax=n_max,r=r,extend=True)
        if n_min>=0:
            bouguer_min = bouguer_coeff.expand(lmax=n_max,lmax_calc=n_min,r=r,extend=True)
            deltag_boug = bouguer.total*1e+5 - bouguer_min.total*1e+5
        else:
            deltag_boug = bouguer.total*1e+5
        if saving_dir is not None: deltag_boug.to_file(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        

    if plot_opt=='single':
        fig, ax = deltag_boug.plot(colorbar='right', projection=proj_opt, cb_label='mGal')
        if saving_dir is not None: fig.savefig(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".pdf", dpi=600)


    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------

    # Multiple plot

    if plot_opt=='multiple':

        if verbose_opt: 
            print(" ")
            print("Plotting:")


        fig, axs = plt.subplots(2, 2, figsize=(13, 7))

        U_matrix.plot(ax=axs[0, 0], colorbar='right',projection=proj_opt, title='Gravitational Potential', cb_label='$m^2/s^2$',cmap=cmap)
        topog_matrix.plot(ax=axs[0, 1], colorbar='right',projection=proj_opt, title='Topography', cb_label='km',cmap=cmap)
        deltag_freeair.plot(ax=axs[1, 0], colorbar='right',projection=proj_opt, title='Free-Air anomalies', cb_label='mGal',cmap=cmap)
        deltag_boug.plot(ax=axs[1, 1], colorbar='right', projection=proj_opt, title='Bouguer anomalies', cb_label='mGal',cmap=cmap)

        plt.tight_layout()
        plt.show()
        if saving_dir is not None: fig.savefig(saving_dir+"/U_h_FreeAir_Boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".pdf", dpi=600)

        if verbose_opt:
            print("Done")
            print(" ")
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")


    return U_matrix, topog_matrix, deltag_freeair, deltag_boug

##########################################################################################################################
##########################################################################################################################



def Spectrum(coeffs,n_max,saving_dir=None,save_opt: Literal['all','total',None] = None,n_min=2, load_opt=False,plot_opt=False,verbose_opt=False):

    """
    Usage
    ----------
    Evaluate spectrum of spherical harmonics coefficients (SHGravCoeffs/SHCoeffs)


    Parameters
    ----------
    coeffs          : pyshtools.SHGravCoeffs
                      Gravitational coefficients (single set or list)
                      Note: for list, remember to set the name (coeff.name) for each element to visualize correctly the legend
    n_max           : int
                      The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    n_min           : int, default = 2
                      The minimum spherical harmonic degree of the output spherical harmonic coefficients.
    saving_dir      : str,
                      Saving directory for the output files (data and images)
    save_opt        : str, option ['all','total',None]
                    - None, no saving
                    - 'all', save each layers spectrum
                    - 'total', save just the global spectrum
    load_opt        : bool, default = False
                      If True, load previously calculated spectrum from files
                      IF False, evaluated the spectrum ex novo
    plot_opt        : bool, default = False
                      Plotting spectrum
    verbose_opt     : bool, default = False
                      Verbose option for progress and displaying sections


    Output
    ---------
    spectrum        : list (numpy.ndarray)
                      List of numpy arrays containing the spectrum An = sqrt((Cnm^2 + Snm^2)/(2n+1)) for all of the layers
                      and the total one. Spherical layers have zero effect.
    """


    # Gravitational Spectrum
    if verbose_opt: print("Gravitational Spectrum:")


    if plot_opt:
        plt.figure(figsize=(9,4), constrained_layout=True)

    degree_grav = np.arange(0,n_max+1)

    spectrum=[]
    for coeff in coeffs:

        if load_opt and os.path.isfile(saving_dir+"/spectrum_grav"+coeff.name+".dat") is True:
            spectrum_grav = np.loadtxt(saving_dir+"/spectrum_grav"+coeff.name+".dat")
            if plot_opt:
                plt.plot(degree_grav[n_min:],np.sqrt(spectrum_grav[n_min:]), linewidth=2, label=coeff.name)

        else:
            if coeff.coeffs[0][3,0] != 0:   # some topopgrahy (at least polar flattening)
                # WARNING: use or convention='power',unit='per_lm' or convention='l2norm',unit='per_l' + divide by (2*degree+1)
                spectrum_grav = pysh.spectralanalysis.spectrum(coeff.coeffs,convention='l2norm',unit='per_l',lmax=n_max)
                spectrum_grav /= (2*degree_grav+1) # only for 'l2norm'
                if save_opt == 'all': np.savetxt(saving_dir+"/spectrum_grav_"+coeff.name+".dat",spectrum_grav)
                if save_opt == 'total' and ("Layer" in coeff.name) is False: np.savetxt(saving_dir+"/spectrum_grav_"+coeff.name+".dat",spectrum_grav)
                if plot_opt:
                    plt.plot(degree_grav[n_min:],np.sqrt(spectrum_grav[n_min:]), linewidth=2, label=coeff.name)

        spectrum.append(np.sqrt(spectrum_grav))

    if plot_opt:
        # plt.ylim([10**-np.ceil(-np.log10(ymin)), 10**-np.ceil(-np.log10(ymax)-1)])
        # plt.ylim([10**-9, 10**-4])
        plt.xlim([0, n_max])
        plt.yscale("log")
        plt.xlabel("Degree $n$")
        plt.ylabel("Power Spectrum")
        plt.xticks(np.arange(0,n_max+10 ,10))
        plt.legend()
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
        plt.grid(visible=True, which='minor', linestyle='--', linewidth=0.2)
        plt.show()
        if save_opt is not None: plt.savefig(saving_dir+"/spectrum_grav.pdf", dpi=600, bbox_inches='tight')


    if verbose_opt:
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")


    return spectrum



##########################################################################################################################
##########################################################################################################################



def CrustThickness(coeffs_topo,coeffs_grav,rho_boug,delta_rho,mean_radius,n_max,i_max, filter_opt,filter_deg=None,verbose_opt=False):


    """
    Usage
    ----------
    Calculate the thickness of the crust based on Bouguer Anomaly (see Wieczorek and Phillips, 1998).
    This function uses the pyshtools library BAtoHilmDH method (https://shtools.github.io/SHTOOLS/pybatohilmdh.html).

    
    Parameters
    -----------

    coeffs_grav     : pyshtools.SHGravCoeffs 
                      Gravitational coefficients
    coeffs_topo     : pyshtools.SHGravCoeffs, [km]
                      Topography coefficients
    rho_boug        : float, [kg/m^3]
                      Crust density; used both in the Bouguer correction and the crust thickness calculation 
    delta_rho       : float, [kg/m^3]
                      Density contrast between Mantle and Crust layers
    mean_radius     : float, [km]
                      Mean radius of the mantle-crust interface (=mean crustal thickness) 
    n_max           : int
                      The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    i_max           : int
                      The maximum order used in the Taylor-series expansion when calculating the potential
                      coefficients (Bouguer correction and Crustal thickness).
    filter_opt      : int, option=[0,1,2]
                      Type of filter to apply in the conversion: if 0, no filtering is applied.
                                                                 if 1, DownContFilterMA (minimum amplitude filter)  eq. 19, Wieczorek and Phillips, 1998
                                                                 if 2, DownContFilterMC (minimum curvature filter)
    verbose_opt     : bool, default = False
                      Verbose option for progress and displaying sections

    Output:
    --------
    mantle_crust_relief : The calculated relief of the mantle-crust interface (ie. crust thickness), as SHGrid.

    """

    # Crustal thickness calculation
    if verbose_opt: 
        print("Crust Thickness:")
        print('     Delta_rho: ' + str(delta_rho))
    

    # Bouguer coefficient evaluation
    bouger_correction = pysh.SHGravCoeffs.from_shape(shape=coeffs_topo*1e+3, rho=rho_boug, gm=coeffs_grav.gm, lmax=n_max, nmax=i_max)   
    bouger_correction = bouger_correction.change_ref(r0=coeffs_grav.r0)
    coeff_boug = coeffs_grav - bouger_correction

    # Mean interface grid
    mean_interface = pysh.SHGrid.from_zeros(lmax=n_max)+mean_radius*1e+3


    # Mantle - Crust interface evaluation
    if filter_deg is None:
        filter_deg = n_max/2
    if verbose_opt: print('     Cutting filter (n_half): '+ str(filter_deg))

    mantle_crust_coeffs = pysh.gravmag.BAtoHilmDH(ba=coeff_boug.coeffs,griddh=mean_interface.data,nmax=i_max,lmax=n_max,mass=coeffs_grav.gm/G_const,r0=coeff_boug.r0,rho=delta_rho,filter_type=filter_opt,filter_deg=filter_deg)
    mantle_crust_relief= pysh.SHGrid.from_array(pysh.expand.MakeGridDH(mantle_crust_coeffs/1e+3,lmax=n_max,sampling=2,extend=True))


    return mantle_crust_relief



##########################################################################################################################
##########################################################################################################################



def SynthGen(param_bulk,param_int,n_max,coeffs_grav,coeffs_topo,i_max,filter_deg,saving_dir,save_opt: Literal['all','total',None],
             mode: Literal['layer','interface'] ='layer',load_opt=False,plot_opt=False,proj_opt=ccrs.Mollweide(), verbose_opt=False):

    """
    Usage
    ----------
    Synthetic generation of spherical harmonics coefficients (SHGravCoeffs/SHCoeffs) from an interiors model
    This function generates synthetic spherical harmonics coefficients based on a given interior model 
    of a planetary body. It calculates the gravitational coefficients for each layer of the model and
    combines them to produce the total gravitational field. 


    Parameters
    -----------
    param_bulk   : list
                  A list containing body bulk parameters, namely:
                    - r_e_fact,r_p_fact: equatorial radius factor
    param_int   : list
                  A list containing interior model parameters:
                  [0] - rho_layers: list of average densities for each layer    [kg/m^3]
                  [1] - radius_layers: list of radii for each layer interface   [km]
                  [2] - interface_type: list of interface types for each layer  [string]
    n_max       : int
                  Maximum degree of spherical harmonics expansion
    coeffs_grav : pyshtools.SHGravCoeffs
                  Input gravitational coefficients
    coeffs_topo : list, pyshtools.SHCoeffs, [km]
                  Input topography coefficients for each layer interfrace. Used just for 'surface' interface
    i_max       : int
                  Maximum order used in Taylor series expansion for potential coefficients
    filter_deg  : int
                  Degree of filter to apply in downward continuation filter (n_half)
    saving_dir  : str
                  Saving directory for the output files (data and images)
    save_opt    : str, option ['all','total',None]
                  - None, no saving
                  - 'all', save each layers gravitational and interface topography coefficients (+ interiors parameters)
                  - 'total', save just the global gravitational coefficients (+ interiors parameters)
    mode        : str, option ['layer','interface']
                  Generation mode:
                    - 'layer': generate coefficients related to each layer of the model (evaluation from bottom to top interfaces)
                    - 'interface': generate coefficients realted to each interface of the model (evaluation for the density jump on the interface)
    load_opt    : bool, default = False
                  If True, load previously calculated coefficients from files
                  IF False, generate new coefficients
    plot_opt    : bool, default = False
                  Plotting Interface - U 
    proj_opt    : cartopy.crs, default = ccrs.Mollweide()
                  Map projection for plots
    verbose_opt : bool, default = False
                  Verbose option for progress and displaying sections

                  
    Returns
    --------
    coeffs_tot      : pyshtools.SHGravCoeffs
                      Total synthetic gravitational coefficients
    coeffs_list     : list
                      List of pyshtools.SHGravCoeffs objects for each layer
    """



    # Unpacking Interiors parameters
    rho_layers      = param_int[0]
    radius_layers   = param_int[1]
    interface_type  = param_int[2]
    n_layers= np.size(rho_layers)

    # Flattening parameters
    r_e_fact        = param_bulk[8]
    r_p_fact        = param_bulk[9]

    ref_rho        = param_bulk[4]



    if not load_opt or os.path.isfile(saving_dir+"/coeffs_tot.dat") is False:

        surf_prec = pysh.SHGrid.from_zeros(lmax=n_max)
        theta = surf_prec.lats()* np.pi / 180.
        phi = surf_prec.lons()* np.pi / 180.
        delta_theta = -np.diff(theta)[0]
        delta_phi = np.diff(phi)[0]
        cos_theta = np.reshape(np.repeat(np.cos(theta),np.size(phi),axis=0),np.shape(surf_prec.data))

        M_layer=[]
        coeffs_list=[]
        coeffs_tot = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=coeffs_grav.gm, r0=coeffs_grav.r0)

        if plot_opt:
            fig, axs = plt.subplots(n_layers, 2, figsize =(8,8))

        for i,interface in enumerate(interface_type):

            if verbose_opt:
                print("Generating layer: " + str(i+1) + '\n')
                print("Interface type: " + interface + '')
                print("Layer's radius: " + str(radius_layers[i]) + " km")
                print("Layer's density: " + str(rho_layers[i]) + " kg/m^3")

        
            # Evalute interface topography

            match interface:

                case 'sph':
                    surf = pysh.SHGrid.from_zeros(lmax=n_max)+radius_layers[i]

                case 'dwnbg':
                    surf = CrustThickness(coeffs_topo=coeffs_topo,coeffs_grav=coeffs_grav,rho_boug=rho_layers[i],
                                          delta_rho=rho_layers[i]-rho_layers[i+1],mean_radius=radius_layers[i],
                                          n_max=n_max,i_max=i_max,filter_opt=1,filter_deg=filter_deg[i],
                                          verbose_opt=verbose_opt)

                case 'surf':
                    surf = coeffs_topo.expand(lmax=n_max,extend=True)

                case 'sphflat':
                    r_p = radius_layers[i]*r_p_fact
                    r_e = radius_layers[i]*r_e_fact
                    theta_grid = np.reshape(np.repeat(theta,np.size(phi),axis=0),np.shape(surf_prec.data))
                    surf = pysh.SHGrid.from_array(np.sqrt(r_p*r_p + (r_e*r_e -r_p*r_p)*np.cos(theta_grid)*np.cos(theta_grid)))
                    if verbose_opt:
                        print("Polar flattenig:")
                        print("     flattening f: " + str(np.round(1-r_p_fact/r_e_fact,4)))
                        print("     polar radius R_p: " + str(np.round(r_p,2)))
                        print("     equatorial radius R_e: " + str(np.round(r_e,2)))

                case 'rng':
                    # Kaula's rule + lowpass filtering (Ri/R ^n)
                    degrees = np.arange(n_max+1, dtype=float)
                    degrees[0] = np.inf
                    surf_coeff_rng = pysh.SHCoeffs.from_random(degrees**(-2) * (radius_layers[i]/radius_layers[-1])**(degrees),power_unit='per_lm', seed=42*i)
                    surf = surf_coeff_rng.expand(lmax=n_max,extend=True)+radius_layers[i]
                    x = float(input("Insert the DeltaH (max - min) for interface topography: "))
                    a = x/(np.max(surf.data) - np.min(surf.data))
                    surf_coeff_rng.coeffs *= a
                    surf = surf_coeff_rng.expand(lmax=n_max,extend=True)+radius_layers[i]

                case _:
                    print("ERROR: No interface type recognized")
                    sys.exit()



            # ------------------------------------------------------------------------------------------------------

            # Check for topography conflicts (between current layer and the previous one):

            if ((surf.data - surf_prec.data)< 0).any():
                print("ERROR: Interface Conflict")
                print("Layer "+str(i+1)+"-th lower than "+str(i)+"-th\n")

                if plot_opt:
                    topog_conflict = pysh.SHGrid.from_array((surf.data-surf_prec.data) < 0)
                    topog_conflict.plot(projection=proj_opt, title='Interface conflict Layer '+str(i+1)+'-th - '+str(i)+'-th', colorbar='right')
                    surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title='Interface '+str(i+1), cb_label='$km$',cmap=cmap)
                return None,None           


            # ------------------------------------------------------------------------------------------------------

            # Layer's mass:
            M_layer.append((1e+9*(surf.data**3 - surf_prec.data**3)  * rho_layers[i]/3 * cos_theta * delta_theta * delta_phi).sum())
            if verbose_opt: print("Layer's mass: " + str(M_layer[-1]) + " kg\n")

            # ------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------
            

            # Synthetic gravitational coefficients generation:


            # ------------------------------------------------------------------------------------------------------
            # Layer generation:

            if mode == 'layer':

                # inizializing layer coefficients to zero
                coeffs_i = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i]*1e+3, gm=G_const*M_layer[i])


                # First layer
                if i == 0:
                    if interface_type[i] != 'sph':
                        coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                    lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                # Other layers
                else:
                    # Bottom interface
                    if interface_type[i-1] == 'sph':
                        coeffs_bot = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i-1]*1e+3, gm=coeffs_grav.gm)
                    else:
                        coeffs_bot = pysh.SHGravCoeffs.from_shape(shape=surf_prec*1e+3, rho=rho_layers[i],
                                                                lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                    # Top interface
                    if interface_type[i] == 'sph':
                        coeffs_top = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i]*1e+3, gm=coeffs_grav.gm)
                    else:
                        coeffs_top = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                    coeffs_i.coeffs = coeffs_top.coeffs - coeffs_bot.coeffs
                
                coeffs_i.name = "Layer "+str(i+1)


            # ------------------------------------------------------------------------------------------------------
            # Interface generation   
            else: 

                if interface == 'surf':
                    coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                        lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)
                elif interface == 'sph':
                    coeffs_i = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=coeffs_grav.gm, r0=radius_layers[i]*1e+3)                                                              
                else:
                    coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i]-rho_layers[i+1],
                                                                        lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                coeffs_i.name = "Interface "+str(i+1)
            

            # ------------------------------------------------------------------------------------------------------

            # Save coefficients (gravity and topography)
            if save_opt == 'all':
                if mode=='layer':coeffs_i.to_file(saving_dir+"/coeffs_layer"+str(i+1)+".dat")
                if mode=='interface':coeffs_i.to_file(saving_dir+"/coeffs_interface"+str(i+1)+".dat")
                surf.to_file(saving_dir+"/interface"+str(i+1)+".dat")



            # ------------------------------------------------------------------------------------------------------
            if plot_opt:
                U_layer_matrix = coeffs_i.expand(lmax=n_max,r=coeffs_grav.r0,extend=True)
                U_layer_matrix_min = coeffs_i.expand(lmax=n_max,lmax_calc=1,r=coeffs_grav.r0,extend=True)
                U_layer = U_layer_matrix.pot - U_layer_matrix_min.pot

                U_layer.plot(ax=axs[i,1], colorbar='right',projection=proj_opt, title=coeffs_i.name, cb_label='$m^2/s^2$',cmap=cmap)
                surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title='Interface '+str(i+1), cb_label='$km$',cmap=cmap)
                plt.tight_layout()
                plt.show()
                if save_opt == 'all':
                    fig.savefig(saving_dir+"/Interface - U.pdf", dpi=600)

            # ------------------------------------------------------------------------------------------------------

            # Summing the coefficients:
            coeffs_list.append(coeffs_i)
            coeffs_tot.coeffs += coeffs_i.coeffs

            surf_prec = surf
            if verbose_opt: print("\n")
            gc.collect()
            FreeMemory()


            surf_prec = surf
            if verbose_opt: print("\n")

        coeffs_tot.name = 'Synthetic'
        if save_opt=='total' or save_opt=='all': coeffs_tot.to_file(saving_dir+"/coeffs_tot.dat")

        if verbose_opt:
            print("Synthetic Generation done")
            print("Total's mass: " + str(np.array(M_layer).sum()) + " kg\n")
            print("Synthetic coefficients saved in: " + saving_dir + '\n')
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")


    else:

        print("Loading Synthetic coefficients...\n")

        # Loading coefficients:
        coeffs_list=[]

        if plot_opt:
            fig, axs = plt.subplots(n_layers, 2, figsize =(8,8))

        for i in range(0,n_layers):
            if mode=='layer': coeffs_i =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_layer"+str(i+1)+".dat", format="shtools", lmax=n_max, header=True, name="Layer "+str(i+1))
            if mode=='interface': coeffs_i =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_interface"+str(i+1)+".dat", format="shtools", lmax=n_max, header=True, name="Interface "+str(i+1))
            coeffs_list.append(coeffs_i)

            surf =  pysh.SHGrid.from_file(saving_dir+"/interface"+str(i+1)+".dat")

            U_i_matrix = coeffs_i.expand(lmax=n_max,r=coeffs_grav.r0,extend=True)
            U_i_matrix_min = coeffs_i.expand(lmax=n_max,lmax_calc=1,r=coeffs_grav.r0,extend=True)
            U_i = U_i_matrix.pot - U_i_matrix_min.pot

            if plot_opt:
                U_i.plot(ax=axs[i,1], colorbar='right',projection=proj_opt, title=coeffs_i.name, cb_label='$m^2/s^2$',cmap=cmap)
                surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title='Interface '+str(i+1), cb_label='$km$',cmap=cmap)
                plt.tight_layout()
                plt.show()

        coeffs_tot =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_tot.dat", format="shtools", lmax=n_max, header=True, name='Synthetic')
        if verbose_opt:
            print("Synthetic Generation Loaded")
            sub_dir = saving_dir.split("/")[-1]
            print("Synthetic coefficients from: " + sub_dir + '\n')
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")





    return coeffs_tot,coeffs_list



##########################################################################################################################
##########################################################################################################################

def InputRange(n_layers,param_int,n_half):

    """
    Usage
    ----------
    Input range for the parameters grid:
    - n_counts = number of simulations
    - ranges for densty, radius and (eventually) cutting degree n_half
    Option for default values: rho and radius +/- 100, n_half= 5-50


    Parameters
    ----------
    n_layers        : int,
                      number of internal layers
    param_int       : array,
                      Array containing interior parameters (as function of n_layers):
                        rho_layers          [kg/m^3]
                        radius_layers       [km]
                        interface_type      [string]    
    n_half          : int,
                      Degree of filter to apply in downward continuation filter (n_half)


    Output
    ----------
    n_counts                                : int,
                                              number of simulations for the grid
    rho_range, radius_range, nhalf_range    : tuple [2,n_layers]
                                              Tuples with the min and max values of the input range for each layer                    
    """

    n_counts = int(input("Insert number of simulations you want to evaluate: "))


    rho_layers      = param_int[0]
    radius_layers   = param_int[1]
    interface_type  = param_int[2]

    rho_range = np.zeros((n_layers,2))
    radius_range = np.zeros((n_layers,2))
    nhalf_range = np.zeros((n_layers,2))

    default_range=200

    default_opt = input("Use default ranges? (rho=+/-"+str(default_range)+" [kg/m^3], radius=+/-"+str(default_range)+" [km], n_half=3-100)")


# NB: - starting from 1 because the innermost layer is not part of the grid (M and MoI conservation)
#     - last layer not radius range (= surface)


    if default_opt=="":
        for i in range(1,n_layers):
            rho_range[i]  = [rho_layers[i]-default_range, rho_layers[i]+default_range]
            if i == n_layers-1:
                radius_range[i]  = [radius_layers[i]-default_range, radius_layers[i]]
            else:
                if radius_layers[i]+default_range>radius_layers[n_layers-1]:
                    radius_range[i]  = [radius_layers[i]-default_range, radius_layers[i+1]-5]
                else:
                    radius_range[i]  = [radius_layers[i]-default_range, radius_layers[i]+default_range]
            if interface_type[i] == 'dwnbg':
                nhalf_range[i]  = [3, 100]


    else:
        for i in range(1,n_layers):
            print("\n")

            if i == n_layers-1:
                print("Layer: "+ str(i+1) + "(surface)")
                print("Insert grid range parameters (MIN, MAX values): ")

                print("Average Density [kg/m^3]: ",rho_layers[i])
                rho_string = input("Density range [kg/m^3]: ")
                rho_range[i]  = CheckRange(rho_string,[rho_layers[i-1],rho_layers[i+1],"rho"])

            else:
                print("Layer: ", i+1)
                print("Insert grid range parameters (min , max): ")

                print("Average Density [kg/m^3]: ",rho_layers[i])
                rho_string = input("Density range [kg/m^3]: ")
                rho_range[i]  = CheckRange(rho_string,[rho_layers[i-1],rho_layers[i+1],"rho"])
                print("Average Radius [km]: ",radius_layers[i])
                radius_string = input("Radius range [km]: ")
                radius_range[i]  = CheckRange(radius_string,[radius_layers[i-1],radius_layers[i+1],"r"])

            if interface_type[i] == 'dwnbg':
                print("Cutting degree n_half: ",n_half)
                nhalf_string = input("Cutting degree range : ")
                nhalf_range[i]  = CheckRange(nhalf_string)



    return n_counts,rho_range, radius_range, nhalf_range



##########################################################################################################################
##########################################################################################################################


def CheckRange(array,array_control=None):

    """
    Usage
    ----------
    Check the input range for the parameters grid:
    - number of elements
    - float values
    - ascending order


    Parameters
    ----------
    array          : string
                     Input string inserted by the user. To be splitted into two values (min, max)
    array_control  : float, default = None
                     Control array to compare with the input array to constraint the range.

    Output
    ----------
    range-arr       : tuple
                      A tuple with the min and max values of the input range
    """

    if np.size(array.split(",")) != 2:
        print("ERROR: Invalid input. Please insert two numbers separated by a comma.")

    try:
        range_arr = float(array.split(",")[0]), float(array.split(",")[1])
    except ValueError:
        print("ERROR: Invalid input. Please insert two float")
        sys.exit() 

    if float(array.split(",")[0])>float(array.split(",")[1]):
        print(array.split(",")[0])
        print(array.split(",")[1])
        print("ERROR: Invalid input. Range must be in ascending order.")
        sys.exit() 



    # Checkign values against control array
    if array_control is not None:
        match array_control[1]:

            case "rho":
                if range_arr[0] > array_control[0][0] or range_arr[1] < array_control[0][1]:
                        print("ERROR: Invalid input. Layer density heavier than the previous one or lighter than the following one.")
                        sys.exit() 

            case "r":
                if range_arr[0] < array_control[0][0] or range_arr[1] > array_control[0][1]:
                        print("ERROR: Invalid input. Layer radius lower than the previous one or higher than the following one.")
                        sys.exit() 

            case _:
                print("ERROR: Invalid control. Control parameter type not recognized.")
                sys.exit()



    return range_arr 





##########################################################################################################################
##########################################################################################################################



def Solver_M_MoI(param_bulk,array_int):

    """
    Usage
    ----------
    Solver of the M and MoI equations for the innermost layer (inner core). The equations are manually solved by hand and here
    just the solution are implemented.
    The extreme of the core parameters are calculated (namely, "plus" and "minus") as ranges.

    Parameters
    ----------
    param_bulk      : array,
                      Array containing bulk constants:
                        ref_radius      [km]    
                        ref_mass        [kg]
                        ref_MoI         (I/MR^2)   
                        err_MoI         
    array_int      : array,
                     Array containing interior parameters (density and radius):
                        rho_layers          [kg/m^3]
                        radius_layers       [km]

    Output
    ----------
    R_core_minus,R_core_plus        = float,
                                      minimum and maximum values for the core radius according to the Mass and MoI conservation
                                      + MoI uncertainties.
    rho_core_minus,rho_core_plus    = float,
                                      minimum and maximum values for the core density according to the Mass and MoI conservation 
                                      + MoI uncertainties.
    """


    # Parameters
    ref_radius      = param_bulk[0]
    ref_mass        = param_bulk[3]
    ref_MoI         = param_bulk[6]
    err_MoI         = param_bulk[7]

    rho_layers      = array_int[0]
    radius_layers   = array_int[1]

    n_layers = np.size(rho_layers)



    # Solver the M and MoI equations:

    known_term_MoI           = 0
    known_term_M            = 0

    for i in range(2,n_layers):
        known_term_MoI          += 8*np.pi/15*(rho_layers[i]*(radius_layers[i]**5 - radius_layers[i-1]**5)*1e+15)
        known_term_M            += 4*np.pi/3*(rho_layers[i]*(radius_layers[i]**3 - radius_layers[i-1]**3)*1e+9)
    A_plus = 15/(8*np.pi) * ((ref_MoI+err_MoI)*ref_mass*ref_radius*ref_radius*1e+6 - known_term_MoI) - (rho_layers[1]*radius_layers[1]**5)*1e+15
    A_minus = 15/(8*np.pi) * ((ref_MoI-err_MoI)*ref_mass*ref_radius*ref_radius*1e+6 - known_term_MoI) - (rho_layers[1]*radius_layers[1]**5)*1e+15
    B = 3/(4*np.pi) * (ref_mass - known_term_M) - (rho_layers[1]*radius_layers[1]**3)*1e+9

    if A_plus/B < 0 or A_minus/B < 0:
        print("ERROR: Not valiable solution")
        return None, None, None, None
    
    else:
        R_core_plus = np.sqrt(A_plus/B) / 1e+3
        rho_core_plus = rho_layers[1] + np.sqrt(A_plus**(-3) * B**5)

        R_core_minus = np.sqrt(A_minus/B) / 1e+3
        rho_core_minus = rho_layers[1] + np.sqrt(A_minus**(-3) * B**5)

        return R_core_minus,R_core_plus, rho_core_plus,rho_core_minus




##########################################################################################################################
##########################################################################################################################



def NormalizeData_MinMax(data):

    """
    Usage
    ----------
    Normalize data between 0 and 1


    Parameters
    ----------
    data          : array,
                    Input data to be normalized

    Output
    ----------
    normalizedData       : array,
                           Normalized data between 0 and 1
    """


    normalizedData = (data-np.min(data))/(np.max(data)-np.min(data)) 



    return normalizedData




##########################################################################################################################
##########################################################################################################################


def FreeMemory(verbose_opt=False):

    """
    Usage
    ----------
    Freeing memory and cache

    Parameters
    ----------
    verbose_opt : bool, default = False
                  Verbose option
    """


    if verbose_opt: print("Memory is a free elf!")
    gc.collect()
    os.system("sudo sh -c \"echo 3 >'/proc/sys/vm/drop_caches' && swapoff -a && swapon -a \"")
    if verbose_opt: 
        os.system("printf '\n%s\n' 'Ram-cache and Swap Cleared'\"")
        print("\n")
        print("\n")



##########################################################################################################################
##########################################################################################################################

def Corr2_Edo(A,B):


    # Detrending:
    A -= np.mean(A)
    B -= np.mean(B)

    r = np.sum(np.sum(A*B))/np.sqrt(np.sum(np.sum(A*A))*np.sum(np.sum(B*B)))

    return r


##########################################################################################################################
##########################################################################################################################


def MetricsAnalysis(metrics_list, load_opt, saving_dir, models_dir, real_matrix, input_param):

    """
    Usage
    ----------
    Load/Compute input metrics for comparing real and synthetic gravity models.
    The metrics are normalized between 0 and 1.

    
    Parameters
    ----------
    metrics_list : list
                   List of metrics to compute. Options include:
                   - "Delta_mean": Mean difference
                   - "Delta_std": Standard deviation of the difference
                   - "RMSE": Root Mean Squared Error
                   - "R^2": Coefficient of determination
                   - "SSIM": Structural Similarity Index
                   - "spectrum": Spectrum ratio
    load_opt     : bool
                   If True, load precomputed metrics from files.
                   If False, compute metrics from the GRID models (see models_dir).
    saving_dir   : str
                   Directory where metrics and parameters are saved or loaded from.
    models_dir   : str
                   Directory containing GRID subdirectories with synthetic model data.
    real_matrix  : list
                   List of real data matrices:
                   - [0]: U_matrix_real (Gravitational potential)
                   - [1]: deltag_freeair_real (Free-Air anomalies)
                   - [2]: deltag_boug_real (Bouguer anomalies)
                   - [3]: coeffs_topo (Topography coefficients)
                   - [4]: spectrum_real (Gravity spectrum)
    input_param  : list
                   List of input parameters:
                   - [0]: n_min (Minimum spherical harmonic degree)
                   - [1]: n_max (Maximum spherical harmonic degree)
                   - [2]: r (Radius for evaluation)
                   - [3]: i_max (Maximum order for Taylor expansion)
                   - [4]: rho_boug (Crust density for Bouguer correction)

    Output
    ----------
    metrics : numpy.ndarray
              Array containing the computed metrics for the specified metrics_list.
    """


    # ------------------------------------------------------------------------------------------------------


    U_matrix_real       = real_matrix[0]
    deltag_freeair_real = real_matrix[1]
    deltag_boug_real    = real_matrix[2]
    coeffs_topo         = real_matrix[3]
    spectrum_real       = real_matrix[4]

    n_min    = input_param[0]
    n_max    = input_param[1]
    r        = input_param[2]
    i_max    = input_param[3]
    rho_boug = input_param[4]


    # ------------------------------------------------------------------------------------------------------


    # Loading already calculated metrics and interiors parameters:
    if load_opt:

        rho_rng_arr     = np.loadtxt(saving_dir+'rho_layers.dat')
        radius_rng_arr  = np.loadtxt(saving_dir+'radius_layers.dat')
        nhalf_rng_arr   = (np.loadtxt(saving_dir+'n_half.dat')).astype(int)

        if "Delta_mean" in metrics_list and os.path.isfile(saving_dir+'delta_mean.dat'):
            delta_mean = np.loadtxt(saving_dir+'delta_mean.dat')

        if "Delta_std" in metrics_list and os.path.isfile(saving_dir+'delta_std.dat'):
            delta_std = np.loadtxt(saving_dir+'delta_std.dat')

        if "RMSE" in metrics_list and os.path.isfile(saving_dir+'RMSE.dat'):
            RMSE = np.loadtxt(saving_dir+'RMSE.dat')

        if "MAE" in metrics_list and os.path.isfile(saving_dir+'MAE.dat'):
            MAE = np.loadtxt(saving_dir+'MAE.dat')

        if "R^2" in metrics_list and os.path.isfile(saving_dir+'R2.dat'):
            R2 = np.loadtxt(saving_dir+'R2.dat')

        if "SSIM" in metrics_list and os.path.isfile(saving_dir+'SSIM.dat'):
            SSIM = np.loadtxt(saving_dir+'SSIM.dat')

        if "PSNR" in metrics_list and os.path.isfile(saving_dir+'PSNR.dat'):
            PSNR = np.loadtxt(saving_dir+'PSNR.dat')

        if "NCC" in metrics_list and os.path.isfile(saving_dir+'NCC.dat'):
            NCC = np.loadtxt(saving_dir+'NCC.dat')

        if "spectrum" in metrics_list and os.path.isfile(saving_dir+'spectrum_ratio.dat'):
            spectrum_ratio =np.loadtxt(saving_dir+'spectrum_ratio.dat')



    # ------------------------------------------------------------------------------------------------------

    # Evaluate the metrics from the GRID models:
    else:


        # Metrics
        if "Delta_mean" in metrics_list:
            delta_U_mean                = []
            delta_FreeAir_mean          = []
            delta_Bouguer_mean          = []
        if "Delta_std" in metrics_list:
            delta_U_std                 = []
            delta_FreeAir_std           = []
            delta_Bouguer_std           = []
        if "RMSE" in metrics_list:
            RMSE_U                      = []
            RMSE_FreeAir                = []
            RMSE_Bouguer                = []
        if "MAE" in metrics_list:
            MAE_U                      = []
            MAE_FreeAir                = []
            MAE_Bouguer                = []
        if "R^2" in metrics_list:
            R2_U                        = []
            R2_FreeAir                  = []
            R2_Bouguer                  = []
        if "SSIM" in metrics_list:
            ssim_U                      = []
            ssim_FreeAir                = []
            ssim_Bouguer                = []
        if "PSNR" in metrics_list:
            psnr_U                      = []
            psnr_FreeAir                = []
            psnr_Bouguer                = []
        if "NCC" in metrics_list:
            ncc_U                      = []
            ncc_FreeAir                = []
            ncc_Bouguer                = []
        if "spectrum" in metrics_list:
            spectrum_ratio             = []

        # Grid parameters
        rho_rng_arr     = []
        radius_rng_arr  = []
        nhalf_rng_arr   = []

        FreeMemory()
        # Loop over the sub-directories:
        for counter, subdir in tqdm(enumerate(os.listdir(models_dir)), total=len(os.listdir(models_dir))):

            # memory leak issues:
            if counter % 100 == 0:
                FreeMemory()


            # Loading SynthGen coefficients:
            coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')


            # Global analysis (U, H, FreeAir, Bouguer):
            U_matrix,_,deltag_freeair,deltag_boug=Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,
                                                                  r=r,rho_boug=rho_boug,
                                                                  i_max=i_max,plot_opt=None,load_opt=False,verbose_opt=False)
            # Spectrum analysis:
            spectrum_synth = Spectrum(coeffs=[coeffs_tot],n_min=n_min,n_max=n_max,
                                        save_opt=None,load_opt=load_opt,verbose_opt=False)


            # Evaluate metrics:
            if "Delta_mean" in metrics_list:
                delta_U_mean.append(np.mean(U_matrix_real-U_matrix.data))
                delta_FreeAir_mean.append(np.mean(deltag_freeair_real-deltag_freeair.data))
                delta_Bouguer_mean.append(np.mean(deltag_boug_real-deltag_boug.data))
            if "Delta_std" in metrics_list:
                delta_U_std.append(np.std(U_matrix_real-U_matrix.data))
                delta_FreeAir_std.append(np.std(deltag_freeair_real-deltag_freeair.data))
                delta_Bouguer_std.append(np.std(deltag_boug_real-deltag_boug.data))
            if "RMSE" in metrics_list:
                RMSE_U.append(sklearn.metrics.root_mean_squared_error(U_matrix_real,U_matrix.data))
                RMSE_FreeAir.append(sklearn.metrics.root_mean_squared_error(deltag_freeair_real,deltag_freeair.data))
                RMSE_Bouguer.append(sklearn.metrics.root_mean_squared_error(deltag_boug_real,deltag_boug.data))
            if "MAE" in metrics_list:
                MAE_U.append(np.mean(np.abs(U_matrix_real-U_matrix.data)))
                MAE_FreeAir.append(np.mean(np.abs(deltag_freeair_real-deltag_freeair.data)))
                MAE_Bouguer.append(np.mean(np.abs(deltag_boug_real-deltag_boug.data)))
            if "R^2" in metrics_list:
                R2_U.append(Corr2_Edo(U_matrix_real.flatten(),U_matrix.data.flatten()))
                R2_FreeAir.append(Corr2_Edo(deltag_freeair_real.flatten(),deltag_freeair.data.flatten()))
                R2_Bouguer.append(Corr2_Edo(deltag_boug_real.flatten(),deltag_boug.data.flatten())) 
            if "SSIM" in metrics_list:
                ssim_U.append(skimage.metrics.structural_similarity(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                ssim_FreeAir.append(skimage.metrics.structural_similarity(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                ssim_Bouguer.append(skimage.metrics.structural_similarity(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
            if "PSNR" in metrics_list:
                psnr_U.append(skimage.metrics.peak_signal_noise_ratio(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                psnr_FreeAir.append(skimage.metrics.peak_signal_noise_ratio(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                psnr_Bouguer.append(skimage.metrics.peak_signal_noise_ratio(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
            if "NCC" in metrics_list:
                U_matrix_real_mean = U_matrix_real - np.mean(U_matrix_real)
                U_matrix_mean = U_matrix.data - np.mean(U_matrix.data)
                deltag_freeair_real_mean = deltag_freeair_real - np.mean(deltag_freeair_real)
                deltag_freeair_mean = deltag_freeair.data - np.mean(deltag_freeair.data)
                deltag_boug_real_mean = deltag_boug_real - np.mean(deltag_boug_real)
                deltag_boug_mean = deltag_boug.data - np.mean(deltag_boug.data)
                ncc_U.append(np.sum(U_matrix_real_mean * U_matrix_mean) / (np.sqrt(np.sum(U_matrix_real_mean**2)) * np.sqrt(np.sum(U_matrix_mean**2))))
                ncc_FreeAir.append(np.sum(deltag_freeair_real_mean * deltag_freeair_mean) / (np.sqrt(np.sum(deltag_freeair_real_mean**2)) * np.sqrt(np.sum(deltag_freeair_mean**2))))
                ncc_Bouguer.append(np.sum(deltag_boug_real_mean * deltag_boug_mean) / (np.sqrt(np.sum(deltag_boug_real_mean**2)) * np.sqrt(np.sum(deltag_boug_mean**2))))
                
            # if "spectrum" in metrics_list:
            #     spectrum_ratio.append(np.mean(spectrum_real/spectrum_synth))



            # Store interiors parameters:
            rho_rng_arr.append(np.loadtxt(models_dir+subdir+'/rho_layers.dat'))
            radius_rng_arr.append(np.loadtxt(models_dir+subdir+'/radius_layers.dat')) 
            nhalf_rng_arr.append(np.loadtxt(models_dir+subdir+'/n_half.dat'))


        # Normalize the metrics:
        if "Delta_mean" in metrics_list:
            delta_U_mean         = NormalizeData_MinMax(np.abs(delta_U_mean))
            delta_FreeAir_mean   = NormalizeData_MinMax(np.abs(delta_FreeAir_mean))
            delta_Bouguer_mean   = NormalizeData_MinMax(np.abs(delta_Bouguer_mean))
            delta_mean = np.vstack([delta_U_mean, delta_FreeAir_mean, delta_Bouguer_mean])
            np.savetxt(saving_dir+'delta_mean.dat',delta_mean)

        if "Delta_std" in metrics_list:
            delta_U_std          = NormalizeData_MinMax(np.abs(delta_U_std))
            delta_FreeAir_std    = NormalizeData_MinMax(np.abs(delta_FreeAir_std))
            delta_Bouguer_std    = NormalizeData_MinMax(np.abs(delta_Bouguer_std))
            delta_std = np.vstack([delta_U_std, delta_FreeAir_std, delta_Bouguer_std])
            np.savetxt(saving_dir+'delta_std.dat',delta_std)

        if "RMSE" in metrics_list:
            RMSE_U               = NormalizeData_MinMax(np.abs(RMSE_U))
            RMSE_FreeAir         = NormalizeData_MinMax(np.abs(RMSE_FreeAir))
            RMSE_Bouguer         = NormalizeData_MinMax(np.abs(RMSE_Bouguer))
            # RMSE_U                  = 1/(1+np.array(RMSE_U))
            # RMSE_FreeAir            = 1/(1+np.array(RMSE_FreeAir))
            # RMSE_Bouguer            = 1/(1+np.array(RMSE_Bouguer))
            RMSE = np.vstack([RMSE_U, RMSE_FreeAir, RMSE_Bouguer])
            np.savetxt(saving_dir+'RMSE.dat',RMSE)

            
        if "MAE" in metrics_list:
            MAE_U               = NormalizeData_MinMax(np.abs(MAE_U))
            MAE_FreeAir         = NormalizeData_MinMax(np.abs(MAE_FreeAir))
            MAE_Bouguer         = NormalizeData_MinMax(np.abs(MAE_Bouguer))
            # MAE_U               = 1/(1+np.array(MAE_U))
            # MAE_FreeAir         = 1/(1+np.array(MAE_FreeAir))
            # MAE_Bouguer         = 1/(1+np.array(MAE_Bouguer))
            MAE = np.vstack([MAE_U, MAE_FreeAir, MAE_Bouguer])
            np.savetxt(saving_dir+'MAE.dat',MAE)

        if "R^2" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            R2_U                 = (np.array(R2_U) + 1)/2
            R2_FreeAir           = (np.array(R2_FreeAir) + 1)/2
            R2_Bouguer           = (np.array(R2_Bouguer) + 1)/2
            R2 = np.vstack([R2_U, R2_FreeAir, R2_Bouguer])
            np.savetxt(saving_dir+'R2.dat',R2)

        if "SSIM" in metrics_list:
            # ssim_U               = NormalizeData_MinMax(np.abs(ssim_U))
            # R2_FreeAir           = NormalizeData_MinMax(np.abs(ssim_FreeAir))
            # R2_Bouguer           = NormalizeData_MinMax(np.abs(ssim_Bouguer))
            SSIM = np.vstack([ssim_U, ssim_FreeAir, ssim_Bouguer])
            np.savetxt(saving_dir+'SSIM.dat',SSIM)

        if "PSNR" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            
            psnr_U                 = np.array(psnr_U)/np.max(psnr_U)
            psnr_FreeAir           = np.array(psnr_FreeAir)/np.max(psnr_FreeAir)
            psnr_Bouguer           = np.array(psnr_Bouguer)/np.max(psnr_Bouguer)
            PSNR = np.vstack([psnr_U, psnr_FreeAir, psnr_Bouguer])
            np.savetxt(saving_dir+'PSNR.dat',PSNR)

        if "NCC" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            ncc_U                 = (np.array(ncc_U) + 1)/2
            ncc_FreeAir           = (np.array(ncc_FreeAir) + 1)/2
            ncc_Bouguer           = (np.array(ncc_Bouguer) + 1)/2
            NCC = np.vstack([ncc_U, ncc_FreeAir, ncc_Bouguer])
            np.savetxt(saving_dir+'NCC.dat',NCC)


        # if "spectrum" in metrics_list:
        #     spectrum_ratio       = 1-NormalizeData_MinMax(spectrum_ratio)

        np.savetxt(saving_dir+'/rho_layers.dat',rho_rng_arr)
        np.savetxt(saving_dir+'/radius_layers.dat',radius_rng_arr)
        np.savetxt(saving_dir+'/n_half.dat',np.array(nhalf_rng_arr, dtype=np.int64)) 



    # ------------------------------------------------------------------------------------------------------

    metrics = np.empty((0, len(os.listdir(models_dir))),float)
    for metric in metrics_list:
        if metric=="Delta_mean": metrics=np.vstack([metrics, delta_mean])
        if metric=="Delta_std": metrics=np.vstack([metrics, delta_std])
        if metric=="RMSE": metrics=np.vstack([metrics, RMSE])
        if metric=="MAE": metrics=np.vstack([metrics, MAE])
        if metric=="R^2": metrics=np.vstack([metrics, R2])
        if metric=="SSIM": metrics=np.vstack([metrics, SSIM])
        if metric=="PSNR": metrics=np.vstack([metrics, PSNR])
        if metric=="NCC": metrics=np.vstack([metrics, NCC])
        if metric=="spectrum": metrics=np.vstack([metrics, spectrum_ratio])
        

    interiors_parameters = [rho_rng_arr,radius_rng_arr,nhalf_rng_arr]


    return metrics, interiors_parameters








##########################################################################################################################
##########################################################################################################################



def Gaussian_func(x, a, x0, sigma):


    return a * np.exp(-(x-x0)**2/(2*sigma**2))





##########################################################################################################################
##########################################################################################################################


def TopThreshold_Analysis(rho_rng_sort,radius_rng_sort,nhalf_rng_sort, final_metric, threshold_arr, plot_opt: Literal['all','top'] = 'top',saving_dir=None):

    """
    Usage
    ----------
    Analyze and visualize the top% models based on a threshold percentage of the final metric.
    The function generates histograms for density, radius, and degree n_half for each layer, 
    fits Gaussian distributions, and calculates their mean and standard deviation.

    
    Parameters
    ----------
    rho_rng_sort     : numpy.ndarray
                       Sorted array of density values for each layer.
    radius_rng_sort  : numpy.ndarray
                       Sorted array of radius values for each layer.
    nhalf_rng_sort   : numpy.ndarray
                       Sorted array of degree n_half values for each layer.
    final_metric     : numpy.ndarray
                       Array of final metric values for all models.
    threshold_arr    : list
                       List of threshold percentages (e.g., [0.05, 0.1]) to select top models.
    plot_opt         : str, options = ['all', 'top'], default = 'top'
                       - 'top': Plot histograms for the top models only.
                       - 'all': Plot histograms for top models + all models.
    saving_dir       : str, default = None
                       Directory to save the generated histogram plots.
                       If None, plots are not saved.

                       
    Output
    ----------
    rho     : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of density for each layer.
    radius  : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of radius for each layer.
    n_half  : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of degree n_half for each layer.
    fig     : matplotlib.figure.Figure
              Figure handle containing the generated histograms.
    """


    # ------------------------------------------------------------------------------------------------------

    n_layers = np.shape(rho_rng_sort)[1]

    hist_color=matlab_colors

    rho     = np.zeros((n_layers,2))
    radius  = np.zeros((n_layers,2))
    n_half  = np.zeros((n_layers,2))
    thresh_name=''

    # ------------------------------------------------------------------------------------------------------

    # Selecting top [threshold] %
    fig, axs = plt.subplots(n_layers, 3, figsize=(12,10))
    labels=[]
    handles = []

    for j,thresh in enumerate(threshold_arr):

        best_idx_arr = np.where(final_metric >= 1-thresh)
        if np.shape(best_idx_arr)[1]!=0:
            best_idx = best_idx_arr[0][0]
        else:
            print("Top "+ str(thresh*100) + "% models: ", 0, "/", len(final_metric))
            continue
        
        rho_rng_valid_sort_best      = rho_rng_sort[best_idx:]
        radius_rng_valid_sort_best   = radius_rng_sort[best_idx:]
        nhalf_rng_valid_sort_best    = nhalf_rng_sort[best_idx:]

        print("Top "+ str(thresh*100) + "% models: ", np.shape(best_idx_arr)[1], "/", len(final_metric))


        thresh_name+= str(np.round(thresh*100))
        if j!=np.shape(threshold_arr)[0]: thresh_name +='_'

        # ------------------------------------------------------------------------------------------------------

        # Histogram and Analysis

        for i in range(n_layers):

            ax=axs[i, 0]
            n, bins,_ = ax.hist(rho_rng_valid_sort_best[:,i],bins = 100, alpha=1,color=hist_color[j])
            # Fitting normal distribution:
            if np.std(rho_rng_valid_sort_best[:,i]) != 0:
                try:
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(rho_rng_valid_sort_best[:,i]), np.std(rho_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(rho_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(rho_rng_valid_sort_best[:,i]), np.inf)))
                    mu,sigma = popt[1],popt[2]
                    ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                    ax.legend()
                    rho[i,0] = mu
                    rho[i,1] = sigma
                except:
                    print('No Gaussian fit (Density)')
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Density $[kg/m^3]$')
            ax.set_title(r'Layer '+str(i+1))




            ax=axs[i, 1]
            n, bins,_ = ax.hist(radius_rng_valid_sort_best[:,i],bins = 100, alpha=1,color=hist_color[j])
            # Fitting normal distribution:
            if np.std(radius_rng_valid_sort_best[:,i]) != 0:
                try:
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(radius_rng_valid_sort_best[:,i]), np.std(radius_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(radius_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(radius_rng_valid_sort_best[:,i]), np.inf)))
                    mu,sigma = popt[1],popt[2]
                    ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                    ax.legend()
                    radius[i,0] = mu
                    radius[i,1] = sigma
                except:
                    print('No Gaussian fit (Radius)')
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Radius $[km]$')
            ax.set_title(r'Layer '+str(i+1))


            ax=axs[i, 2]
            if nhalf_rng_valid_sort_best[:,i].all() != 0 and len(nhalf_rng_valid_sort_best[:,i]) !=0:
                n, bins,_ = ax.hist(nhalf_rng_valid_sort_best[:,i],bins = int(np.max(nhalf_rng_valid_sort_best[:,i])-np.min(nhalf_rng_valid_sort_best[:,i]))+1, alpha=1,color=hist_color[j])
                # Fitting normal distribution:
                if np.std(nhalf_rng_valid_sort_best[:,i]) != 0:
                    try:
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(nhalf_rng_valid_sort_best[:,i]), np.std(nhalf_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(nhalf_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(nhalf_rng_valid_sort_best[:,i]), np.inf)))
                        mu,sigma = np.round(popt[1]),np.round(popt[2])
                        ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                        ax.legend()
                        n_half[i,0] = mu
                        n_half[i,1] = sigma
                    except:
                        print('No Gaussian fit (n_half)')
            else:
                ax.set_xlim([-5,5])
                ax.set_xticks(np.arange(-5,6))
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Degree n_{half}')
            ax.set_title(r'Layer '+str(i+1))



        labels.append(str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)')
        handles.append(Patch(edgecolor=hist_color[j], facecolor=hist_color[j], fill=True, alpha=0.5))




    if plot_opt=='all':
        for i in range(n_layers):
            ax=axs[i, 0]
            n, bins,_ = ax.hist(rho_rng_sort[:,i],bins = 100, alpha=0.5,color=hist_color[j+1])

            ax=axs[i, 1]
            n, bins,_ = ax.hist(radius_rng_sort[:,i],bins = 100, alpha=0.5,color=hist_color[j+1])

            if nhalf_rng_sort[:,i].all() != 0 or len(nhalf_rng_sort[:,i]) !=0: 
                ax=axs[i, 2]
                n, bins,_ = ax.hist(nhalf_rng_sort[:,i],bins = int(np.max(nhalf_rng_sort[:,i])-np.min(nhalf_rng_sort[:,i]))+1, alpha=0.5,color=hist_color[j+1])

        labels.append('All ('+str(len(final_metric))+' models)')
        handles.append(Patch(edgecolor=hist_color[j+1], facecolor=hist_color[j+1], fill=True, alpha=0.5))



    fig.legend(handles=handles,labels=labels, loc = 'upper center', ncol=n_layers+1,fontsize=13)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)    
    plt.show()


    if saving_dir is not None: plt.savefig(saving_dir+'histograms_'+thresh_name+'.png', dpi=600)


    return rho, radius, n_half, fig








##########################################################################################################################
##########################################################################################################################



def Blurring_map(data, blur_factor):

    """
    Usage
    ----------
    Apply a blurring effect to a 2D map, keeping the original size;
    It performs by resizing it to a smaller resolution (see blur_factor) and then resizing it back to the original resolution.

    Parameters
    ----------
    data         : numpy.ndarray
                   Input 2D array (original data).
    blur_factor  : int
                   Factor by which the resolution of the map is reduced before resizing it back.

    Output
    ----------
    data_blurred : numpy.ndarray
                   Blurred version of the input 2D map.
    """

    dim_orig = np.shape(data)
    dim_blur = (int(dim_orig[0]/blur_factor), int(2*dim_orig[0]/blur_factor))

    data_resized = skimage.transform.resize(data, dim_blur, order=3, mode='reflect', anti_aliasing=True)
    data_blurred = skimage.transform.resize(data_resized, dim_orig, order=3, mode='reflect', anti_aliasing=True)

    return data_blurred




##########################################################################################################################
##########################################################################################################################


