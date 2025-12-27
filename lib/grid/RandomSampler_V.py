from lib.lib_dep import *
from lib.misc.Volume import *
from lib.misc.Solver_M_MoI_ic_oc import *


def RandomSampler_V(param_int,rho_range, radius_range, rigid_range, visc_range, nhalf_range,ref_MoI,err_MoI,ref_mass,error_flag):


    """
    Usage
    ----------
    Random sampler of interior-model parameters for a layered spherical body subject to mass/MoI constraints.
    The function draws random values for layer densities, layer radii, mechanical properties (rigidity, viscosity)
    and nhalf values from the provided ranges and returns arrays of sampled parameters. 
    If `error_flag` is True, samples are constrained to match the reference mass and moment-of-inertia (MoI) within the provided error.
    Radii are sampled uniformly in volume space to ensure an even distribution of layer thicknesses.
    
    Parameters
    ----------
    param_int    : list
                   Interior model template with fields (expected order):
                   [rho_layers, radius_layers, interface_type, interface_addinfo, ...]
                   - rho_layers: base densities for each layer (used as defaults)
                   - radius_layers: interface radii (km)
                   - interface_type, interface_addinfo: per-interface descriptors
    rho_range    : array-like or list of tuples
                   Ranges for layer densities. Either a global [min,max] applied to all layers or
                   one min/max tuple per layer: [(rmin0,rmax0), ..., (rminN,rmaxN)].
    radius_range : array-like or list of tuples
                   Ranges for interface radii. Either global [min,max] or per-layer tuples.
    rigid_range  : array-like or list of tuples
                   Per-layer ranges for elastic rigidity (Pa) or a global range.
    visc_range   : array-like or list of tuples
                   Per-layer ranges for viscosity (Pa·s) or a global range.
    nhalf_range  : array-like or list
                   Acceptable nhalf values per interface (or a set to choose from).
    ref_MoI      : float
                   Reference moment of inertia (I/MR^2) used to enforce constraints.
    err_MoI      : float
                   Allowed deviation for MoI from the reference (same units as ref_MoI).
    ref_mass     : float
                   Reference mass (kg) used to enforce mass conservation.
    error_flag   : bool
                   If True, enforce mass/MoI constraints by adjusting sampled internal densities (using Solver_M_MoI2).
                   If False, no mass/MoI enforcement is applied.

    Output
    ----------
    rho_rng             : numpy.ndarray
                          Sampled density array with shape (N, n_layers), N = number of sampled configurations.
    radius_rng          : numpy.ndarray
                          Sampled radii array with shape (N, n_layers).
    nhalf_rng           : numpy.ndarray
                          Sampled nhalf array with shape (N, n_layers) or (N, n_interfaces).
    rigid_rng           : numpy.ndarray
                          Sampled rigidity array with shape (N, n_layers).
    visc_rng            : numpy.ndarray
                          Sampled viscosity array with shape (N, n_layers).
    interface_addinfo_rng : list or array
                          Corresponding sampled interface_addinfo; unchanged entries are copied from param_int.
    """




    radius_layers       = param_int[1]
    interface_type      = param_int[2]
    interface_addinfo   = param_int[3]


    n_layers = len(radius_layers)



    rho_rng     =  np.zeros([1,n_layers])
    radius_rng  =  np.zeros([1,n_layers])
    nhalf_rng   =  np.zeros([1,n_layers])       
    interface_addinfo_rng = interface_addinfo

    rigid_rng        = np.zeros([1,n_layers])
    visc_rng         = np.zeros([1,n_layers])





    # LAYERS:

    radius_rng[0,-1] = radius_layers[-1]

    for i in reversed(range(1,n_layers)):

        rho_rng[0,i]                = random.uniform(rho_range[i][0],rho_range[i][1])
        rigid_rng[0,i]              = random.uniform(rigid_range[i][0],rigid_range[i][1])
        visc_rng[0,i]               = 10**random.uniform(visc_range[i][0],visc_range[i][1])


        # Extraction of Volume to have an uniform sampling on the radius space:
        V_min,_ = Volume([radius_range[i-1][0],radius_rng[0,i]])
        V_min = V_min[-1]
        V_max,_ = Volume([radius_range[i-1][1],radius_rng[0,i]])
        V_max = V_max[-1]   
        V_rng               = random.uniform(V_min,V_max)
        radius_rng[0,i-1]   = (radius_rng[0,i]**3 - 3/(4*np.pi)*V_rng)**(1/3)


        if interface_type[i] == 'dwnbg':
            nhalf_rng[0,i]              = random.randint(nhalf_range[i][0],nhalf_range[i][1])
            interface_addinfo_rng[i]    = nhalf_rng[0,i]





    # INNER CORE - OUTER CORE densities (Mass and MoI conservation):
    MoI_rng = random.uniform(ref_MoI-err_MoI,ref_MoI+err_MoI)
    [rho_icore, rho_ocore] = Solver_M_MoI_ic_oc([ref_mass,MoI_rng],[rho_rng[0,:],radius_rng[0,:]])



    if rho_icore is None or rho_ocore is None:
        print("ERROR: No viable Core solution")
        error_flag=True
        return None, None, None, None, None, error_flag


    # Core parameters:
    if not error_flag:
        rho_rng[0,0] = rho_icore
        rho_rng[0,1] = rho_ocore

    rigid_rng[0,0]  = random.uniform(rigid_range[0][0],rigid_range[0][1])
    visc_rng[0,0]   = 10**random.uniform(visc_range[0][0],visc_range[0][1])


    # ----------------------------------------------------------------------------------------------------------------


    # Rounding grid extraction results:
    rho_rng      = np.round(rho_rng,1)
    radius_rng   = np.round(radius_rng,1)



    return rho_rng, radius_rng, rigid_rng, visc_rng, interface_addinfo_rng, error_flag