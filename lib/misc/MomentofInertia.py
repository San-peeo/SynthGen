from lib.lib_dep import *

def MomentofInertia(radius_layers,rho_layers):

    """
    Usage
    ----------
    Compute the moment of inertia for a multi-layered spherical body given the radii and densities of each layer.

    Parameters
    ----------
    radius_layers : list or numpy.ndarray
                    Radii of the layers (in km).
    rho_layers    : list or numpy.ndarray
                    Densities of the layers (in kg/m^3).

    Output
    ----------
    MoI           : float
                    Moment of inertia of the body (in kg·m²).
    """

    n_layers = len(radius_layers)


    MoI=0
    for i in range(n_layers):
        if i != 0:
            MoI += 8*np.pi/15*rho_layers[i]*(radius_layers[i]**5*1e+15 - radius_layers[i-1]**5*1e+15)
        else:
            MoI += 8*np.pi/15*rho_layers[i]*radius_layers[i]**5*1e+15


    return MoI

##########################################################################################################################
##########################################################################################################################
