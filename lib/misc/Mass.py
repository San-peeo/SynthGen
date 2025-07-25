from lib.lib_dep import *

def Mass(radius_layers,rho_layers):

    """
    Usage
    ----------
    Compute the total mass of a multi-layered spherical body given the radii and densities of each layer.

    Parameters
    ----------
    radius_layers : list or numpy.ndarray
                    Radii of the layers (in km).
    rho_layers    : list or numpy.ndarray
                    Densities of the layers (in kg/m^3).

    Output
    ----------
    M             : float
                    Total mass of the body (in kg).
    """


    n_layers = len(radius_layers)


    M=0
    for i in range(n_layers):
        if i != 0:
            M += 4*np.pi/3*rho_layers[i]*(radius_layers[i]**3*1e+9 - radius_layers[i-1]**3*1e+9)
        else:
            M += 4*np.pi/3*rho_layers[i]*radius_layers[i]**3*1e+9
            

    return M

##########################################################################################################################
##########################################################################################################################
