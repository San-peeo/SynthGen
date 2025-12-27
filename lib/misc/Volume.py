from lib.lib_dep import *

def Volume(radius_layers):

    """
    Usage
    ----------
    Compute the total mass of a multi-layered spherical body given the radii and densities of each layer.

    Parameters
    ----------
    radius_layers : list or numpy.ndarray
                    Radii of the layers (in km).

    Output
    ----------
    V             : float
                    Total mass of the body (in km^3).
    """


    n_layers = len(radius_layers)


    V=0
    V_layers=[]
    for i in range(n_layers):
        if i != 0:
            V_layers.append(4*np.pi/3*(radius_layers[i]**3 - radius_layers[i-1]**3))
            V += V_layers[-1]
        else:
            V_layers.append( 4*np.pi/3*radius_layers[i]**3)
            V += V_layers[-1]


    return V_layers, V

##########################################################################################################################
##########################################################################################################################
