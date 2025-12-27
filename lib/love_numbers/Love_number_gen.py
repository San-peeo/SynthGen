from lib.lib_dep import *
from lib.misc.Mass import *
from lib.love_numbers.Mu_Complex import *
from lib.love_numbers.LN_Matrices import *

# Credit to Dr. Consorzi Anastasia's work


def Love_number_gen(radius_layers, rho_layers, rheo_layers, rigid_layers, visc_layers, rheo_addinfo, period, l=2,saving_dir=None):


    """
    Usage
    ----------
    Compute complex Love numbers (h_n, l_n, k_n) for a layered spherical body at a given tidal frequency.
    The function builds the propagation/transfer matrices for each layer, applies boundary conditions and normalization,
    and returns the degree-n Love numbers for the specified harmonic degree l (default l=2).

    Parameters
    ----------
    radius_layers : list,
                    Radii of layer interfaces (km). Ordered from center to surface.
    rho_layers    : list,
                    Densities for each layer (kg/m^3). One value per layer.
    rheo_layers   : list,
                    Rheology identifier for each layer (e.g., 'elastic','viscoelastic', ...).
    rigid_layers  : list,
                    Elastic rigidity (shear modulus) for each layer (Pa).
    visc_layers   : list,
                    Viscosity for each layer (Pa·s), if applicable for the chosen rheology.
    rheo_addinfo  : list,
                    Additional rheology parameters per layer (e.g., Andrade parameters).
    period        : float,
                    Tidal period (s). Frequency used to compute complex rigidity mu(s) = mu(omega).
    l             : int, optional
                    Spherical harmonic degree for which to compute Love numbers (default: 2).
    saving_dir    : str
                    Saving directory for the output result file


    Output
    ----------
    [h_n, l_n, k_n] : list, mp.mpc
                      Complex Love numbers for degree l:
                      - h_n : complex (vertical displacement Love number)
                      - l_n : complex (horizontal displacement Love number)
                      - k_n : complex (tidal potential Love number)
    """




    n_layers = len(radius_layers)

    # Layer's Mass
    M_layers,_   = Mass(radius_layers,rho_layers)

    # Conversion from km to m
    radius_layers = [r * 1e+3 for r in radius_layers]





    # Complex rigidity mu(s) and gravity g computation at each layer
    g=[]
    mu_complex=[]
    for i in range(n_layers):
        g.append(G_const*np.sum(M_layers[:i+1])/(radius_layers[i]**2))
        mu_complex.append(Mu_Complex(rheology=rheo_layers[i], rigidity=rigid_layers[i],viscosity=visc_layers[i],
                            rheo_addinfo=rheo_addinfo[i],frequency=2*np.pi/period))


    #  Boundary condition matrix (.T = transpose)
    B = mp.matrix([[0,0,-5/(radius_layers[-1])]], dtype=mp.mpc).T

    # Normalization dimensions matrix
    N_tidal = mp.matrix([[1/g[-1],0,0],[0,1/g[-1],0],[0,0,1.0]], dtype=mp.mpc)


    # Projector matrices
    P1 = mp.zeros(3,6, dtype=mp.mpc)
    P1[0,0]=1
    P1[1,1]=1
    P1[2,4]=1

    P2 = mp.zeros(3,6, dtype=mp.mpc)
    P2[0,2]=1
    P2[1,3]=1
    P2[2,5]=1


    # Computation of Love numbers:



    # Temporary term matrix
    temp = Y(radius_layers[-1], rho_layers[-1], mu_complex[-1], g[-1], l)    

    for i in reversed(range(n_layers-1)):
        if i!=0:    
            temp = temp * Y_inv(radius_layers[i], rho_layers[i+1],mu_complex[i+1], g[i],l) * Y(radius_layers[i], rho_layers[i],mu_complex[i], g[i], l)              
        else: break

    # Core Matrix
    I_core=Ic(radius_layers[0], rho_layers[0], mu_complex[0], g[0], l)

    # Final step:
    temp = temp * Y_inv(radius_layers[0], rho_layers[1], mu_complex[1], g[0],l) * I_core

    # Love number vector
    LN_array = (N_tidal)**-1 * (P1*temp) * (P2*temp)**-1 * B


    h_n = LN_array[0]                       # Love number h_n
    l_n = LN_array[1]                       # Love number l_n
    k_n=-(1+LN_array[2])                    # Love number k_n


    if saving_dir is not None:
        # Save Love numbers to a text file
        np.savetxt(saving_dir+'/LN_number_'+str(l)+'.dat',[h_n,l_n,k_n],fmt='%32s')



    return [h_n,l_n,k_n]
