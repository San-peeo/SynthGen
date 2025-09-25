from lib.lib_dep import *

def Solver_M_MoI(array_bulk,array_int):

    """
    Usage
    ----------
    Solver of the M and MoI equations for the innermost layer (inner core). The equations are manually solved by hand and here
    just the solution are implemented.
    The extreme of the core parameters are calculated (namely, "plus" and "minus") as ranges.

    Parameters
    ----------
    array_bulk      : array,
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
    R_core          = float,
                      core radius according to the Mass and MoI conservation + MoI uncertainties.                
    rho_core        = float,
                      core density according to the Mass and MoI conservation + MoI uncertainties.
    """


    # Parameters
    ref_mass        = array_bulk[0]
    MoI_rng         = array_bulk[1]

    rho_layers      = array_int[0]
    radius_layers   = array_int[1]
    
    ref_radius = radius_layers[-1]

    n_layers = np.size(rho_layers)



    # Solver the M and MoI equations:
 
    known_term_MoI           = 0
    known_term_M             = 0

    for i in range(2,n_layers):
        known_term_MoI          += 8*np.pi/15*(rho_layers[i]*(radius_layers[i]**5 - radius_layers[i-1]**5)*1e+15)
        known_term_M            += 4*np.pi/3*(rho_layers[i]*(radius_layers[i]**3 - radius_layers[i-1]**3)*1e+9)

    MoI_eq = 15/(8*np.pi) * (MoI_rng*ref_mass*ref_radius*ref_radius*1e+6 - known_term_MoI) - (rho_layers[1]*radius_layers[1]**5)*1e+15
    M_eq = 3/(4*np.pi) * (ref_mass - known_term_M) - (rho_layers[1]*radius_layers[1]**3)*1e+9

    if MoI_eq < 0 or M_eq < 0:
        # print("ERROR: Not valiable solution")
        return None, None
    
    else:     
        R0 = np.sqrt(MoI_eq/M_eq)/10**3
        rho0 = rho_layers[1] + M_eq/(R0**3*10**9)

        return R0,rho0



##########################################################################################################################
##########################################################################################################################
