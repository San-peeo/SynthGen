from lib.lib_dep import *

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
    known_term_M             = 0

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
