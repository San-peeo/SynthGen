from lib.lib_dep import *


def InputRange(n_layers,param_int,run_n_counts=[]):

    """
    Usage
    ----------
    Input range for the parameters grid:
    - n_counts = number of simulations
    - ranges for density, radius and (eventually) cutting degree n_half, and rheology parameters as rigidity and viscosity 
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
                        rigid_layers        [Pa]    
                        visc_layers         [Pa s]    



    Output
    ----------
    n_counts                                : int,
                                              number of simulations for the grid
    rho_range, radius_range, nhalf_range    : tuple [2,n_layers]
                                              Tuples with the min and max values of the input range for each layer                    
    """


# ------------------------------------------------------------------------------------------------------------------------------------
    # N counts simulations

    n_counts = input("Insert number of simulations you want to evaluate (default=max): ")
    print("\n")

    # Maximum value possible
    if n_counts=="":
        return n_counts,[],[],[],[],[]




# ------------------------------------------------------------------------------------------------------------------------------------
    # Parameters ranges


    n_counts = int(n_counts)

    rho_layers          = param_int[0]
    radius_layers       = param_int[1]
    interface_type      = param_int[2]
    interface_addinfo   = param_int[3]
    rigid_layers        = param_int[4]
    visc_layers         = param_int[5]
    rheo_layers         = param_int[6]


    rho_range       = np.zeros((n_layers,2))
    radius_range    = np.zeros((n_layers,2))
    nhalf_range     = np.zeros((n_layers,2))
    rigid_range     = np.zeros((n_layers,2))
    visc_range      = np.zeros((n_layers,2))





    inherit_opt="no"
    if len(run_n_counts)!=0:
        inherit_opt = input("Inherit previous parameters range (default=yes)? (y/yes/no) : ")
        print("\n")

    if inherit_opt=="" or inherit_opt=="y" or inherit_opt=="yes":
        return n_counts,[],[],[],[],[]



    # Inserting parameters ranges:
    print("GRID:")
    print("Insert grid range parameters (MIN, MAX values): ")

    for i in range(0,n_layers):
        print("\n")

        print("Layer: ", i+1)

        print("Average Density [kg/m^3]: ",rho_layers[i])
        rho_string = input("Density range [kg/m^3]: ")
        rho_range[i]  = CheckRange(rho_string,rho_layers[i]) 
        print("Average Radius [km]: ",radius_layers[i])
        radius_string = input("Radius range [km]: ")
        radius_range[i]  = CheckRange(radius_string,radius_layers[i]) 
        if interface_type[i] == 'dwnbg':
            print("Cutting degree n_half: ",interface_addinfo[i])
            nhalf_string = input("Cutting degree range: ")
            nhalf_range[i]  = CheckRange(nhalf_string)

        print("Rheology: "+rheo_layers[i])
        print("Average Rigidity [Pa]: ",rigid_layers[i])
        rigid_string = input("Rigidity range [Pa]: ")
        rigid_range[i]  = CheckRange(rigid_string,rigid_layers[i])
        print("Average Viscosity log10 [Pa s]: ",np.log10(visc_layers[i]))
        visc_string = input("Viscosity range exponent log10[Pa s]: ")
        visc_range[i]  = CheckRange(visc_string,np.log10(visc_layers[i]))




    return n_counts,rho_range, radius_range, nhalf_range, rigid_range, visc_range





##########################################################################################################################
##########################################################################################################################



def CheckRange(array,default_value=None):

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


    Output
    ----------
    range-arr       : tuple
                      A tuple with the min and max values of the input range
    """

    if array=="":
        range_arr=default_value,default_value
        return range_arr 


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



    return range_arr 





##########################################################################################################################
##########################################################################################################################
