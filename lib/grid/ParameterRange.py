from lib.lib_dep import *


def InputRange(n_layers,param_int):

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
        return n_counts,[],[],[]




# ------------------------------------------------------------------------------------------------------------------------------------
    # Paramters ranges


    n_counts = int(n_counts)

    rho_layers          = param_int[0]
    radius_layers       = param_int[1]
    interface_type      = param_int[2]
    interface_addinfo   = param_int[3]

    rho_range = np.zeros((n_layers,2))
    radius_range = np.zeros((n_layers,2))
    nhalf_range = np.zeros((n_layers,2))

    default_Rrange=500
    default_rhorange=500



    inherit_opt = input("Inherit previous parameters range (default=yes)? (y/yes/no)  : ")
    print("\n")
    if inherit_opt=="" or inherit_opt=="y" or inherit_opt=="yes":
        return n_counts,[],[],[]



    default_opt = input("Use default ranges? (rho=+/-"+str(default_rhorange)+" [kg/m^3], radius=+/-"+str(default_Rrange)+" [km], n_half=3-100)")
    print("\n")

# NB: - starting from 1 because the innermost layer is not part of the grid (M and MoI conservation)
#     - last layer not radius range (= surface)


    if default_opt=="":
        for i in range(1,n_layers):
            rho_range[i]  = [rho_layers[i]-default_rhorange, rho_layers[i]+default_rhorange]
            if i == n_layers-1:
                radius_range[i]  = [radius_layers[i]-default_Rrange, radius_layers[i]]
            else:
                if radius_layers[i]+default_Rrange>radius_layers[n_layers-1]:
                    radius_range[i]  = [radius_layers[i]-default_Rrange, radius_layers[i+1]-5]
                else:
                    radius_range[i]  = [radius_layers[i]-default_Rrange, radius_layers[i]+default_Rrange]
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
                rho_range[i]  = CheckRange(rho_string) #,[rho_layers[i-1],0])

            else:
                print("Layer: ", i+1)
                print("Insert grid range parameters (min , max): ")

                print("Average Density [kg/m^3]: ",rho_layers[i])
                rho_string = input("Density range [kg/m^3]: ")
                rho_range[i]  = CheckRange(rho_string) #,[rho_layers[i-1],rho_layers[i+1]])
                print("Average Radius [km]: ",radius_layers[i])
                radius_string = input("Radius range [km]: ")
                radius_range[i]  = CheckRange(radius_string) #,[radius_layers[i-1],radius_layers[i+1]])

            if interface_type[i] == 'dwnbg':
                print("Cutting degree n_half: ",interface_addinfo[i])
                nhalf_string = input("Cutting degree range : ")
                nhalf_range[i]  = CheckRange(nhalf_string)



    return n_counts,rho_range, radius_range, nhalf_range



##########################################################################################################################
##########################################################################################################################



def CheckRange(array):

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



    return range_arr 





##########################################################################################################################
##########################################################################################################################
