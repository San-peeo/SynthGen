from lib.lib_dep import *
from lib.io.Planets_ConfigFiles import *


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
    param_int      : array,
                      Array containing interior parameters (as function of n_layers):
                        rho_layers          [kg/m^3]
                        radius_layers       [km]
                        interface_type      [string]
                        interface_info      [float]
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
        coeffs_grav.gm = GM_const                   # m^3/s^2
        coeffs_grav.r0 = ref_radius*1e+3            # m
    else:
        print('No gravitational file (zero data) \n')
        coeffs_grav = None 




    # Topography data (+ conversion into [km])
    print('Topography datafile:')

    if topo_file is not None:
        print(topo_file + '\n')
        coeffs_topo = pysh.SHCoeffs.from_file(topo_file, format=format_topo, lmax=n_max, units = 'km',header=header_opt_topo)
        if topo_factor != 1:
            coeffs_topo /= topo_factor
            if body == "Earth": coeffs_topo += ref_radius 
    else:
        print('No Topography file (zero data) \n')
        coeffs_topo = None 




    if n_layers is not None: 
        return param_bulk,param_body,param_int, coeffs_grav, coeffs_topo
    else:
        return param_bulk,param_body, coeffs_grav, coeffs_topo

##########################################################################################################################
##########################################################################################################################
