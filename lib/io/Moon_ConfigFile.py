import numpy as np
import sys


class Moon_ConfigFile():

    '''
    Methods:
    -----------

    bulk        : Bulk parameters and constants
    data        : Main datafiles for gravity and topography (+ Bouguer density)    
    interiors   : Interiors parameters as function of layers' number
                  Mass and MoI conservation for deepest layer (inner core) to preserve constants
    '''
   
    def bulk():
        
        # Global parameters
        ref_radius      = 1738.1                                                            # [km]
        GM_const        = 4.9028001218467998e+12                                            # [m^3/sec^2]
        errGM_const     = 0
        ref_mass        = 0.07346e+24                                                       # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 2.7e-06                                                           # [rad/sec]
        ref_MoI         = 0.3929                                                            # (I/MR^2)     (Williams and James, 1996)  
        err_MoI         = 0.0009

        # Polar Flattening
        r_e = 1738.1             # [km]
        r_p = 1736.0             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Moon/GRGM1200l_data.txt'
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Moon/MoonTopo2600p.shape'
        topo_factor=1e+3    # topography data are in km (=1)
        header_opt_top = False
        format_topo='shtools'

        # ----------------------------------------------------------------------------------------------------------------

        # Bouguer density (crustal density)
        rho_boug = 2900   # [Kg/m^3]

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    


    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Moon_ConfigFile.bulk()[8]
        r_p_fact = Moon_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 4:                     

                rho_layers      = [0,0,0,0]
                radius_layers   = [0,0,0,0]
                interface_type  = ['sph','sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                n_half = 40  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [0,[r_e_fact,r_p_fact],n_half,0]

            case _:

                print('No existing Moon model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type,interface_addinfo]

        return param_int
    




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------