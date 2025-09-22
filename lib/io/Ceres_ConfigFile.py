import numpy as np
import sys





class Ceres_ConfigFile():

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
        ref_radius      = 470.0                                                             # [km]
        GM_const        = 0.6262889690250000E+11                                            # [m^3/sec^2]
        errGM_const     = 0.2632927000000000E+06
        ref_mass        = 9.3839e+20                                                        # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 2.502e-04                                                         # [rad/sec]
        ref_MoI         = 0.36                                                              # (I/MR^2)   
        err_MoI         = 0.15                                                              # 

        # Polar Flattening
        r_e = 482.0           # [km]
        r_p = 445.9            # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Ceres/JGDWN_C70E_KAULA01_SHA.TAB'
        # grav_file = 'Data/Ceres/JGDWN_C70E_ISOSIG01_SHA.TAB'
        # grav_file = 'Data/Ceres/JGDWN_C70E01_SHA.TAB'
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Ceres/Ceres_shape_719.sh'
        # top_file = 'Data/Ceres/Ceres_shape_2879.sh'
        topo_factor=1e+3    # topography data are in km (=1)
        header_opt_top = False
        format_topo='bshc'

        # Bouguer density (crustal density)
        rho_boug = 1400   # [Kg/m^3]
                 

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Ceres_ConfigFile.bulk()[8]
        r_p_fact = Ceres_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [3410,2225,1400]
                radius_layers   = [200,435,469.4]
                interface_type  = ['sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg, rng or custom)
                n_half = 10  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [[r_e_fact,r_p_fact],n_half,0]






            case _:

                print('No existing Ceres model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type, interface_addinfo]

        return param_int




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------




