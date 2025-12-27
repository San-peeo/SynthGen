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
        GM_const        = 0.6262905361210000E+11                                            # [m^3/sec^2]
        errGM_const     = 0.3500000000000000E+06
        ref_mass        = 9.38361380400941e+20                                              # [kg]  Gravity file
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 2.502e-04                                                         # [rad/sec]
        ref_MoI         = 0.364                                                             # (I/MR^2)   
        err_MoI         = 0.022                                                             # 0.15 (estimate), 0.022 Mao and McKinnon 2018

        # Polar Flattening  
        r_e = 482.0           # [km]
        r_p = 445.9            # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius
        f = (r_e - r_p)/r_e


        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        # grav_file = 'Data/Ceres/JGDWN_C70E_KAULA01_SHA.TAB'
        # grav_file = 'Data/Ceres/JGDWN_C70E_ISOSIG01_SHA.TAB'
        grav_file = 'Data/Ceres/JGDWN_CER18D_SHA.TAB'
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Ceres/Ceres_shape_719.sh'
        # top_file = 'Data/Ceres/Ceres_shape_2879.sh'
        topo_factor=1e+3    # topography data are in km (=1)
        header_opt_top = False
        format_topo='bshc'

        # Bouguer density (crustal density)
        rho_boug = 920   # [Kg/m^3]
                 

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Ceres_ConfigFile.bulk()[8]
        r_p_fact = Ceres_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 2:                     

                rho_layers      = [2429,1215]
                radius_layers   = [420.9,470.0]
                interface_type  = ['dwnbg','surf']

                # Additional information for the interface (dwnbg, rng or custom)
                n_half = 18  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [n_half,0]

                layer_name    = ["Core","Crust"]


            case 3:                     

                rho_layers      = [2600,1300,920]
                radius_layers   = [410,435,470.0]
                interface_type  = ['dwnbg','sph','surf']

                # Additional information for the interface (dwnbg, rng or custom)
                n_half = 10  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [[r_e_fact,r_p_fact],0,0]

                layer_name    = ["Core","Mantle","Crust"]





            case _:

                print('No existing Ceres model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type, interface_addinfo,layer_name]

        return param_int
    




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------




