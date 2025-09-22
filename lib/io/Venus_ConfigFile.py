import numpy as np
import sys



class Venus_ConfigFile():

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
        ref_radius      = 6051.8                                                            # [km]
        GM_const        = 3.248585920790000e+14                                             # [m^3/sec^2]
        errGM_const     = 0.6376000000000000e-02
        ref_mass        = 4.8673e+24                                                         # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 3.232e-07                                                         # [rad/sec]
        ref_MoI         = 0.337                                                             # (I/MR^2)     (Margot et al., 2021)  
        err_MoI         = 0.024

        # Polar Flattening
        r_e = 6051.8             # [km]
        r_p = 6051.8             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk

    # ---------------------------------------------------------------------------------------------------------------------------

    def data():

        # Gravity data file
        grav_file = 'Data/Venus/shgj180u_noheader.a01'
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Venus/VenusTopo719.shape'
        topo_factor=1e+3    # topography data are in km (=1)
        header_opt_top = False
        format_topo='shtools'

        # ----------------------------------------------------------------------------------------------------------------

        # Bouguer density (crustal density)
        rho_boug = 2800   # [Kg/m^3]
                    

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Venus_ConfigFile.bulk()[8]
        r_p_fact = Venus_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [13000,3300,2800]
                radius_layers   = [3200,6020,6051.8]
                interface_type  = ['sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                n_half = 80  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [[r_e_fact,r_p_fact],n_half,0]


            case _:

                print('No existing Venus model with '+str(n_layers)+' layers')
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


