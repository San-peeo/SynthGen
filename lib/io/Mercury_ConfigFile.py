import numpy as np
import sys



# NB: interiors interface options:
#         - sph        = sph on the layer radius (No interface topography)
#         - sphflat    = polar flattening
#         - dwnbg      = Downwarding Bouguer anomalies to infer the interface relief
#         - surf       = surf topography (from data)





class Mercury_ConfigFile():

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
        ref_radius      = 2439.4                                                            # [km]
        GM_const        = 2.203186356600000e+13                                             # [m^3/sec^2]
        errGM_const     = 0
        ref_mass        = 3.301e+23                                                         # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 8.264e-07                                                         # [rad/sec]
        ref_MoI         = 0.34597                                                           # (I/MR^2)   Margot et al, 2018
        err_MoI         = 0.00089                                                                 # formal error = 0.00089, distribution error = 0.014  (different estimates)

        # Polar Flattening
        r_e = 2440.5             # [km]
        r_p = 2438.3             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius
        f = (r_e - r_p)/r_e

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Mercury/HgM009/HgM009.sha'
        header_opt_grav = True

        # grav_file = 'Data/Mercury/i1_sph_r1_666.577_rho1_8652.52_i2_sphflat_r2_2023.66_rho2_6909.98_i3_dwnbg_r3_2402.61_rho3_3343.35_nhalf3_40_i4_surf_r4_2439.4_rho4_2903.03/coeffs_tot.dat'
        # header_opt_grav = True

        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Mercury/gtmes_150v05/gtmes_150v05_sha_nohead.txt'
        topo_factor=1    # topography data are in km (=1)
        header_opt_top = False
        format_topo='shtools'

        # Bouguer density (crustal density)
        rho_boug = 2800   # [Kg/m^3]
                 

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Mercury_ConfigFile.bulk()[8]
        r_p_fact = Mercury_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [7257.26,3300.0,2900.0]
                radius_layers   = [1989.28,2404.0,2439.4]
                interface_type  = ['sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg, rng or custom)
                interface_addinfo  = [[r_e_fact,r_p_fact],40,0]

                layer_name    = ["Core","Mantle","Crust"]


            # case 3:                    

            #     rho_layers      = [7256.2,3581.2,2514.3]
            #     radius_layers   = [1933.7,2411.5,2439.4]
            #     interface_type  = ['sphflat','dwnbg','surf']

            #     # Additional information for the interface (dwnbg,rng or custom)
            #     interface_addinfo  = [0,[r_e_fact,r_p_fact],39,0]
            #     layer_name    = ["Core","Mantle","Crust"]


            
            case 4:                     # see Margot et al., "Mercury's Internal Structure, 2018

                rho_layers      = [8652.52,6909.98, 3343.35, 2903.03]
                radius_layers   = [666.577, 2023.66, 2402.61, 2439.4]
                interface_type  = ['sph','sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                interface_addinfo  = [0,[r_e_fact,r_p_fact],40,0]

                layer_name    = ["Inner Core","Outer Core","Mantle","Crust"]


            # case 4:                     

            #     rho_layers      = [7398.0,6097.0, 3390.0, 2650.0]
            #     radius_layers   = [1894.0, 2000.0, 2418.0, 2439.4]
            #     interface_type  = ['sph','sphflat','dwnbg','surf']

            #     # Additional information for the interface (dwnbg,rng or custom)
            #     interface_addinfo  = [0,[r_e_fact,r_p_fact],67,0]

            #     layer_name    = ["Inner Core","Outer Core","Mantle","Crust"]
 

            case _:

                print('No existing Mercury model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type, interface_addinfo, layer_name]

        return param_int




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------




