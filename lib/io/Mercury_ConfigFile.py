import numpy as np
import sys
from lib.lib_dep import *



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
        ref_radius      = 2439.4                                            # [km]
        GM_const        = 2.203186356600000e+13                             # [m^3/sec^2]
        errGM_const     = 0
        ref_mass        = 3.306e+23                                         # [kg] = 3.301e+23   
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)           # [kg/m^3]
        ref_ang_vel     = 8.264e-07                                         # [rad/sec]
        ref_MoI         = 0.350                                             # (I/MR^2)   Margot et al, 2018 = 0.346
        err_MoI         = 0.014                                             # formal error = 0.00089, distribution error = 0.014  (different estimates)
        ref_period      = 78.969*24*3600                                    # [sec]  

        # [h2,l2,k2]
        ref_LN          = [mp.mpc(0.7865,-0.0023), None, mp.mpc(0.4496,-0.0013)]                           #   Margot et al, 2018 k2_real = 0.455
        err_LN          = [mp.mpc(0.015*0.7865,0.015*0.0023), None, mp.mpc(0.05*0.4496,0.2*0.0013)]                          # Margot et al, 2018  k2_imm = 0.012
        # ref_LN          = [None, None, mp.mpc(0.4496,-0.0013)]                           #   Margot et al, 2018 k2_real = 0.455
        # err_LN          = [None, None, mp.mpc(0.05*0.4496,0.2*0.0013)]                          # Margot et al, 2018  k2_imm = 0.012
        # ref_LN          = [None, None, None]                           
        # err_LN          = [None, None, None]                            


        # Polar Flattening
        r_e = 2440.5             # [km]
        r_p = 2438.3             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius
        f = (r_e - r_p)/r_e

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact,f,ref_period,ref_LN,err_LN]

        return param_bulk

    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Mercury/HgM009/HgM009.sha'
        header_opt_grav = True

        # grav_file = 'Results/Synthetic/Mercury/i1_sph_r1_1119.7_rho1_7140.4_i2_sphflat_r2_2019.7_rho2_7000.0_i3_dwnbg_r3_2399.7_rho3_3400.0_nhalf3_40_i4_surf_r4_2439.7_rho4_2800.0/coeffs_tot.dat'
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

        layers_name = None

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [7257.26,3300.0,2900.0]
                radius_layers   = [1989.28,2404.0,2439.4]
                interface_type  = ['sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg, rng or custom)
                interface_addinfo  = [[r_e_fact,r_p_fact],40,0]

                layers_name    = ["Core","Mantle","Crust"]


            # case 3:                    

            #     rho_layers      = [7256.2,3581.2,2514.3]
            #     radius_layers   = [1933.7,2411.5,2439.4]
            #     interface_type  = ['sphflat','dwnbg','surf']

            #     # Additional information for the interface (dwnbg,rng or custom)
            #     interface_addinfo  = [0,[r_e_fact,r_p_fact],39,0]
            #     layers_name    = ["Core","Mantle","Crust"]


            
            # case 4:                     # see Margot et al., "Mercury's Internal Structure, 2018

            #     # Gravity:
            #     rho_layers      = [8652.52,6909.98, 3343.35, 2903.03]
            #     radius_layers   = [666.577, 2023.66, 2402.61, 2439.4]
            #     interface_type  = ['sph','sphflat','dwnbg','surf']

            #     # Additional information for the interface (dwnbg,rng or custom)
            #     interface_addinfo  = [0,[r_e_fact,r_p_fact],40,0]


            #     # Rheology:
            #     rigid_layers    = [1.0e+11, 0, 8.0e+10, 5.5e+10]                    # rigidity [Pa] / shear modulus
            #     visc_layers     = [1.0e+20, 1e+6, 5e+22, 1e+22]                     # viscosity [Pa s]
            #     rheo_layers     = ['andrade', 'newton', 'andrade', 'elastic']       # rheology type
            #     rheo_addinfo    = [0.33333, None, 0.33333, None]                    # additional info for rheology (e.g., andrade exponent)

            #     # Layer names:
            #     layers_name    = ["Inner Core","Outer Core","Mantle","Crust"]



            
            case 4:                     

                # Gravity:
                rho_layers      = [7140.4, 7000.0, 3400.0, 2800.0]
                radius_layers   = [1119.7, 2019.7, 2399.7, 2439.7]
                interface_type  = ['sph','sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                interface_addinfo  = [0,[r_e_fact,r_p_fact],40,0]


                # Rheology (# Credit to Dr. Consorzi Anastasia work):
                rigid_layers    = [1.0e+11, 0, 8.0e+10, 5.5e+10]                    # rigidity [Pa] / shear modulus
                visc_layers     = [1.0e+20, 1e+6, 5e+22, 1e+22]                     # viscosity [Pa s]
                rheo_layers     = ['andrade', 'newton', 'andrade', 'elastic']       # rheology type
                rheo_addinfo    = [0.33333, None, 0.33333, None]                    # additional info for rheology (e.g., andrade exponent)

                # Layer names:
                layers_name    = ["Inner Core","Outer Core","Mantle","Crust"]



            case _:

                print('No existing Mercury model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------


        if layers_name is None:
            layers_name = ['Layer '+str(i+1) for i in range(n_layers)]


        # Output:
        param_int = [rho_layers,radius_layers,interface_type, interface_addinfo, rigid_layers, visc_layers, rheo_layers, rheo_addinfo, layers_name]

        return param_int




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------




