import numpy as np
import sys


class Earth_ConfigFile():

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
        ref_radius      = 6378.137                                                          # [km]
        GM_const        = 3.986004418e+14                                                   # [m^3/sec^2]
        errGM_const     = 0.0008
        ref_mass        = 5.9722e+24                                                        # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)   # [kg/m^3]
        ref_ang_vel     = 1.992e-07                                                         # [rad/sec]
        ref_MoI         = 0.3308                                                            # (I/MR^2)     (Willimas and James, 1994)  
        err_MoI         = 0

        # Polar Flattening
        r_e = 6378             # [km]
        r_p = 6357             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Earth/EGM2008/EGM2008_to2190_TideFree'
        header_opt_grav = False
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Earth/Earth2014.BED2014.degree10800.bshc'
        topo_factor=1e+3    # topography data are in m (=1e+3)
        header_opt_top = False
        format_topo='bshc'

        # ----------------------------------------------------------------------------------------------------------------

        # Bouguer density (crustal density)
        rho_boug = 1800   # [Kg/m^3]
                    

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Earth_ConfigFile.bulk()[8]
        r_p_fact = Earth_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            case 5:                     
                # Simplified model: No oceans, single crust
                rho_layers      = [13088.5,12581.5,7956.5,7090.9,2800]
                radius_layers   = [1221.5,3480.0,5701.0,6151.0,6371.0]
                interface_type  = ['sph','sphflat','sphflat','dwnbg','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                n_half = 80  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [0,[r_e_fact,r_p_fact],[r_e_fact,r_p_fact],n_half,0]


            case 8:                     

                rho_layers      = [13088.5,12581.5,7956.5,7090.9,2691.0,2900,2600,1020]
                radius_layers   = [1221.5,3480.0,5701.0,6151.0,6346.0,6356.0,6368.0,6371.0]
                interface_type  = ['sph','sphflat','sphflat','sphflat','dwnbg','sphflat','surf','surf']

                # Additional information for the interface (dwnbg,rng or custom)
                n_half = 80  # Cutting degree n_half (crust thickness filtering)
                interface_addinfo  = [0,[r_e_fact,r_p_fact],[r_e_fact,r_p_fact],[r_e_fact,r_p_fact],n_half,[r_e_fact,r_p_fact],0,0]

            case _:

                print('No existing Earth model with '+str(n_layers)+' layers')
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


