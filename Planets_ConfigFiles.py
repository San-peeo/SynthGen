import numpy as np
import sys


# CONFIG FILES FOR PLANETS



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
        err_MoI         = 0.00089                                                           # formal error = 0.00089, distribution error = 0.014  (different estimates)

        # Polar Flattening
        r_e = 2440.5             # [km]
        r_p = 2438.3             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = 'Data/Mercury/HgM009/HgM009.sha'
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = 'Data/Mercury/gtmes_150v05/gtmes_150v05_sha_nohead.txt'
        topo_factor=1    # topography data are in km (=1)
        header_opt_top = False
        format_topo='shtools'

        # Bouguer density (crustal density)
        rho_boug = 2900   # [Kg/m^3]
                    
        # Cutting degree n_half (crust thickness filtering)
        n_half = 40   

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug,n_half]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [6992,3200,2900]
                radius_layers   = [2039,2404,2439.4]
                interface_type  = ['sphflat','dwnbg','surf']



            case 4:                     # see Margot et al., "Mercury's Internal Structure, 2018

                rho_layers      = [8652.52,6909.98, 3343.35, 2903.03]
                radius_layers   = [666.577, 2023.66, 2402.61, 2439.4]
                interface_type  = ['sph','sphflat','dwnbg','surf']



            case _:

                print('No existing Mercury model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type]

        return param_int




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------





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
                    
                    
        # Cutting degree n_half (crust thickness filtering)
        n_half = 80   

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug,n_half]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        #layers:
        match n_layers:

            case 3:                     

                rho_layers      = [13000,3300,2800]
                radius_layers   = [3200,6020,6051.8]
                interface_type  = ['sphflat','dwnbg','surf']

   
            case _:

                print('No existing Venus model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type]

        return param_int





# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------





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
                    
        # Cutting degree n_half (crust thickness filtering)
        n_half = 80   

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug,n_half]
        
        return param_body
    

    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        #layers:
        match n_layers:

            case 5:                     
                # Simplified model: No oceans, single crust
                rho_layers      = [13088.5,12581.5,7956.5,7090.9,2800]
                radius_layers   = [1221.5,3480.0,5701.0,6151.0,6371.0]
                interface_type  = ['sph','sphflat','sphflat','dwnbg','surf']

   
            case 8:                     

                rho_layers      = [13088.5,12581.5,7956.5,7090.9,2691.0,2900,2600,1020]
                radius_layers   = [1221.5,3480.0,5701.0,6151.0,6346.0,6356.0,6368.0,6371.0]
                interface_type  = ['sph','sphflat','sphflat','sphflat','dwnbg','sphflat','surf','surf']



            case _:

                print('No existing Earth model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type]

        return param_int



# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------



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
                    
        # Cutting degree n_half (crust thickness filtering)
        n_half = 40   

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug,n_half]
        
        return param_body
    


    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        #layers:
        match n_layers:

            case 4:                     

                rho_layers      = [0,0,0,0]
                radius_layers   = [0,0,0,0]
                interface_type  = ['sph','sphflat','dwnbg','surf']

   
            case _:

                print('No existing Moon model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type]

        return param_int
    




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------



class Ganymede_ConfigFile():

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
        ref_radius      = 2631.2                                                         # [km]
        GM_const        = 9.8780e+12                                                     # [m^3/sec^2]
        errGM_const     = 0
        ref_mass        = 1.48e+23                                                       # [kg]
        ref_rho         = ref_mass/(4/3*np.pi*ref_radius**3*1e+9)                        # [kg/m^3]
        ref_ang_vel     = 8.264e-07                                                      # [rad/sec]
        ref_MoI         = 0.3115                                                         # (I/MR^2)    
        err_MoI         = 0.01

        # Polar Flattening
        r_e = 2631.2             # [km]
        r_p = 2631.2             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius

        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact]

        return param_bulk


    # ---------------------------------------------------------------------------------------------------------------------------


    def data():

        # Gravity data file
        grav_file = None
        header_opt_grav = True
        format_grav='shtools'

        # Topography data file
        top_file = None
        topo_factor=1e+3    # topography data are in km (=1)
        header_opt_top = False
        format_topo='shtools'

        # ----------------------------------------------------------------------------------------------------------------

        # Bouguer density (crustal density)
        rho_boug = 920   # [Kg/m^3]
                    
        # Cutting degree n_half (crust thickness filtering)
        n_half = 25   

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug,n_half]
        
        return param_body
    


    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        #layers:
        match n_layers:

            case 7:                     

                rho_layers      = [8000, 3400, 3100, 1320, 1235, 1100, 920]
                radius_layers   = [570,1820,1870,2000,2280,2460,2631.2]
                interface_type  = ['sph','sph','rng','sph','sph','sph','surf']

   
            case _:

                print('No existing Moon model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type]

        return param_int