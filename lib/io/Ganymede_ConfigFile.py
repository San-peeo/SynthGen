import numpy as np
import sys

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
        ref_MoI         = 0.3115                                                         # (I/MR^2)    Schubert, G.; Anderson, J. D.; Spohn, T.; McKinnon, W. B. (2004). "Interior composition, structure and dynamics of the Galilean satellites". In Bagenal, F.; Dowling, T. E.; McKinnon, W. B. (eds.). Jupiter: the planet, satellites, and magnetosphere. New York: Cambridge University Press. pp. 281–306. ISBN 978-0521035453. OCLC 54081598. Archived from the original on April 16, 2023. Retrieved July 23, 2019.   
        err_MoI         = 0.0028                                                         # see Petricca et al., 2024 (pyALMA3 paper)
        ref_period      = 7.155*24*3600                                                  # [sec]  




        # Polar Flattening
        r_e = 2633.2             # [km]
        r_p = 2628.8             # [km]
        r_e_fact = r_e/ref_radius
        r_p_fact = r_p/ref_radius
        f = (r_e - r_p)/r_e


        # Output:
        param_bulk = [ref_radius,GM_const,errGM_const,ref_mass,ref_rho,ref_ang_vel,ref_MoI,err_MoI,r_e_fact,r_p_fact,f,ref_period]

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
                    

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_body = [grav_file,top_file,topo_factor,header_opt_grav,format_grav,header_opt_top,format_topo,rho_boug]
        
        return param_body
    


    # ---------------------------------------------------------------------------------------------------------------------------


    def interiors(n_layers):

        r_e_fact = Ganymede_ConfigFile.bulk()[8]
        r_p_fact = Ganymede_ConfigFile.bulk()[9]

        #layers:
        match n_layers:

            # case 2:
            #     rho_layers          = [3530.0,1136.1]
            #     radius_layers       = [1830,2631.2]
            #     interface_type      = ['rng','surf']
            #     interface_addinfo   = [20,10]

            #     layers_name    = ["Rocky interior","Hydrosphere"]                     


            case 2:
                rho_layers          = [3650.0,1136.1]
                radius_layers       = [1800,2631.2]
                interface_type      = ['rng','surf']
                interface_addinfo   = [20,5]

                # Rheology:
                rigid_layers    = [0,0]                    # rigidity [Pa] / shear modulus
                visc_layers     = [0,0]                     # viscosity [Pa s]
                rheo_layers     = ['newton', 'newton' ]       # rheology type
                rheo_addinfo    = [None, None]                    # additional info for rheology (e.g., andrade exponent)



                layers_name    = ["Rocky interior","Hydrosphere"]          

            case 7:                     

                # rho_layers      = [8000, 3400, 3100, 1320, 1235, 1100, 920]
                # radius_layers   = [500,1680,1870,2000,2280,2460,2631.2]
                # interface_type  = ['sph','sphflat','rng','rng','sph','sph','rng']

                # # Additional information for the interface (rng or custom)
                # interface_addinfo  = [0,0,20,1,0,0,10]


                # rho_layers      = [8000, 3400, 3100, 1320, 1235, 1100, 920]
                # radius_layers   = [500,1680,1870,2000,2280,2460,2631.2]



                # rho_layers      = [8000, 3200, 2900, 1320, 1235, 1100, 920]
                # radius_layers   = [670,1840,1870,2000,2280,2460,2631.2]
                # interface_type  = ['sph','sphflat','rng','sph','sph','sph','surf']


                # rho_layers      = [8000, 3200, 2900, 1320, 1235, 1100, 920]
                # radius_layers   = [588,1837.5,1867.5,2015,2281.5,2594.3,2631.2]
                # interface_type  = ['sph','sphflat','rng','sph','sph','sph','surf']

                rho_layers      = [5300, 3300, 2900, 1320, 1235, 1100, 920]
                radius_layers   = [620,1750,1800,2014.8,2281.3,2594.1,2631.2]
                interface_type  = ['sph','sphflat','rng','rng','sph','sph','surf']
                interface_addinfo  = [0,[r_e_fact,r_p_fact],20,2,0,0,5]

                # rho_layers      = [5300, 3300, 3650, 1136, 1235, 1100, 1136]
                # radius_layers   = [620,1700,1800,2014.8,2281.3,2594.1,2631.2]
                # interface_type  = ['sph','sph','rng','sph','sph','sph','surf']
                # interface_addinfo  = [0,0,20,0,0,0,5]


                # Rheology:
                rigid_layers    = [0,0,0,0,0,0,0]                    # rigidity [Pa] / shear modulus
                visc_layers     = [0,0,0,0,0,0,0]                     # viscosity [Pa s]
                rheo_layers     = ['newton', 'newton', 'newton', 'newton', 'newton', 'newton', 'newton']       # rheology type
                rheo_addinfo    = [None, None, None, None, None, None, None]                    # additional info for rheology (e.g., andrade exponent)


   
                layers_name    = ["Core","Mantle","Crust","Ice VI","Ice V","Ocean","Ice I"]                         


            case _:

                print('No existing Moon model with '+str(n_layers)+' layers')
                print('\n')
                sys.exit()
                return

        

        # ----------------------------------------------------------------------------------------------------------------

        # Output:
        param_int = [rho_layers,radius_layers,interface_type, interface_addinfo, rigid_layers, visc_layers, rheo_layers, rheo_addinfo, layers_name]

        return param_int
    




# ---------------------------------------------------------------------------------------------------------------------------
#############################################################################################################################
#############################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------