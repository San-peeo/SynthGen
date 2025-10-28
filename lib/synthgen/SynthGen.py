from lib.lib_dep import *
from lib.misc.FreeMemory import *
from lib.globe_analysis.CrustThickness import *


def SynthGen(param_int,n_max,coeffs_grav,coeffs_topo,i_max,saving_dir,save_opt: Literal['all','total',None],region=None,
             mode: Literal['layer','interface'] ='layer',load_opt=False,layers_name=[],plot_opt=False,proj_opt=ccrs.Mollweide(), verbose_opt=False):

    """
    Usage
    ----------
    Synthetic generation of spherical harmonics coefficients (SHGravCoeffs/SHCoeffs) from an interiors model
    This function generates synthetic spherical harmonics coefficients based on a given interior model 
    of a planetary body. It calculates the gravitational coefficients for each layer of the model and
    combines them to produce the total gravitational field. 


    Parameters
    -----------
    param_int   : list
                  A list containing interior model parameters:
                  [0] - rho_layers: list of average densities for each layer    [kg/m^3]
                  [1] - radius_layers: list of radii for each layer interface   [km]
                  [2] - interface_type: list of interface types for each layer  [string]
                  [3] - interface_addinfo: additional info for RNG and custom option  
    n_max       : int
                  Maximum degree of spherical harmonics expansion
    coeffs_grav : pyshtools.SHGravCoeffs
                  Input gravitational coefficients
    coeffs_topo : list, pyshtools.SHCoeffs, [km]
                  Input topography coefficients for each layer interfrace. Used just for 'surface' interface
    i_max       : int
                  Maximum order used in Taylor series expansion for potential coefficients
    saving_dir  : str
                  Saving directory for the output files (data and images)
    region      : list, default = None
                  [[lon_min, lon_max], [lat_min, lat_max]] [degree]; region to display on the map.
                  If None, the global map is shown.
    save_opt    : str, option ['all','total',None]
                  - None, no saving
                  - 'all', save each layers gravitational and interface topography coefficients (+ interiors parameters)
                  - 'total', save just the global gravitational coefficients (+ interiors parameters)
    mode        : str, option ['layer','interface']
                  Generation mode:
                    - 'layer': generate coefficients related to each layer of the model (evaluation from bottom to top interfaces)
                    - 'interface': generate coefficients realted to each interface of the model (evaluation for the density jump on the interface)
    load_opt    : bool, default = False
                  If True, load previously calculated coefficients from files
                  IF False, generate new coefficients
    layer_name  : list, default = []
                  List of layers name for plotting title.
    plot_opt    : bool, default = False
                  Plotting Interface - U 
    proj_opt    : cartopy.crs, default = ccrs.Mollweide()
                  Map projection for plots
    verbose_opt : bool, default = False
                  Verbose option for progress and displaying sections

                  
    Returns
    --------
    coeffs_tot      : pyshtools.SHGravCoeffs
                      Total synthetic gravitational coefficients
    coeffs_list     : list
                      List of pyshtools.SHGravCoeffs objects for each layer
    """



    # Unpacking Interiors parameters
    rho_layers      = param_int[0]
    radius_layers   = param_int[1]
    interface_type  = param_int[2]
    interface_addinfo  = param_int[3]
    n_layers= np.size(rho_layers)


    if layers_name == []:
        layers_name = ['Layer '+str(i+1) for i in range(n_layers)]


    if not load_opt or os.path.isfile(saving_dir+"/coeffs_tot.dat") is False:

        surf_prec = pysh.SHGrid.from_zeros(lmax=n_max)
        theta = surf_prec.lats()* np.pi / 180.
        phi = surf_prec.lons()* np.pi / 180.
        delta_theta = -np.diff(theta)[0]
        delta_phi = np.diff(phi)[0]
        cos_theta = np.reshape(np.repeat(np.cos(theta),np.size(phi),axis=0),np.shape(surf_prec.data))

        M_layer=[]
        coeffs_list=[]
        coeffs_tot = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=coeffs_grav.gm, r0=coeffs_grav.r0)


        if plot_opt:
            fig, axs = plt.subplots(n_layers, 2, figsize =(9,9),subplot_kw={'projection': proj_opt})

        for i,interface in enumerate(interface_type):

            if verbose_opt:
                print("Generating layer: " + str(i+1) + '\n')
                print("Interface type: " + interface + '')
                print("Layer's radius: " + str(radius_layers[i]) + " km")
                print("Layer's density: " + str(rho_layers[i]) + " kg/m^3")

        
            # Evalute interface topography

            match interface:


                case 'sph':
                    surf = pysh.SHGrid.from_zeros(lmax=n_max)+radius_layers[i]


                case 'dwnbg':
                    surf = CrustThickness(coeffs_topo=coeffs_topo,coeffs_grav=coeffs_grav,rho_boug=rho_layers[i],
                                          delta_rho=rho_layers[i]-rho_layers[i+1],mean_radius=radius_layers[i],
                                          n_max=n_max,i_max=i_max,filter_opt=1,filter_deg=interface_addinfo[i],
                                          verbose_opt=verbose_opt)

                case 'surf':
                    surf = coeffs_topo.expand(lmax=n_max,extend=True)


                case 'sphflat':
                    # Flattening parameters
                    r_e_fact        = interface_addinfo[i][0]
                    r_p_fact        = interface_addinfo[i][1]
                    r_p = radius_layers[i]*r_p_fact
                    r_e = radius_layers[i]*r_e_fact

                    theta_grid = np.reshape(np.repeat(theta,np.size(phi),axis=0),np.shape(surf_prec.data))
                    surf = pysh.SHGrid.from_array(np.sqrt(r_p*r_p + (r_e*r_e -r_p*r_p)*np.cos(theta_grid)*np.cos(theta_grid)))
                    if verbose_opt:
                        print("Polar flattenig:")
                        print("     flattening f: " + str(np.round(1-r_p_fact/r_e_fact,4)))
                        print("     polar radius R_p: " + str(np.round(r_p,2)))
                        print("     equatorial radius R_e: " + str(np.round(r_e,2)))


                case 'rng':
                    # Kaula's rule + lowpass filtering (Ri/R ^n)
                    degrees = np.arange(n_max+1, dtype=float)
                    degrees[0] = np.inf
                    surf_coeff_rng = pysh.SHCoeffs.from_random(degrees**(-2) * (radius_layers[i]/radius_layers[-1])**(degrees),power_unit='per_lm', seed=41*i)
                    surf = surf_coeff_rng.expand(lmax=n_max,extend=True)+radius_layers[i]
                    surf_coeff_rng.coeffs *= interface_addinfo[i]/(np.max(surf.data) - np.min(surf.data))
                    surf = surf_coeff_rng.expand(lmax=n_max,extend=True)+radius_layers[i]
                    if verbose_opt:
                        print("RNG:")
                        print("     Delta H: " + str(interface_addinfo[i]) + 'km')


                case 'custom':
                    # Reading topography from file:
                    surf = pysh.SHGrid.from_file(interface_addinfo[i])
                    if surf.lmax != n_max:
                        print("ERROR: No resolution matching! (Custom interface file has a different lmax)")
                        exit()
                    if verbose_opt:
                        print("Custom datafile:")
                        print("     file: " + interface_addinfo[i])



                case _:
                    print("ERROR: No interface type recognized")
                    sys.exit()



            # ------------------------------------------------------------------------------------------------------

            # Check for topography conflicts (between current layer and the previous one):

            if ((surf.data - surf_prec.data)<= 0).any():
                print("ERROR: Interface Conflict")
                print("Layer "+str(i+1)+"-th lower than "+str(i)+"-th\n")

                if plot_opt:
                    topog_conflict = pysh.SHGrid.from_array((surf.data-surf_prec.data) <= 0)

                    # [fig_confl,ax_confl] = MapPlotting(values=topog_conflict.data, region=region, proj_opt=proj_opt, title=r'Interface conflict Layer '+str(i+1)+'-th - '+str(i)+'-th',cmap=cmap)
                    # MapPlotting([fig, axs[i, 0]], surf.data, region=region, proj_opt=proj_opt, title='Interface '+str(i+1), cb_label='$km$',cmap=cmap)
                    topog_conflict.plot(projection=proj_opt, title='Interface conflict Layer '+str(i+1)+'-th - '+str(i)+'-th', colorbar='right')
                    surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title='Interface '+str(i+1), cb_label='$km$',cmap=cmap)
                return None,None           


            # ------------------------------------------------------------------------------------------------------

            # Layer's mass:
            M_layer.append((1e+9*(surf.data**3 - surf_prec.data**3)  * rho_layers[i]/3 * cos_theta * delta_theta * delta_phi).sum())
            if verbose_opt: print("Layer's mass: " + str(M_layer[-1]) + " kg\n")

            # ------------------------------------------------------------------------------------------------------
            # ------------------------------------------------------------------------------------------------------
            

            # Synthetic gravitational coefficients generation:


            # ------------------------------------------------------------------------------------------------------
            # Layer generation:

            if mode == 'layer':

                # inizializing layer coefficients to zero
                coeffs_i = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i]*1e+3, gm=G_const*M_layer[i])


                # First layer
                if i == 0:
                    if interface_type[i] != 'sph':
                        coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                    lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                # Other layers
                else:
                    # Bottom interface
                    if interface_type[i-1] == 'sph':
                        coeffs_bot = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i-1]*1e+3, gm=coeffs_grav.gm)
                    else:
                        coeffs_bot = pysh.SHGravCoeffs.from_shape(shape=surf_prec*1e+3, rho=rho_layers[i],
                                                                lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                    # Top interface
                    if interface_type[i] == 'sph':
                        coeffs_top = pysh.SHGravCoeffs.from_zeros(lmax=n_max, r0=radius_layers[i]*1e+3, gm=coeffs_grav.gm)
                    else:
                        coeffs_top = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                    coeffs_top = coeffs_top.change_ref(r0=coeffs_grav.r0,gm=coeffs_grav.gm)
                    coeffs_bot = coeffs_bot.change_ref(r0=coeffs_grav.r0,gm=coeffs_grav.gm)
                    coeffs_i = coeffs_top - coeffs_bot
                    
                coeffs_i = coeffs_i.change_ref(r0=coeffs_grav.r0,gm=coeffs_grav.gm)
                coeffs_i.name = layers_name[i]+" (layer)"


            # ------------------------------------------------------------------------------------------------------
            # Interface generation   
            else: 

                if interface == 'surf' or i==n_layers-1:
                    coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i],
                                                                        lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)
                elif interface == 'sph':
                    coeffs_i = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=coeffs_grav.gm, r0=radius_layers[i]*1e+3)                                                              
                else:
                    coeffs_i = pysh.SHGravCoeffs.from_shape(shape=surf*1e+3, rho=rho_layers[i]-rho_layers[i+1],
                                                                        lmax=n_max, nmax=i_max, gm=coeffs_grav.gm)

                coeffs_i = coeffs_i.change_ref(r0=coeffs_grav.r0,gm=coeffs_grav.gm)
                coeffs_i.name = layers_name[i]+" (interface)"
            

            # ------------------------------------------------------------------------------------------------------

            # Save coefficients (gravity and topography)
            if save_opt == 'all':
                if mode=='layer':coeffs_i.to_file(saving_dir+"/coeffs_layer"+str(i+1)+".dat")
                if mode=='interface':coeffs_i.to_file(saving_dir+"/coeffs_interface"+str(i+1)+".dat")
                surf.to_file(saving_dir+"/interface"+str(i+1)+".dat")



            # ------------------------------------------------------------------------------------------------------
            if plot_opt:
                U_layer_matrix = coeffs_i.expand(lmax=n_max,r=coeffs_grav.r0,extend=True)
                U_layer_matrix_min = coeffs_i.expand(lmax=n_max,lmax_calc=1,r=coeffs_grav.r0,extend=True)
                U_layer = U_layer_matrix.pot - U_layer_matrix_min.pot

                # MapPlotting([fig, axs[i,1]], U_layer.data, region=region, proj_opt=proj_opt, title=coeffs_i.name, cb_label=r'$m^2/s^2$',cmap=cmap)
                # MapPlotting([fig, axs[i,0]], surf.data, region=region, proj_opt=proj_opt, title='Interface '+str(i+1), cb_label=r'$km$',cmap=cmap)
                U_layer.plot(ax=axs[i,1], colorbar='right',projection=proj_opt, title='$U_'+str(i+1)+'$', cb_label='$m^2/s^2$',cmap=cmap)
                surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title= layers_name[i] + ' $h_'+str(i+1)+'$', cb_label='$km$',cmap=cmap)
                plt.tight_layout()
                plt.show()
                if save_opt == 'all':
                    fig.savefig(saving_dir+"/Interface_U.png", dpi=600)

            # ------------------------------------------------------------------------------------------------------

            # Summing the coefficients:
            coeffs_list.append(coeffs_i)
            coeffs_tot += coeffs_i

            surf_prec = surf
            if verbose_opt: print("\n")
            gc.collect()
            FreeMemory()


            surf_prec = surf
            if verbose_opt: print("\n")

        coeffs_tot.name = 'Synthetic'
        if save_opt=='total' or save_opt=='all': coeffs_tot.to_file(saving_dir+"/coeffs_tot.dat")

        if verbose_opt:
            print("Synthetic Generation: DONE")
            # print("Total's mass: " + str(np.array(M_layer).sum()) + " kg\n")
            print("Synthetic coefficients saved in: " + saving_dir + '\n')
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")


    else:

        print("Loading Synthetic coefficients...\n")

        # Loading coefficients:
        coeffs_list=[]

        if plot_opt:
            fig, axs = plt.subplots(n_layers, 2, figsize =(9,9),subplot_kw={'projection': proj_opt})

        for i in range(0,n_layers):
            if mode=='layer': coeffs_i =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_layer"+str(i+1)+".dat", format="shtools", lmax=n_max, header=True, name=layers_name[i]+" (layer)")
            if mode=='interface': coeffs_i =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_interface"+str(i+1)+".dat", format="shtools", lmax=n_max, header=True, name=layers_name[i]+" (interface)")
            coeffs_list.append(coeffs_i)

            surf =  pysh.SHGrid.from_file(saving_dir+"/interface"+str(i+1)+".dat")

            U_i_matrix = coeffs_i.expand(lmax=n_max,r=coeffs_grav.r0,extend=True)
            U_i_matrix_min = coeffs_i.expand(lmax=n_max,lmax_calc=1,r=coeffs_grav.r0,extend=True)
            U_i = U_i_matrix.pot - U_i_matrix_min.pot

            if plot_opt:
                # MapPlotting([fig, axs[i,1]], U_i.data, region=region, proj_opt=proj_opt, title=coeffs_i.name, cb_label=r'$m^2/s^2$',cmap=cmap)
                # MapPlotting([fig, axs[i,0]], surf.data, region=region, proj_opt=proj_opt, title=r'Interface '+str(i+1), cb_label=r'$km$',cmap=cmap)
                U_i.plot(ax=axs[i,1], colorbar='right',projection=proj_opt, title='$U_'+str(i+1)+'$', cb_label='$m^2/s^2$',cmap=cmap)
                surf.plot(ax=axs[i,0], colorbar='right',projection=proj_opt, title=layers_name[i]+' $h_'+str(i+1)+'$', cb_label='$km$',cmap=cmap)
                plt.tight_layout()
                plt.show()

        coeffs_tot =  pysh.SHGravCoeffs.from_file(saving_dir+"/coeffs_tot.dat", format="shtools", lmax=n_max, header=True, name='Synthetic')
        if verbose_opt:
            print("Synthetic Generation Loaded")
            sub_dir = saving_dir.split("/")[-1]
            print("Synthetic coefficients from: " + sub_dir + '\n')
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")





    return coeffs_tot,coeffs_list



##########################################################################################################################
##########################################################################################################################
