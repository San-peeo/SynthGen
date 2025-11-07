from lib.lib_dep import *
from lib.misc.MapPlotting import *

def Global_Analysis(coeffs_grav, coeffs_topo, n_max, r, rho_boug, i_max, region=None, saving_dir=None,
                    plot_opt: Literal[None,'single','multiple'] = None, load_opt=False, proj_opt=ccrs.Mollweide(),n_min=3,
                    verbose_opt=False):


    """
    Usage
    ----------
    Evaluate gravity and topography coefficients (SHGravCoeffs/SHCoeffs) in spherical harmonics and produce values 
    and maps for:
    - gravity potential U   [m^2/s^2]
    - Topography H          [km]
    - Free-Air anomalies    [mGal]
    - Bouguer anomalies     [mGal]

    Maps are in .png files
    Data are in .dat files

    Parameters
    ----------
    coeffs_grav         : pyshtools.SHGravCoeffs
                          Gravitational coefficients
    coeffs_topo         : pyshtools.SHGravCoeffs, [km]
                          Topography coefficients
    r                   : float, [m]
                          Radius where the maps are to be evaluated or r = [a,f] for an ellipsoidal evaluation (a = semi-major axis, f = flattening =(b-a)/a)
    n_max               : int
                          The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    rho_boug            : float, [kg/m^3]
                          Crust density, used both in the Bouguer correction  
    i_max               : int
                          The maximum order used in the Taylor-series expansion when calculating the potential coefficients (Bouguer correction).
    region              : list, default = None
                          [[lon_min, lon_max], [lat_min, lat_max]] [degree]; region to display on the map.
                          If None, the global map is shown.
    saving_dir          : str, default = None
                          Saving directory for the output files (data and images).
                          If None, no files are saved.
    plot_opt            : str, options = [None,'single','multiple'], default = 'multiple
                          None = no plotting
                          single = each product has its own map
                          multiple = all of the products are plotted in a single map together
    load_opt            : bool, default = False
                          If True, load previously calculated matrices
                          If False, evaluate the products and save/overwrite them
    n_min               : int, default=3
                          The minimum spherical harmonic degree of the output spherical harmonic coefficients (GRAVITY).
    proj_opt            : cartopy.crs, default = ccrs.Mollweide()
                          Map projection for plots 
    verbose_opt         : bool, default = False
                          Verbose option for progress and displaying sections


    Output
    ----------
    U_matrix        : pyshtools.SHGrid
                      Gravitational Potential matrix SHGrid
    topog_matrix    : pyshtools.SHGrid
                      Topography matrix SHGrid
    deltag_freeair  : pyshtools.SHGrid
                      Free-Air anomalies matrix SHGrid
    deltag_boug     : pyshtools.SHGrid
                      Bouguer anomalies matrix SHGrid
          
    """

    # Global analysis:

    if verbose_opt:
        print("# ------------------------------------------------------------------------------------------------------\n")
        print("\n")
        print("Global analysis:")
        print("\n")


    # ------------------------------------------------------------------------------------------------------


    # Compute Gravity (U - FreeAir)
    if len(r)==1: grav = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,r=r,extend=True)
    else: grav = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,a=r[0],f=r[1],extend=True)
    if n_min>=0: 
        if len(r)==1: grav_min = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,lmax_calc=n_min,r=r,extend=True)
        else: grav_min = pysh.SHGravCoeffs.expand(coeffs_grav,lmax=n_max,lmax_calc=n_min,a=r[0],f=r[1],extend=True)


    # ------------------------------------------------------------------------------------------------------


    # Gravitational Potential  (m^2/s^2)
    if verbose_opt: print("Gravitational Potential:")

    if load_opt and os.path.isfile(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        U_matrix = np.loadtxt(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        U_matrix = pysh.SHGrid.from_array(U_matrix)
    else:
        if n_min>=0: 
            U_matrix = grav.pot - grav_min.pot
        else: U_matrix = grav.pot
        
        if saving_dir is not None: U_matrix.to_file(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")

    if plot_opt=='single':
        [fig,ax] = MapPlotting(values=U_matrix, region=region, proj_opt=proj_opt, title=r'Gravitational Potential $U(\theta,\phi)$', cb_label='$m^2/s^2$',cmap=cmap)
        # fig, ax = U_matrix.plot(colorbar='right',projection=proj_opt,title=r'Gravitational Potential $U(\theta,\phi)$', cb_label='$m^2/s^2$',cmap=cmap)
        if saving_dir is not None: fig.savefig(saving_dir+"/U_matrix_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".png", dpi=600)

    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Topography (km)
    if verbose_opt: print("Topography:")

    if load_opt and os.path.isfile(saving_dir+"/topog_matrix.dat") is True:
        topog_matrix = np.loadtxt(saving_dir+"/topog_matrix.dat")
        topog_matrix = pysh.SHGrid.from_array(topog_matrix)
    else:
        topog_matrix = coeffs_topo.expand(lmax=n_max,extend=True)
        if n_min>=0:
            topog_matrix_min = coeffs_topo.expand(lmax=n_max,lmax_calc=n_min,extend=True)
            topog_matrix = topog_matrix-topog_matrix_min

        theta = topog_matrix.lats()
        phi = topog_matrix.lons()
        if saving_dir is not None:
            np.savetxt(saving_dir+"/theta.dat",theta)
            np.savetxt(saving_dir+"/phi.dat",phi)
            topog_matrix.to_file(saving_dir+"/topog_matrix.dat")

    if plot_opt=='single':
        [fig,ax] = MapPlotting(values=topog_matrix, region=region, proj_opt=proj_opt, title=r'Topography $h(\theta,\phi)$', cb_label='$km$',cmap=cmap)
        # fig, ax = topog_matrix.plot(colorbar='right',projection=proj_opt, cb_label='km',cmap=cmap)
        if saving_dir is not None: fig.savefig(saving_dir+"/topog_matrix.png", dpi=600)

    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Gravitational Accelleration (Free-Air anomalies)  (mGal)
    if verbose_opt: print("Free-Air anomalies:")


    if load_opt and os.path.isfile(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        deltag_freeair = np.loadtxt(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        deltag_freeair = pysh.SHGrid.from_array(deltag_freeair)
    else:
        if n_min>=0: 
            deltag_freeair = grav.total*1e+5  - grav_min.total*1e+5  
        else: deltag_freeair = grav.total*1e+5
         
        if saving_dir is not None: deltag_freeair.to_file(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")

    if plot_opt=='single':
        [fig,ax] = MapPlotting(values=deltag_freeair, region=region, proj_opt=proj_opt, title=r'Free-Air anomalies $\frac{\partial U}{\partial r}(\theta,\phi)$', cb_label='$mGal$',cmap=cmap)
        # fig, ax = deltag_freeair.plot(colorbar='right',projection=proj_opt, cb_label='mGal',cmap=cmap)
        if saving_dir is not None: fig.savefig(saving_dir+"/deltag_freeair_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".png", dpi=600)


    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------


    # Bouguer anomalies
    if verbose_opt: print("Bouguer anomalies:")

    if load_opt and os.path.isfile(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat") is True:
        deltag_boug = np.loadtxt(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        deltag_boug = pysh.SHGrid.from_array(deltag_boug)
    else:
        # NB: topography coeff in m
        bouger_correction = pysh.SHGravCoeffs.from_shape(shape=coeffs_topo*1e+3, rho=rho_boug, gm=coeffs_grav.gm, lmax=n_max, nmax=i_max) 
        bouger_correction = bouger_correction.change_ref(r0=coeffs_grav.r0)

        bouguer_coeff = coeffs_grav - bouger_correction
        if len(r)==1: bouguer = bouguer_coeff.expand(lmax=n_max,r=r,extend=True)
        else: bouguer = bouguer_coeff.expand(lmax=n_max,a=r[0],f=r[1],extend=True)
        if n_min>=0:
            if len(r)==1: bouguer_min = bouguer_coeff.expand(lmax=n_max,lmax_calc=n_min,r=r,extend=True)
            else: bouguer_min = bouguer_coeff.expand(lmax=n_max,lmax_calc=n_min,a=r[0],f=r[1],extend=True)
            deltag_boug = bouguer.total*1e+5 - bouguer_min.total*1e+5
        else:
            deltag_boug = bouguer.total*1e+5
        if saving_dir is not None: deltag_boug.to_file(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".dat")
        

    if plot_opt=='single':
        [fig,ax] = MapPlotting(values=deltag_boug, region=region, proj_opt=proj_opt, title=r'Bouguer anomalies $Boug(\theta,\phi)$', cb_label='$mGal$',cmap=cmap)
        # fig, ax = deltag_boug.plot(colorbar='right', projection=proj_opt, cb_label='mGal',cmap=cmap)
        if saving_dir is not None: fig.savefig(saving_dir+"/deltag_boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".png", dpi=600)


    if verbose_opt: 
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")
    # ------------------------------------------------------------------------------------------------------

    # Multiple plot

    if plot_opt=='multiple':

        if verbose_opt: 
            print(" ")
            print("Plotting:")

        fig, axs = plt.subplots(2, 2, figsize=(13, 7),subplot_kw={'projection': proj_opt})
        MapPlotting([fig, axs[0, 0]], U_matrix, region=region, proj_opt=proj_opt, title=r'Gravitational Potential $U(\theta,\phi)$', cb_label='$m^2/s^2$',cmap=cmap)
        MapPlotting([fig, axs[0, 1]], topog_matrix, region=region, proj_opt=proj_opt, title=r'Topography $h(\theta,\phi)$', cb_label='$km$',cmap=cmap)
        MapPlotting([fig, axs[1, 0]], deltag_freeair, region=region, proj_opt=proj_opt, title=r'Free-Air anomalies $\frac{dU}{dr}(\theta,\phi)$', cb_label='$mGal$',cmap=cmap)
        MapPlotting([fig, axs[1, 1]], deltag_boug, region=region, proj_opt=proj_opt,title=r'Bouguer anomalies $Boug(\theta,\phi)$', cb_label='$mGal$',cmap=cmap)

        plt.tight_layout()
        plt.show()
        if saving_dir is not None: fig.savefig(saving_dir+"/U_h_FreeAir_Boug_nmin"+str(n_min+1)+"_nmax"+str(n_max)+".png", dpi=600)

        if verbose_opt:
            print("Done")
            print(" ")
            print(" ")
            print("# ------------------------------------------------------------------------------------------------------\n")


    return U_matrix, topog_matrix, deltag_freeair, deltag_boug

##########################################################################################################################
##########################################################################################################################
