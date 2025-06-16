from lib.lib_dep import *

def CrustThickness(coeffs_topo,coeffs_grav,rho_boug,delta_rho,mean_radius,n_max,i_max, filter_opt,filter_deg=None,verbose_opt=False):


    """
    Usage
    ----------
    Calculate the thickness of the crust based on Bouguer Anomaly (see Wieczorek and Phillips, 1998).
    This function uses the pyshtools library BAtoHilmDH method (https://shtools.github.io/SHTOOLS/pybatohilmdh.html).

    
    Parameters
    -----------

    coeffs_grav     : pyshtools.SHGravCoeffs 
                      Gravitational coefficients
    coeffs_topo     : pyshtools.SHCoeffs
                      Topography coefficients (in kilometers)
    rho_boug        : float, [kg/m^3]
                      Crust density; used both in the Bouguer correction and the crust thickness calculation 
    delta_rho       : float, [kg/m^3]
                      Density contrast between Mantle and Crust layers
    mean_radius     : float, [km]
                      Mean radius of the mantle-crust interface (=mean crustal thickness) 
    n_max           : int
                      The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    i_max           : int
                      The maximum order used in the Taylor-series expansion when calculating the potential
                      coefficients (Bouguer correction and Crustal thickness).
    filter_opt      : int, option=[0,1,2]
                      Type of filter to apply in the conversion: if 0, no filtering is applied.
                                                                 if 1, DownContFilterMA (minimum amplitude filter)  eq. 19, Wieczorek and Phillips, 1998
                                                                 if 2, DownContFilterMC (minimum curvature filter)
    verbose_opt     : bool, default = False
                      Verbose option for progress and displaying sections

    Output:
    --------
    mantle_crust_relief : The calculated relief of the mantle-crust interface (ie. crust thickness), as SHGrid.

    """

    # Crustal thickness calculation
    if verbose_opt: 
        print("Crust Thickness:")
        print('     Delta_rho: ' + str(delta_rho))
    

    # Bouguer coefficient evaluation
    # NB: shape needs to be in m, so coeffs_topo*1e+3 (accordance to GM and rho units)
    bouger_correction = pysh.SHGravCoeffs.from_shape(shape=coeffs_topo*1e+3, rho=rho_boug, gm=coeffs_grav.gm, lmax=n_max, nmax=i_max)   
    bouger_correction = bouger_correction.change_ref(r0=coeffs_grav.r0)
    coeff_boug = coeffs_grav - bouger_correction

    # Mean interface grid
    mean_interface = pysh.SHGrid.from_zeros(lmax=n_max)+mean_radius*1e+3


    # Mantle - Crust interface evaluation
    if filter_deg is None:
        filter_deg = n_max/2
    if verbose_opt: print('     Cutting filter (n_half): '+ str(filter_deg))

    mantle_crust_coeffs = pysh.gravmag.BAtoHilmDH(ba=coeff_boug.coeffs,griddh=mean_interface.data,nmax=i_max,lmax=n_max,mass=coeffs_grav.gm/G_const,r0=coeff_boug.r0,rho=delta_rho,filter_type=filter_opt,filter_deg=filter_deg)
    mantle_crust_relief= pysh.SHGrid.from_array(pysh.expand.MakeGridDH(mantle_crust_coeffs/1e+3,lmax=n_max,sampling=2,extend=True))


    return mantle_crust_relief


##########################################################################################################################
##########################################################################################################################

