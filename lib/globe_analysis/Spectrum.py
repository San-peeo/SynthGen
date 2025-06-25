from lib.lib_dep import *

def Spectrum(coeffs,n_max,saving_dir=None,save_opt: Literal['all','total',None] = None,n_min=2, load_opt=False,plot_opt=False,verbose_opt=False):

    """
    Usage
    ----------
    Evaluate spectrum of spherical harmonics coefficients (SHGravCoeffs/SHCoeffs)


    Parameters
    ----------
    coeffs          : pyshtools.SHGravCoeffs
                      Gravitational coefficients (single set or list)
                      Note: for list, remember to set the name (coeff.name) for each element to visualize correctly the legend
    n_max           : int
                      The maximum spherical harmonic degree of the output spherical harmonic coefficients.
    n_min           : int, default = 2
                      The minimum spherical harmonic degree of the output spherical harmonic coefficients.
    saving_dir      : str,
                      Saving directory for the output files (data and images)
    save_opt        : str, option ['all','total',None]
                    - None, no saving
                    - 'all', save each layers spectrum
                    - 'total', save just the global spectrum
    load_opt        : bool, default = False
                      If True, load previously calculated spectrum from files
                      IF False, evaluated the spectrum ex novo
    plot_opt        : bool, default = False
                      Plotting spectrum
    verbose_opt     : bool, default = False
                      Verbose option for progress and displaying sections


    Output
    ---------
    spectrum        : list (numpy.ndarray)
                      List of numpy arrays containing the spectrum An = sqrt((Cnm^2 + Snm^2)/(2n+1)) for all of the layers
                      and the total one. Spherical layers have zero effect.
    fig             : matplotlib.figure.Figure
                      Figure object containing the plot of the spectrum if plot_opt is True
    """


    # Gravitational Spectrum
    if verbose_opt: print("Gravitational Spectrum:")


    if plot_opt:
        fig = plt.figure(figsize=(9,4), constrained_layout=True)
    else:
        fig = None
        

    degree_grav = np.arange(0,n_max+1)


    spectrum=[]
    for coeff in coeffs:

        if load_opt and os.path.isfile(saving_dir+"/spectrum_grav"+coeff.name+".dat") is True:
            spectrum_grav = np.loadtxt(saving_dir+"/spectrum_grav"+coeff.name+".dat")
            if plot_opt:
                plt.plot(degree_grav[n_min:],np.sqrt(spectrum_grav[n_min:]), linewidth=2, label=coeff.name)

        else:
            if coeff.coeffs[0][3,0] != 0:   # some topopgrahy (at least polar flattening)
                # WARNING: use or convention='power',unit='per_lm' or convention='l2norm',unit='per_l' + divide by (2*degree+1)
                spectrum_grav = pysh.spectralanalysis.spectrum(coeff.coeffs,convention='l2norm',unit='per_l',lmax=n_max)
                spectrum_grav /= (2*degree_grav+1) # only for 'l2norm'
                if save_opt == 'all': np.savetxt(saving_dir+"/spectrum_grav_"+coeff.name+".dat",spectrum_grav)
                if save_opt == 'total' and ("Layer" in coeff.name) is False: np.savetxt(saving_dir+"/spectrum_grav_"+coeff.name+".dat",spectrum_grav)
                if plot_opt:
                    plt.plot(degree_grav[n_min:],np.sqrt(spectrum_grav[n_min:]), linewidth=2, label=coeff.name)

        spectrum.append(np.sqrt(spectrum_grav))
      

    if plot_opt:
        ymin = np.min(np.sqrt(spectrum_grav[n_min:]))
        ymax = np.max(np.sqrt(spectrum_grav[n_min:]))
        plt.ylim([10**-np.ceil(-np.log10(ymin)), 10**-np.ceil(-np.log10(ymax)-1)])
        plt.xlim([0, n_max])
        plt.yscale("log")
        plt.xlabel("degree $l$")
        plt.ylabel("Power Spectrum")
        plt.xticks(np.arange(0,n_max+10 ,10))
        plt.legend()
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
        plt.grid(visible=True, which='minor', linestyle='--', linewidth=0.2)
        plt.show()
        if save_opt is not None: plt.savefig(saving_dir+"/spectrum_grav.png", dpi=600, bbox_inches='tight')


    if verbose_opt:
        print("Done")
        print(" ")
        print(" ")
        print("# ------------------------------------------------------------------------------------------------------\n")


    return spectrum,fig



##########################################################################################################################
##########################################################################################################################
