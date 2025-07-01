from lib.lib_dep import *
from lib.utils.FreeMemory import *
from lib.globe_analysis.Global_Analysis import *
from lib.globe_analysis.Spectrum import *


def MetricsAnalysis(metrics_list, load_opt, models_dir, real_matrix, input_param, plot_opt=False):

    """
    Usage
    ----------
    Load/Compute input metrics for comparing real and synthetic gravity models.
    The metrics are normalized between 0 and 1.

    
    Parameters
    ----------
    metrics_list : list
                   List of metrics to compute. Options include:
                   - "Delta_mean": Mean difference
                   - "Delta_std": Standard deviation of the difference
                   - "RMSE": Root Mean Squared Error
                   - "R^2": Coefficient of determination
                   - "SSIM": Structural Similarity Index
                   - "spectrum": Spectrum ratio
    load_opt     : bool
                   If True, load precomputed metrics from files.
                   If False, compute metrics from the GRID models (see models_dir).
    models_dir   : str or list (of str)
                   Directory/List of directories containing GRID subdirectories with synthetic model data.
    real_matrix  : list
                   List of real data matrices:
                   - [0]: U_matrix_real (Gravitational potential)
                   - [1]: deltag_freeair_real (Free-Air anomalies)
                   - [2]: deltag_boug_real (Bouguer anomalies)
                   - [3]: coeffs_topo (Topography coefficients)
                   - [4]: spectrum_real (Gravity spectrum)
    input_param  : list
                   List of input parameters:
                   - [0]: n_min (Minimum spherical harmonic degree)
                   - [1]: n_max (Maximum spherical harmonic degree)
                   - [2]: r (Radius for evaluation)
                   - [3]: i_max (Maximum order for Taylor expansion)
                   - [4]: rho_boug (Crust density for Bouguer correction)
    plot_opt     : bool,
                    If True, plot histograms of the computed metrics.

    Output
    ----------
    metrics : numpy.ndarray
              Array containing the computed metrics for the specified metrics_list.
    """


    # ------------------------------------------------------------------------------------------------------


    U_matrix_real       = real_matrix[0]
    deltag_freeair_real = real_matrix[1]
    deltag_boug_real    = real_matrix[2]
    coeffs_topo         = real_matrix[3]
    spectrum_real       = real_matrix[4]

    n_min    = input_param[0]
    n_max    = input_param[1]
    r        = input_param[2]
    i_max    = input_param[3]
    rho_boug = input_param[4]


    # ------------------------------------------------------------------------------------------------------

    # Metrics
    if "Delta_mean" in metrics_list:
        delta_U_mean_arr                = []
        delta_FreeAir_mean_arr          = []
        delta_Bouguer_mean_arr          = []
    if "Delta_std" in metrics_list:
        delta_U_std_arr                 = []
        delta_FreeAir_std_arr           = []
        delta_Bouguer_std_arr           = []
    if "RMSE" in metrics_list:
        RMSE_U_arr                      = []
        RMSE_FreeAir_arr                = []
        RMSE_Bouguer_arr                = []
    if "MAE" in metrics_list:
        MAE_U_arr                       = []
        MAE_FreeAir_arr                 = []
        MAE_Bouguer_arr                 = []
    if "R^2" in metrics_list:
        R2_U_arr                        = []
        R2_FreeAir_arr                  = []
        R2_Bouguer_arr                  = []
    if "SSIM" in metrics_list:
        ssim_U_arr                      = []
        ssim_FreeAir_arr                = []
        ssim_Bouguer_arr                = []
    if "PSNR" in metrics_list:
        psnr_U_arr                      = []
        psnr_FreeAir_arr                = []
        psnr_Bouguer_arr                = []
    if "NCC" in metrics_list:
        ncc_U_arr                       = []
        ncc_FreeAir_arr                 = []
        ncc_Bouguer_arr                 = []
    # if "spectrum" in metrics_list:
    #     spectrum_ratio_arr              = []

    # Grid parameters
    rho_rng_arr     = []
    radius_rng_arr  = []
    nhalf_rng_arr   = []



    # ------------------------------------------------------------------------------------------------------


    for dir in models_dir:
        print("Analysing directory: ", dir)
        FreeMemory()
        # Loop over the sub-directories:
        for counter, subdir in tqdm(enumerate(os.listdir(dir)), total=len(os.listdir(dir))):

            # memory leak issues:
            if counter % 100 == 0:
                FreeMemory()



            if load_opt:

                # Load the metrics:
                if "Delta_mean" in metrics_list:
                    delta_U_mean = np.loadtxt(dir+subdir+'/delta_U_mean.dat')
                    delta_U_mean_arr.append(delta_U_mean)
                    delta_FreeAir_mean = np.loadtxt(dir+subdir+'/delta_FreeAir_mean.dat')
                    delta_FreeAir_mean_arr.append(delta_FreeAir_mean)
                    delta_Bouguer_mean = np.loadtxt(dir+subdir+'/delta_Bouguer_mean.dat')
                    delta_Bouguer_mean_arr.append(delta_Bouguer_mean)
                if "Delta_std" in metrics_list:
                    delta_U_std = np.loadtxt(dir+subdir+'/delta_U_std.dat')
                    delta_U_std_arr.append(delta_U_std)
                    delta_FreeAir_std = np.loadtxt(dir+subdir+'/delta_FreeAir_std.dat')
                    delta_FreeAir_std_arr.append(delta_FreeAir_std)
                    delta_Bouguer_std = np.loadtxt(dir+subdir+'/delta_Bouguer_std.dat')
                    delta_Bouguer_std_arr.append(delta_Bouguer_std)
                if "RMSE" in metrics_list:
                    RMSE_U = np.loadtxt(dir+subdir+'/RMSE_U.dat')
                    RMSE_U_arr.append(RMSE_U)
                    RMSE_FreeAir = np.loadtxt(dir+subdir+'/RMSE_FreeAir.dat')
                    RMSE_FreeAir_arr.append(RMSE_FreeAir)
                    RMSE_Bouguer = np.loadtxt(dir+subdir+'/RMSE_Bouguer.dat')
                    RMSE_Bouguer_arr.append(RMSE_Bouguer)
                if "MAE" in metrics_list:
                    MAE_U = np.loadtxt(dir+subdir+'/MAE_U.dat')
                    MAE_U_arr.append(MAE_U)
                    MAE_FreeAir = np.loadtxt(dir+subdir+'/MAE_FreeAir.dat')
                    MAE_FreeAir_arr.append(MAE_FreeAir)
                    MAE_Bouguer = np.loadtxt(dir+subdir+'/MAE_Bouguer.dat')
                    MAE_Bouguer_arr.append(MAE_Bouguer)
                if "R^2" in metrics_list:
                    R2_U = np.loadtxt(dir+subdir+'/R2_U.dat')
                    R2_U_arr.append(R2_U)
                    R2_FreeAir = np.loadtxt(dir+subdir+'/R2_FreeAir.dat')
                    R2_FreeAir_arr.append(R2_FreeAir)
                    R2_Bouguer = np.loadtxt(dir+subdir+'/R2_Bouguer.dat')
                    R2_Bouguer_arr.append(R2_Bouguer)
                if "SSIM" in metrics_list:
                    ssim_U = np.loadtxt(dir+subdir+'/ssim_U.dat')
                    ssim_U_arr.append(ssim_U)
                    ssim_FreeAir = np.loadtxt(dir+subdir+'/ssim_FreeAir.dat')
                    ssim_FreeAir_arr.append(ssim_FreeAir)
                    ssim_Bouguer = np.loadtxt(dir+subdir+'/ssim_Bouguer.dat')
                    ssim_Bouguer_arr.append(ssim_Bouguer)
                if "PSNR" in metrics_list:
                    psnr_U = np.loadtxt(dir+subdir+'/psnr_U.dat')
                    psnr_U_arr.append(psnr_U)
                    psnr_FreeAir = np.loadtxt(dir+subdir+'/psnr_FreeAir.dat')
                    psnr_FreeAir_arr.append(psnr_FreeAir)
                    psnr_Bouguer = np.loadtxt(dir+subdir+'/psnr_Bouguer.dat')
                    psnr_Bouguer_arr.append(psnr_Bouguer)
                if "NCC" in metrics_list:
                    ncc_U = np.loadtxt(dir+subdir+'/ncc_U.dat')
                    ncc_U_arr.append(ncc_U)
                    ncc_FreeAir = np.loadtxt(dir+subdir+'/ncc_FreeAir.dat')
                    ncc_FreeAir_arr.append(ncc_FreeAir)
                    ncc_Bouguer = np.loadtxt(dir+subdir+'/ncc_Bouguer.dat')
                    ncc_Bouguer_arr.append(ncc_Bouguer)
                # if "spectrum" in metrics_list:
                #     spectrum_ratio = np.loadtxt(dir+subdir+'/spectrum_ratio.dat')
                #     spectrum_ratio_arr.append(spectrum_ratio)





            else:

                # Loading SynthGen coefficients:
                coeffs_tot=pysh.SHGravCoeffs.from_file(dir+subdir+'/coeffs_tot.dat')


                # Global analysis (U, H, FreeAir, Bouguer):
                U_matrix,_,deltag_freeair,deltag_boug=Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,
                                                                    r=r,rho_boug=rho_boug,
                                                                    i_max=i_max,plot_opt=None,load_opt=False,verbose_opt=False)
                # # Spectrum analysis:
                # spectrum_synth = Spectrum(coeffs=[coeffs_tot],n_min=n_min,n_max=n_max,
                #                             save_opt=None,load_opt=load_opt,verbose_opt=False)


                # Evaluate metrics:
                if "Delta_mean" in metrics_list:
                    delta_U_mean = np.mean(U_matrix_real-U_matrix.data)
                    np.savetxt(dir+subdir+'/delta_U_mean.dat',[delta_U_mean])
                    delta_U_mean_arr.append(delta_U_mean)
                    delta_FreeAir_mean = np.mean(deltag_freeair_real-deltag_freeair.data)
                    np.savetxt(dir+subdir+'/delta_FreeAir_mean.dat',[delta_FreeAir_mean])
                    delta_FreeAir_mean_arr.append(delta_FreeAir_mean)
                    delta_Bouguer_mean = np.mean(deltag_boug_real-deltag_boug.data)
                    np.savetxt(dir+subdir+'/delta_Bouguer_mean.dat',[delta_Bouguer_mean])
                    delta_Bouguer_mean_arr.append(delta_Bouguer_mean)

                if "Delta_std" in metrics_list:
                    delta_U_std = np.std(U_matrix_real-U_matrix.data)
                    np.savetxt(dir+subdir+'/delta_U_std.dat',[delta_U_std])
                    delta_U_std_arr.append(delta_U_std)
                    delta_FreeAir_std = np.std(deltag_freeair_real-deltag_freeair.data)
                    np.savetxt(dir+subdir+'/delta_FreeAir_std.dat',[delta_FreeAir_std])
                    delta_FreeAir_std_arr.append(delta_FreeAir_std)
                    delta_Bouguer_std = np.std(deltag_boug_real-deltag_boug.data)
                    np.savetxt(dir+subdir+'/delta_Bouguer_std.dat',[delta_Bouguer_std])
                    delta_Bouguer_std_arr.append(delta_Bouguer_std)

                if "RMSE" in metrics_list:
                    RMSE_U = sklearn.metrics.root_mean_squared_error(U_matrix_real,U_matrix.data)
                    np.savetxt(dir+subdir+'/RMSE_U.dat',[RMSE_U])
                    RMSE_U_arr.append(RMSE_U)
                    RMSE_FreeAir = sklearn.metrics.root_mean_squared_error(deltag_freeair_real,deltag_freeair.data)
                    np.savetxt(dir+subdir+'/RMSE_FreeAir.dat',[RMSE_FreeAir])
                    RMSE_FreeAir_arr.append(RMSE_FreeAir)
                    RMSE_Bouguer = sklearn.metrics.root_mean_squared_error(deltag_boug_real,deltag_boug.data)
                    np.savetxt(dir+subdir+'/RMSE_Bouguer.dat',[RMSE_Bouguer])
                    RMSE_Bouguer_arr.append(RMSE_Bouguer)

                if "MAE" in metrics_list:
                    MAE_U = np.mean(np.abs(U_matrix_real-U_matrix.data))
                    np.savetxt(dir+subdir+'/MAE_U.dat',[MAE_U])
                    MAE_U_arr.append(MAE_U)
                    MAE_FreeAir = np.mean(np.abs(deltag_freeair_real-deltag_freeair.data))
                    np.savetxt(dir+subdir+'/MAE_FreeAir.dat',[MAE_FreeAir])
                    MAE_FreeAir_arr.append(MAE_FreeAir)
                    MAE_Bouguer = np.mean(np.abs(deltag_boug_real-deltag_boug.data))
                    np.savetxt(dir+subdir+'/MAE_Bouguer.dat',[MAE_Bouguer])
                    MAE_Bouguer_arr.append(MAE_Bouguer)

                if "R^2" in metrics_list:
                    R2_U = Corr2_Edo(U_matrix_real.flatten(),U_matrix.data.flatten())
                    np.savetxt(dir+subdir+'/R2_U.dat',[R2_U])
                    R2_U_arr.append(R2_U)
                    R2_FreeAir = Corr2_Edo(deltag_freeair_real.flatten(),deltag_freeair.data.flatten())
                    np.savetxt(dir+subdir+'/R2_FreeAir.dat',[R2_FreeAir])
                    R2_FreeAir_arr.append(R2_FreeAir)
                    R2_Bouguer = Corr2_Edo(deltag_boug_real.flatten(),deltag_boug.data.flatten())
                    np.savetxt(dir+subdir+'/R2_Bouguer.dat',[R2_Bouguer])
                    R2_Bouguer_arr.append(R2_Bouguer)

                if "SSIM" in metrics_list:
                    ssim_U = skimage.metrics.structural_similarity(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min())
                    np.savetxt(dir+subdir+'/ssim_U.dat',[ssim_U])
                    ssim_U_arr.append(ssim_U)
                    ssim_FreeAir = skimage.metrics.structural_similarity(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min())
                    np.savetxt(dir+subdir+'/ssim_FreeAir.dat',[ssim_FreeAir])
                    ssim_FreeAir_arr.append(ssim_FreeAir)
                    ssim_Bouguer = skimage.metrics.structural_similarity(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min())
                    np.savetxt(dir+subdir+'/ssim_Bouguer.dat',[ssim_Bouguer])
                    ssim_Bouguer_arr.append(ssim_Bouguer)

                if "PSNR" in metrics_list:
                    psnr_U = skimage.metrics.peak_signal_noise_ratio(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min())
                    np.savetxt(dir+subdir+'/psnr_U.dat',[psnr_U])
                    psnr_U_arr.append(psnr_U)
                    psnr_FreeAir = skimage.metrics.peak_signal_noise_ratio(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min())
                    np.savetxt(dir+subdir+'/psnr_FreeAir.dat',[psnr_FreeAir])
                    psnr_FreeAir_arr.append(psnr_FreeAir)
                    psnr_Bouguer = skimage.metrics.peak_signal_noise_ratio(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min())
                    np.savetxt(dir+subdir+'/psnr_Bouguer.dat',[psnr_Bouguer])
                    psnr_Bouguer_arr.append(psnr_Bouguer)
                    
                if "NCC" in metrics_list:
                    U_matrix_real_mean = U_matrix_real - np.mean(U_matrix_real)
                    U_matrix_mean = U_matrix.data - np.mean(U_matrix.data)
                    ncc_U = np.sum(U_matrix_real_mean * U_matrix_mean) / (np.sqrt(np.sum(U_matrix_real_mean**2)) * np.sqrt(np.sum(U_matrix_mean**2)))
                    np.savetxt(dir+subdir+'/ncc_U.dat',[ncc_U])
                    ncc_U_arr.append(ncc_U)

                    deltag_freeair_real_mean = deltag_freeair_real - np.mean(deltag_freeair_real)
                    deltag_freeair_mean = deltag_freeair.data - np.mean(deltag_freeair.data)
                    ncc_FreeAir = np.sum(deltag_freeair_real_mean * deltag_freeair_mean) / (np.sqrt(np.sum(deltag_freeair_real_mean**2)) * np.sqrt(np.sum(deltag_freeair_mean**2)))
                    np.savetxt(dir+subdir+'/ncc_FreeAir.dat',[ncc_FreeAir])
                    ncc_FreeAir_arr.append(ncc_FreeAir)

                    deltag_boug_real_mean = deltag_boug_real - np.mean(deltag_boug_real)
                    deltag_boug_mean = deltag_boug.data - np.mean(deltag_boug.data)
                    ncc_Bouguer = np.sum(deltag_boug_real_mean * deltag_boug_mean) / (np.sqrt(np.sum(deltag_boug_real_mean**2)) * np.sqrt(np.sum(deltag_boug_mean**2)))
                    np.savetxt(dir+subdir+'/ncc_Bouguer.dat',[ncc_Bouguer])
                    ncc_Bouguer_arr.append(ncc_Bouguer)


                # if "spectrum" in metrics_list:
                #     spectrum_ratio.append(np.mean(spectrum_real/spectrum_synth))  



            # Store interiors parameters:
            rho_rng_arr.append(np.loadtxt(dir+subdir+'/rho_layers.dat'))
            radius_rng_arr.append(np.loadtxt(dir+subdir+'/radius_layers.dat')) 
            nhalf_rng_arr.append(np.loadtxt(dir+subdir+'/n_half.dat'))






   
    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------

    # Normalization of the metrics:
    

    if "Delta_mean" in metrics_list:
    #     delta_U_mean         = NormalizeData_MinMax(np.abs(delta_U_mean))
    #     delta_FreeAir_mean   = NormalizeData_MinMax(np.abs(delta_FreeAir_mean))
    #     delta_Bouguer_mean   = NormalizeData_MinMax(np.abs(delta_Bouguer_mean))
        delta_mean = np.vstack([delta_U_mean_arr, delta_FreeAir_mean_arr, delta_Bouguer_mean_arr])

    if "Delta_std" in metrics_list:
    #     delta_U_std          = NormalizeData_MinMax(np.abs(delta_U_std))
    #     delta_FreeAir_std    = NormalizeData_MinMax(np.abs(delta_FreeAir_std))
    #     delta_Bouguer_std    = NormalizeData_MinMax(np.abs(delta_Bouguer_std))
        delta_std = np.vstack([delta_U_std_arr, delta_FreeAir_std_arr, delta_Bouguer_std_arr])

    if "RMSE" in metrics_list:
    #     RMSE_U               = NormalizeData_MinMax(np.abs(RMSE_U))
    #     RMSE_FreeAir         = NormalizeData_MinMax(np.abs(RMSE_FreeAir))
    #     RMSE_Bouguer         = NormalizeData_MinMax(np.abs(RMSE_Bouguer))
    #     # RMSE_U                  = 1/(1+np.array(RMSE_U))
    #     # RMSE_FreeAir            = 1/(1+np.array(RMSE_FreeAir))
    #     # RMSE_Bouguer            = 1/(1+np.array(RMSE_Bouguer))
        RMSE = np.vstack([RMSE_U_arr, RMSE_FreeAir_arr, RMSE_Bouguer_arr])

        
    if "MAE" in metrics_list:
    #     MAE_U               = NormalizeData_MinMax(np.abs(MAE_U))
    #     MAE_FreeAir         = NormalizeData_MinMax(np.abs(MAE_FreeAir))
    #     MAE_Bouguer         = NormalizeData_MinMax(np.abs(MAE_Bouguer))
    #     # MAE_U               = 1/(1+np.array(MAE_U))
    #     # MAE_FreeAir         = 1/(1+np.array(MAE_FreeAir))
    #     # MAE_Bouguer         = 1/(1+np.array(MAE_Bouguer))
        MAE = np.vstack([MAE_U_arr, MAE_FreeAir_arr, MAE_Bouguer_arr])

    if "R^2" in metrics_list:
    #     # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
    #     # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
    #     # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
    #     R2_U                 = (np.array(R2_U) + 1)/2
    #     R2_FreeAir           = (np.array(R2_FreeAir) + 1)/2
    #     R2_Bouguer           = (np.array(R2_Bouguer) + 1)/2
        R2 = np.vstack([R2_U_arr, R2_FreeAir_arr, R2_Bouguer_arr])

    if "SSIM" in metrics_list:
    #     # ssim_U               = NormalizeData_MinMax(np.abs(ssim_U))
    #     # R2_FreeAir           = NormalizeData_MinMax(np.abs(ssim_FreeAir))
    #     # R2_Bouguer           = NormalizeData_MinMax(np.abs(ssim_Bouguer))
        SSIM = np.vstack([ssim_U_arr, ssim_FreeAir_arr, ssim_Bouguer_arr])

    if "PSNR" in metrics_list:
    #     # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
    #     # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
    #     # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
    #     psnr_U                 = np.array(psnr_U)/np.max(psnr_U)
    #     psnr_FreeAir           = np.array(psnr_FreeAir)/np.max(psnr_FreeAir)
    #     psnr_Bouguer           = np.array(psnr_Bouguer)/np.max(psnr_Bouguer)
        PSNR = np.vstack([psnr_U_arr, psnr_FreeAir_arr, psnr_Bouguer_arr])

    if "NCC" in metrics_list:
    #     # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
    #     # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
    #     # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
    #     ncc_U                 = (np.array(ncc_U) + 1)/2
    #     ncc_FreeAir           = (np.array(ncc_FreeAir) + 1)/2
    #     ncc_Bouguer           = (np.array(ncc_Bouguer) + 1)/2
        NCC = np.vstack([ncc_U_arr, ncc_FreeAir_arr, ncc_Bouguer_arr])


    # if "spectrum" in metrics_list:
    #     spectrum_ratio       = 1-NormalizeData_MinMax(spectrum_ratio)



    # ------------------------------------------------------------------------------------------------------

    metrics = np.empty((0, np.shape(rho_rng_arr)[0]),float)
    for metric in metrics_list:
        if metric=="Delta_mean": metrics=np.vstack([metrics, delta_mean])
        if metric=="Delta_std": metrics=np.vstack([metrics, delta_std])
        if metric=="RMSE": metrics=np.vstack([metrics, RMSE])
        if metric=="MAE": metrics=np.vstack([metrics, MAE])
        if metric=="R^2": metrics=np.vstack([metrics, R2])
        if metric=="SSIM": metrics=np.vstack([metrics, SSIM])
        if metric=="PSNR": metrics=np.vstack([metrics, PSNR])
        if metric=="NCC": metrics=np.vstack([metrics, NCC])
        # if metric=="spectrum": metrics=np.vstack([metrics, spectrum_ratio])
        

    interiors_parameters = [rho_rng_arr,radius_rng_arr,nhalf_rng_arr]


    # ------------------------------------------------------------------------------------------------------


    if plot_opt:
        fig, axs = plt.subplots(np.shape(metrics_list)[0], 3, figsize=(10,9))

        if np.shape(metrics_list)[0] == 1:
            axs[0].set_title(r'U')
            axs[1].set_title(r'Free-Air')
            axs[2].set_title(r'Bouguer')
            for i in range(np.shape(metrics_list)[0]):
                n, bins,_ = axs[0].hist(metrics[0,:],bins = 100)
                axs[0].grid()
                n, bins,_ = axs[1].hist(metrics[1,:],bins = 100)
                axs[1].grid()   
                n, bins,_ = axs[2].hist(metrics[2,:],bins = 100)
                axs[2].grid()
                axs[0].set_ylabel(r'$'+metrics_list[0]+'$')

        else:
            axs[0,0].set_title(r'U')
            axs[0,1].set_title(r'Free-Air')
            axs[0,2].set_title(r'Bouguer')
            j=0
            for i in range(np.shape(metrics_list)[0]):
                ax=axs[j, 0]
                n, bins,_ = ax.hist(metrics[3*j,:],bins = 100)
                ax.grid()
                ax=axs[j, 1]
                n, bins,_ = ax.hist(metrics[3*j+1,:],bins = 100)
                ax.grid()   
                ax=axs[j, 2]
                n, bins,_ = ax.hist(metrics[3*j+2,:],bins = 100)
                ax.grid()

                axs[j,0].set_ylabel(r'$'+metrics_list[j]+'$')

                j+=1
        plt.show()






    return metrics, interiors_parameters








##########################################################################################################################
##########################################################################################################################



def NormalizeData_MinMax(data):

    """
    Usage
    ----------
    Normalize data between 0 and 1


    Parameters
    ----------
    data          : array,
                    Input data to be normalized

    Output
    ----------
    normalizedData       : array,
                           Normalized data between 0 and 1
    """


    normalizedData = (data-np.min(data))/(np.max(data)-np.min(data)) 



    return normalizedData


##########################################################################################################################
##########################################################################################################################




def Corr2_Edo(A,B):

    r = 1 - (np.sum((B-A)**2) / np.sum((B - np.mean(B))**2))

    return r


##########################################################################################################################
##########################################################################################################################
