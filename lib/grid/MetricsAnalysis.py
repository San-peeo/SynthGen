from lib.lib_dep import *
from lib.utils.FreeMemory import *
from lib.grid.MetricsEvaluation import *



def MetricsAnalysis(metrics_list, models_dir, real_dir, eval_param, plot_opt=False):

    """
    Usage
    ----------
    Evaluate and aggregate metrics for a set of synthetic gravity models compared to real data.
    For each model in the provided directory, the function loads or computes the requested metrics
    (e.g., RMSE, MAE, R^2, SSIM, PSNR, etc.) for gravitational potential, Free-Air, and Bouguer anomalies.
    It returns arrays of metrics and the corresponding interior parameters. Optionally, it can plot 
    histograms of the metrics for all models.

    Parameters
    ----------
    metrics_list : list
                   List of metrics to evaluate. Options include:
                   - "Delta_mean": Mean difference
                   - "Delta_std": Standard deviation of the difference
                   - "MAE": Mean Absolute Error
                   - "RMSE": Root Mean Squared Error
                   - "R^2": Coefficient of determination
                   - "PSNR": Peak Signal-to-Noise Ratio
                   - "SSIM": Structural Similarity Index
                   - "NCC": Normalized Cross-Correlation
    models_dir   : str
                   Directory containing subdirectories for each synthetic model.
    real_dir     : str
                   Directory containing real data matrices.
    eval_param   : list
                   [coeffs_topo, n_min, n_max, i_max, r]; parameters for evaluation.
    plot_opt     : bool, optional
                   If True, plot histograms of the computed metrics (default: False).

    Output
    ----------
    metrics               : numpy.ndarray
                            Array containing the computed metrics for all models and requested metrics.
    interiors_parameters  : list
                            [rho_rng_arr, radius_rng_arr, nhalf_rng_arr]; lists of interior parameters for each model.
    """

    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    # Extract evaluation parameters:
    # eval_param = [coeffs_topo, n_min, n_max, i_max, r]
    coeffs_topo = eval_param[0]
    n_min       = eval_param[1]
    n_max       = eval_param[2]
    i_max       = eval_param[3]
    r           = eval_param[4]
    


    # Loading/Evaluating the metrics

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


    # Grid parameters
    rho_rng_arr     = []
    radius_rng_arr  = []
    nhalf_rng_arr   = []




    # Loop over the sub-directories:
    for counter, subdir in tqdm(enumerate(os.listdir(models_dir)), total=len(os.listdir(models_dir))):

        # memory leak issues:
        if counter % 100 == 0:
            FreeMemory()

        # Check if the subdir is a directory or empty
        if not os.path.isdir(models_dir+subdir):
            continue
        if len(os.listdir(models_dir+subdir)) == 0:
            print("Empty directory: ", models_dir+subdir)
            continue


        # Store interiors parameters:
        rho_rng_arr.append(np.loadtxt(models_dir+subdir+'/rho_layers.dat'))
        radius_rng_arr.append(np.loadtxt(models_dir+subdir+'/radius_layers.dat')) 
        nhalf_rng_arr.append(np.loadtxt(models_dir+subdir+'/n_half.dat'))



        # Load the metrics:
        if "Delta_mean" in metrics_list:
            try:
                delta_U_mean = np.loadtxt(models_dir+subdir+'/delta_U_mean.dat')
                delta_FreeAir_mean = np.loadtxt(models_dir+subdir+'/delta_FreeAir_mean.dat')
                delta_Bouguer_mean = np.loadtxt(models_dir+subdir+'/delta_Bouguer_mean.dat')

            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["Delta_mean"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                delta_U_mean = np.loadtxt(models_dir+subdir+'/delta_U_mean.dat')
                delta_FreeAir_mean = np.loadtxt(models_dir+subdir+'/delta_FreeAir_mean.dat')
                delta_Bouguer_mean = np.loadtxt(models_dir+subdir+'/delta_Bouguer_mean.dat')
            delta_U_mean_arr.append(delta_U_mean)
            delta_FreeAir_mean_arr.append(delta_FreeAir_mean)
            delta_Bouguer_mean_arr.append(delta_Bouguer_mean)

        if "Delta_std" in metrics_list:
            try:
                delta_U_std = np.loadtxt(models_dir+subdir+'/delta_U_std.dat')
                delta_FreeAir_std = np.loadtxt(models_dir+subdir+'/delta_FreeAir_std.dat')
                delta_Bouguer_std = np.loadtxt(models_dir+subdir+'/delta_Bouguer_std.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["Delta_std"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                delta_U_std = np.loadtxt(models_dir+subdir+'/delta_U_std.dat')
                delta_FreeAir_std = np.loadtxt(models_dir+subdir+'/delta_FreeAir_std.dat')
                delta_Bouguer_std = np.loadtxt(models_dir+subdir+'/delta_Bouguer_std.dat')
            delta_U_std_arr.append(delta_U_std)
            delta_FreeAir_std_arr.append(delta_FreeAir_std)
            delta_Bouguer_std_arr.append(delta_Bouguer_std)

        if "RMSE" in metrics_list:
            try:
                RMSE_U = np.loadtxt(models_dir+subdir+'/RMSE_U.dat')
                RMSE_FreeAir = np.loadtxt(models_dir+subdir+'/RMSE_FreeAir.dat')
                RMSE_Bouguer = np.loadtxt(models_dir+subdir+'/RMSE_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["RMSE"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                RMSE_U = np.loadtxt(models_dir+subdir+'/RMSE_U.dat')
                RMSE_FreeAir = np.loadtxt(models_dir+subdir+'/RMSE_FreeAir.dat')
                RMSE_Bouguer = np.loadtxt(models_dir+subdir+'/RMSE_Bouguer.dat')
            RMSE_U_arr.append(RMSE_U)
            RMSE_FreeAir_arr.append(RMSE_FreeAir)
            RMSE_Bouguer_arr.append(RMSE_Bouguer)

        if "MAE" in metrics_list:
            try:
                MAE_U = np.loadtxt(models_dir+subdir+'/MAE_U.dat')
                MAE_FreeAir = np.loadtxt(models_dir+subdir+'/MAE_FreeAir.dat')
                MAE_Bouguer = np.loadtxt(models_dir+subdir+'/MAE_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["MAE"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                MAE_U = np.loadtxt(models_dir+subdir+'/MAE_U.dat')
                MAE_FreeAir = np.loadtxt(models_dir+subdir+'/MAE_FreeAir.dat')
                MAE_Bouguer = np.loadtxt(models_dir+subdir+'/MAE_Bouguer.dat')
            MAE_U_arr.append(MAE_U)
            MAE_FreeAir_arr.append(MAE_FreeAir)
            MAE_Bouguer_arr.append(MAE_Bouguer)

        if "R^2" in metrics_list:
            try:
                R2_U = np.loadtxt(models_dir+subdir+'/R2_U.dat')
                R2_FreeAir = np.loadtxt(models_dir+subdir+'/R2_FreeAir.dat')
                R2_Bouguer = np.loadtxt(models_dir+subdir+'/R2_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["R^2"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r,rho_rng_arr[-1][-1])
                R2_U = np.loadtxt(models_dir+subdir+'/R2_U.dat')
                R2_FreeAir = np.loadtxt(models_dir+subdir+'/R2_FreeAir.dat')
                R2_Bouguer = np.loadtxt(models_dir+subdir+'/R2_Bouguer.dat')
            R2_U_arr.append(R2_U)
            R2_FreeAir_arr.append(R2_FreeAir)
            R2_Bouguer_arr.append(R2_Bouguer)

        if "SSIM" in metrics_list:
            try:
                ssim_U = np.loadtxt(models_dir+subdir+'/ssim_U.dat')
                ssim_FreeAir = np.loadtxt(models_dir+subdir+'/ssim_FreeAir.dat')
                ssim_Bouguer = np.loadtxt(models_dir+subdir+'/ssim_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["SSIM"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                ssim_U = np.loadtxt(models_dir+subdir+'/ssim_U.dat')
                ssim_FreeAir = np.loadtxt(models_dir+subdir+'/ssim_FreeAir.dat')
                ssim_Bouguer = np.loadtxt(models_dir+subdir+'/ssim_Bouguer.dat')
            ssim_U_arr.append(ssim_U)
            ssim_FreeAir_arr.append(ssim_FreeAir)
            ssim_Bouguer_arr.append(ssim_Bouguer)

        if "PSNR" in metrics_list:
            try:
                psnr_U = np.loadtxt(models_dir+subdir+'/psnr_U.dat')
                psnr_FreeAir = np.loadtxt(models_dir+subdir+'/psnr_FreeAir.dat')
                psnr_Bouguer = np.loadtxt(models_dir+subdir+'/psnr_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["PSNR"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                psnr_U = np.loadtxt(models_dir+subdir+'/psnr_U.dat')
                psnr_FreeAir = np.loadtxt(models_dir+subdir+'/psnr_FreeAir.dat')
                psnr_Bouguer = np.loadtxt(models_dir+subdir+'/psnr_Bouguer.dat')
            psnr_U_arr.append(psnr_U)
            psnr_FreeAir_arr.append(psnr_FreeAir)
            psnr_Bouguer_arr.append(psnr_Bouguer)

        if "NCC" in metrics_list:
            try:
                ncc_U = np.loadtxt(models_dir+subdir+'/ncc_U.dat')
                ncc_FreeAir = np.loadtxt(models_dir+subdir+'/ncc_FreeAir.dat')
                ncc_Bouguer = np.loadtxt(models_dir+subdir+'/ncc_Bouguer.dat')
            except:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')
                MetricsEvaluation(["NCC"], [coeffs_tot,coeffs_topo], real_dir,models_dir+subdir,
                                    n_min, n_max, i_max, r, rho_rng_arr[-1][-1])
                ncc_U = np.loadtxt(models_dir+subdir+'/ncc_U.dat')
                ncc_FreeAir = np.loadtxt(models_dir+subdir+'/ncc_FreeAir.dat')
                ncc_Bouguer = np.loadtxt(models_dir+subdir+'/ncc_Bouguer.dat')
            ncc_U_arr.append(ncc_U)
            ncc_FreeAir_arr.append(ncc_FreeAir)
            ncc_Bouguer_arr.append(ncc_Bouguer)






    if "Delta_mean" in metrics_list:
        delta_mean = np.vstack([delta_U_mean_arr, delta_FreeAir_mean_arr, delta_Bouguer_mean_arr])

    if "Delta_std" in metrics_list:
        delta_std = np.vstack([delta_U_std_arr, delta_FreeAir_std_arr, delta_Bouguer_std_arr])

    if "RMSE" in metrics_list:
        RMSE = np.vstack([RMSE_U_arr, RMSE_FreeAir_arr, RMSE_Bouguer_arr])

    if "MAE" in metrics_list:
        MAE = np.vstack([MAE_U_arr, MAE_FreeAir_arr, MAE_Bouguer_arr])

    if "R^2" in metrics_list:
        R2 = np.vstack([R2_U_arr, R2_FreeAir_arr, R2_Bouguer_arr])

    if "SSIM" in metrics_list:
        SSIM = np.vstack([ssim_U_arr, ssim_FreeAir_arr, ssim_Bouguer_arr])

    if "PSNR" in metrics_list:
        PSNR = np.vstack([psnr_U_arr, psnr_FreeAir_arr, psnr_Bouguer_arr])

    if "NCC" in metrics_list:
        NCC = np.vstack([ncc_U_arr, ncc_FreeAir_arr, ncc_Bouguer_arr])




    # ------------------------------------------------------------------------------------------------------------------------------

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
        

    interiors_parameters = [rho_rng_arr,radius_rng_arr,nhalf_rng_arr]


    # ------------------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------------------

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



    return metrics, interiors_parameters
