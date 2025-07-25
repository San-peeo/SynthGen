from lib.lib_dep import *
from lib.misc.FreeMemory import *
from lib.globe_analysis.Global_Analysis import *



def MetricsEvaluation(metrics_list,coeffs,real_dir,saving_dir, n_min,n_max,i_max,r,rho_boug):


    """
    Usage
    ----------
    Compute and save various metrics comparing synthetic and real gravity data (potential, Free-Air, Bouguer) for a planetary body.
    Metrics include mean and standard deviation of differences, RMSE, MAE, R², SSIM, PSNR, and NCC.
    Results are saved as .dat files in the specified directory.

    Parameters
    ----------
    metrics_list : list
                   List of metrics to compute. Options include:
                   - "Delta_mean": Mean difference
                   - "Delta_std": Standard deviation of the difference
                   - "MAE": Mean Absolute Error
                   - "RMSE": Root Mean Squared Error
                   - "R^2": Coefficient of determination
                   - "PSNR": Peak Signal-to-Noise Ratio
                   - "SSIM": Structural Similarity Index
                   - "NCC": Normalized Cross-Correlation
    coeffs       : list
                   [coeffs_grav, coeffs_topo]; spherical harmonic coefficients for gravity and topography.
    real_dir     : str
                   Directory containing real data matrices.
    saving_dir   : str
                   Directory where output metric files will be saved.
    n_min        : int
                   Minimum spherical harmonic degree.
    n_max        : int
                   Maximum spherical harmonic degree.
    i_max        : int
                   Maximum order for Taylor expansion.
    r            : float
                   Reference radius for evaluation.
    rho_boug     : float
                   Crust density for Bouguer correction.

    Output
    ----------
    The function saves each computed metric as a separate .dat file in the specified saving_dir.
    """




    coeffs_tot  = coeffs[0]
    coeffs_topo = coeffs[1]




    # Reading "Real" data:
    U_matrix_real = np.loadtxt(real_dir+'U_matrix_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
    deltag_freeair_real = np.loadtxt(real_dir+'deltag_freeair_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
    deltag_boug_real = np.loadtxt(real_dir+'deltag_boug_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')






 # ------------------------------------------------------------------------------------------------------

    # Global analysis (U, H, FreeAir, Bouguer):
    U_matrix,_,deltag_freeair,deltag_boug=Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,
                                                            r=r,rho_boug=rho_boug,
                                                            i_max=i_max,plot_opt=None,load_opt=False,verbose_opt=False)
    # Evaluate metrics:
    if "Delta_mean" in metrics_list:
        delta_U_mean = np.mean(U_matrix_real-U_matrix.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_U_mean.dat',[delta_U_mean])
        delta_FreeAir_mean = np.mean(deltag_freeair_real-deltag_freeair.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_FreeAir_mean.dat',[delta_FreeAir_mean])
        delta_Bouguer_mean = np.mean(deltag_boug_real-deltag_boug.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_Bouguer_mean.dat',[delta_Bouguer_mean])

    if "Delta_std" in metrics_list:
        delta_U_std = np.std(U_matrix_real-U_matrix.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_U_std.dat',[delta_U_std])
        delta_FreeAir_std = np.std(deltag_freeair_real-deltag_freeair.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_FreeAir_std.dat',[delta_FreeAir_std])
        delta_Bouguer_std = np.std(deltag_boug_real-deltag_boug.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/delta_Bouguer_std.dat',[delta_Bouguer_std])

    if "RMSE" in metrics_list:
        RMSE_U = sklearn.metrics.root_mean_squared_error(U_matrix_real,U_matrix.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/RMSE_U.dat',[RMSE_U])
        RMSE_FreeAir = sklearn.metrics.root_mean_squared_error(deltag_freeair_real,deltag_freeair.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/RMSE_FreeAir.dat',[RMSE_FreeAir])
        RMSE_Bouguer = sklearn.metrics.root_mean_squared_error(deltag_boug_real,deltag_boug.data)
        if saving_dir is not None: np.savetxt(saving_dir+'/RMSE_Bouguer.dat',[RMSE_Bouguer])

    if "MAE" in metrics_list:
        MAE_U = np.mean(np.abs(U_matrix_real-U_matrix.data))
        if saving_dir is not None: np.savetxt(saving_dir+'/MAE_U.dat',[MAE_U])
        MAE_FreeAir = np.mean(np.abs(deltag_freeair_real-deltag_freeair.data))
        if saving_dir is not None: np.savetxt(saving_dir+'/MAE_FreeAir.dat',[MAE_FreeAir])
        MAE_Bouguer = np.mean(np.abs(deltag_boug_real-deltag_boug.data))
        if saving_dir is not None: np.savetxt(saving_dir+'/MAE_Bouguer.dat',[MAE_Bouguer])

    if "R^2" in metrics_list:
        R2_U = Corr2_Edo(U_matrix_real.flatten(),U_matrix.data.flatten())
        if saving_dir is not None: np.savetxt(saving_dir+'/R2_U.dat',[R2_U])
        R2_FreeAir = Corr2_Edo(deltag_freeair_real.flatten(),deltag_freeair.data.flatten())
        if saving_dir is not None: np.savetxt(saving_dir+'/R2_FreeAir.dat',[R2_FreeAir])
        R2_Bouguer = Corr2_Edo(deltag_boug_real.flatten(),deltag_boug.data.flatten())
        if saving_dir is not None: np.savetxt(saving_dir+'/R2_Bouguer.dat',[R2_Bouguer])

    if "SSIM" in metrics_list:
        ssim_U = skimage.metrics.structural_similarity(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/ssim_U.dat',[ssim_U])
        ssim_FreeAir = skimage.metrics.structural_similarity(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/ssim_FreeAir.dat',[ssim_FreeAir])
        ssim_Bouguer = skimage.metrics.structural_similarity(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/ssim_Bouguer.dat',[ssim_Bouguer])

    if "PSNR" in metrics_list:
        psnr_U = skimage.metrics.peak_signal_noise_ratio(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/psnr_U.dat',[psnr_U])
        psnr_FreeAir = skimage.metrics.peak_signal_noise_ratio(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/psnr_FreeAir.dat',[psnr_FreeAir])
        psnr_Bouguer = skimage.metrics.peak_signal_noise_ratio(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min())
        if saving_dir is not None: np.savetxt(saving_dir+'/psnr_Bouguer.dat',[psnr_Bouguer])
        
    if "NCC" in metrics_list:
        U_matrix_real_mean = U_matrix_real - np.mean(U_matrix_real)
        U_matrix_mean = U_matrix.data - np.mean(U_matrix.data)
        ncc_U = np.sum(U_matrix_real_mean * U_matrix_mean) / (np.sqrt(np.sum(U_matrix_real_mean**2)) * np.sqrt(np.sum(U_matrix_mean**2)))
        if saving_dir is not None: np.savetxt(saving_dir+'/ncc_U.dat',[ncc_U])

        deltag_freeair_real_mean = deltag_freeair_real - np.mean(deltag_freeair_real)
        deltag_freeair_mean = deltag_freeair.data - np.mean(deltag_freeair.data)
        ncc_FreeAir = np.sum(deltag_freeair_real_mean * deltag_freeair_mean) / (np.sqrt(np.sum(deltag_freeair_real_mean**2)) * np.sqrt(np.sum(deltag_freeair_mean**2)))
        if saving_dir is not None: np.savetxt(saving_dir+'/ncc_FreeAir.dat',[ncc_FreeAir])

        deltag_boug_real_mean = deltag_boug_real - np.mean(deltag_boug_real)
        deltag_boug_mean = deltag_boug.data - np.mean(deltag_boug.data)
        ncc_Bouguer = np.sum(deltag_boug_real_mean * deltag_boug_mean) / (np.sqrt(np.sum(deltag_boug_real_mean**2)) * np.sqrt(np.sum(deltag_boug_mean**2)))
        if saving_dir is not None: np.savetxt(saving_dir+'/ncc_Bouguer.dat',[ncc_Bouguer])



# ------------------------------------------------------------------------------------------------------
    delta_mean=[]
    delta_std=[]
    RMSE=[]
    MAE=[]
    R2=[]
    SSIM=[]
    PSNR=[]
    NCC=[]


    if "Delta_mean" in metrics_list:
        delta_mean = [delta_U_mean, delta_FreeAir_mean, delta_Bouguer_mean]

    if "Delta_std" in metrics_list:
        delta_std = [delta_U_std, delta_FreeAir_std, delta_Bouguer_std]

    if "RMSE" in metrics_list:
        RMSE = [RMSE_U, RMSE_FreeAir, RMSE_Bouguer]

    if "MAE" in metrics_list:
        MAE = [MAE_U, MAE_FreeAir, MAE_Bouguer]

    if "R^2" in metrics_list:
        R2 = [R2_U, R2_FreeAir, R2_Bouguer]

    if "SSIM" in metrics_list:
        SSIM =[ssim_U, ssim_FreeAir, ssim_Bouguer]

    if "PSNR" in metrics_list:
        PSNR = [psnr_U, psnr_FreeAir, psnr_Bouguer]

    if "NCC" in metrics_list:
        NCC = [ncc_U, ncc_FreeAir, ncc_Bouguer]




    return [delta_mean, delta_std, RMSE, MAE, R2, SSIM, PSNR, NCC]



##########################################################################################################################
##########################################################################################################################




def Corr2_Edo(A,B):

    r = 1 - (np.sum((B-A)**2) / np.sum((B - np.mean(B))**2))

    return r


##########################################################################################################################
##########################################################################################################################