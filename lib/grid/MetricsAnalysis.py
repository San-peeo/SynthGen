from lib.lib_dep import *
from lib.utils.FreeMemory import *
from lib.globe_analysis.Global_Analysis import *
from lib.globe_analysis.Spectrum import *


def MetricsAnalysis(metrics_list, load_opt, saving_dir, models_dir, real_matrix, input_param):

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
    saving_dir   : str
                   Directory where metrics and parameters are saved or loaded from.
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


    # Loading already calculated metrics and interiors parameters:
    if load_opt:

        rho_rng_arr     = np.loadtxt(saving_dir+'rho_layers.dat')
        radius_rng_arr  = np.loadtxt(saving_dir+'radius_layers.dat')
        nhalf_rng_arr   = (np.loadtxt(saving_dir+'n_half.dat')).astype(int)

        if "Delta_mean" in metrics_list and os.path.isfile(saving_dir+'delta_mean.dat'):
            delta_mean = np.loadtxt(saving_dir+'delta_mean.dat')

        if "Delta_std" in metrics_list and os.path.isfile(saving_dir+'delta_std.dat'):
            delta_std = np.loadtxt(saving_dir+'delta_std.dat')

        if "RMSE" in metrics_list and os.path.isfile(saving_dir+'RMSE.dat'):
            RMSE = np.loadtxt(saving_dir+'RMSE.dat')

        if "MAE" in metrics_list and os.path.isfile(saving_dir+'MAE.dat'):
            MAE = np.loadtxt(saving_dir+'MAE.dat')

        if "R^2" in metrics_list and os.path.isfile(saving_dir+'R2.dat'):
            R2 = np.loadtxt(saving_dir+'R2.dat')

        if "SSIM" in metrics_list and os.path.isfile(saving_dir+'SSIM.dat'):
            SSIM = np.loadtxt(saving_dir+'SSIM.dat')

        if "PSNR" in metrics_list and os.path.isfile(saving_dir+'PSNR.dat'):
            PSNR = np.loadtxt(saving_dir+'PSNR.dat')

        if "NCC" in metrics_list and os.path.isfile(saving_dir+'NCC.dat'):
            NCC = np.loadtxt(saving_dir+'NCC.dat')

        if "spectrum" in metrics_list and os.path.isfile(saving_dir+'spectrum_ratio.dat'):
            spectrum_ratio =np.loadtxt(saving_dir+'spectrum_ratio.dat')

    # ------------------------------------------------------------------------------------------------------

    # Evaluate the metrics from the GRID models:
    else:


        # Metrics
        if "Delta_mean" in metrics_list:
            delta_U_mean                = []
            delta_FreeAir_mean          = []
            delta_Bouguer_mean          = []
        if "Delta_std" in metrics_list:
            delta_U_std                 = []
            delta_FreeAir_std           = []
            delta_Bouguer_std           = []
        if "RMSE" in metrics_list:
            RMSE_U                      = []
            RMSE_FreeAir                = []
            RMSE_Bouguer                = []
        if "MAE" in metrics_list:
            MAE_U                      = []
            MAE_FreeAir                = []
            MAE_Bouguer                = []
        if "R^2" in metrics_list:
            R2_U                        = []
            R2_FreeAir                  = []
            R2_Bouguer                  = []
        if "SSIM" in metrics_list:
            ssim_U                      = []
            ssim_FreeAir                = []
            ssim_Bouguer                = []
        if "PSNR" in metrics_list:
            psnr_U                      = []
            psnr_FreeAir                = []
            psnr_Bouguer                = []
        if "NCC" in metrics_list:
            ncc_U                      = []
            ncc_FreeAir                = []
            ncc_Bouguer                = []
        if "spectrum" in metrics_list:
            spectrum_ratio             = []

        # Grid parameters
        rho_rng_arr     = []
        radius_rng_arr  = []
        nhalf_rng_arr   = []

     # ------------------------------------------------------------------------------------------------------
    # If models_dir is a list of directories:


        if type(models_dir) is list:
            for dir in models_dir:
                print("Analysing directory: ", dir)
                FreeMemory()
                # Loop over the sub-directories:
                for counter, subdir in tqdm(enumerate(os.listdir(dir)), total=len(os.listdir(dir))):

                    # memory leak issues:
                    if counter % 100 == 0:
                        FreeMemory()


                    # Loading SynthGen coefficients:
                    coeffs_tot=pysh.SHGravCoeffs.from_file(dir+subdir+'/coeffs_tot.dat')


                    # Global analysis (U, H, FreeAir, Bouguer):
                    U_matrix,_,deltag_freeair,deltag_boug=Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,
                                                                        r=r,rho_boug=rho_boug,
                                                                        i_max=i_max,plot_opt=None,load_opt=False,verbose_opt=False)
                    # Spectrum analysis:
                    spectrum_synth = Spectrum(coeffs=[coeffs_tot],n_min=n_min,n_max=n_max,
                                                save_opt=None,load_opt=load_opt,verbose_opt=False)


                    # Evaluate metrics:
                    if "Delta_mean" in metrics_list:
                        delta_U_mean.append(np.mean(U_matrix_real-U_matrix.data))
                        delta_FreeAir_mean.append(np.mean(deltag_freeair_real-deltag_freeair.data))
                        delta_Bouguer_mean.append(np.mean(deltag_boug_real-deltag_boug.data))
                    if "Delta_std" in metrics_list:
                        delta_U_std.append(np.std(U_matrix_real-U_matrix.data))
                        delta_FreeAir_std.append(np.std(deltag_freeair_real-deltag_freeair.data))
                        delta_Bouguer_std.append(np.std(deltag_boug_real-deltag_boug.data))
                    if "RMSE" in metrics_list:
                        RMSE_U.append(sklearn.metrics.root_mean_squared_error(U_matrix_real,U_matrix.data))
                        RMSE_FreeAir.append(sklearn.metrics.root_mean_squared_error(deltag_freeair_real,deltag_freeair.data))
                        RMSE_Bouguer.append(sklearn.metrics.root_mean_squared_error(deltag_boug_real,deltag_boug.data))
                    if "MAE" in metrics_list:
                        MAE_U.append(np.mean(np.abs(U_matrix_real-U_matrix.data)))
                        MAE_FreeAir.append(np.mean(np.abs(deltag_freeair_real-deltag_freeair.data)))
                        MAE_Bouguer.append(np.mean(np.abs(deltag_boug_real-deltag_boug.data)))
                    if "R^2" in metrics_list:
                        R2_U.append(Corr2_Edo(U_matrix_real.flatten(),U_matrix.data.flatten()))
                        R2_FreeAir.append(Corr2_Edo(deltag_freeair_real.flatten(),deltag_freeair.data.flatten()))
                        R2_Bouguer.append(Corr2_Edo(deltag_boug_real.flatten(),deltag_boug.data.flatten())) 
                    if "SSIM" in metrics_list:
                        ssim_U.append(skimage.metrics.structural_similarity(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                        ssim_FreeAir.append(skimage.metrics.structural_similarity(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                        ssim_Bouguer.append(skimage.metrics.structural_similarity(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
                    if "PSNR" in metrics_list:
                        psnr_U.append(skimage.metrics.peak_signal_noise_ratio(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                        psnr_FreeAir.append(skimage.metrics.peak_signal_noise_ratio(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                        psnr_Bouguer.append(skimage.metrics.peak_signal_noise_ratio(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
                    if "NCC" in metrics_list:
                        U_matrix_real_mean = U_matrix_real - np.mean(U_matrix_real)
                        U_matrix_mean = U_matrix.data - np.mean(U_matrix.data)
                        deltag_freeair_real_mean = deltag_freeair_real - np.mean(deltag_freeair_real)
                        deltag_freeair_mean = deltag_freeair.data - np.mean(deltag_freeair.data)
                        deltag_boug_real_mean = deltag_boug_real - np.mean(deltag_boug_real)
                        deltag_boug_mean = deltag_boug.data - np.mean(deltag_boug.data)
                        ncc_U.append(np.sum(U_matrix_real_mean * U_matrix_mean) / (np.sqrt(np.sum(U_matrix_real_mean*2)) * np.sqrt(np.sum(U_matrix_mean*2))))
                        ncc_FreeAir.append(np.sum(deltag_freeair_real_mean * deltag_freeair_mean) / (np.sqrt(np.sum(deltag_freeair_real_mean*2)) * np.sqrt(np.sum(deltag_freeair_mean*2))))
                        ncc_Bouguer.append(np.sum(deltag_boug_real_mean * deltag_boug_mean) / (np.sqrt(np.sum(deltag_boug_real_mean*2)) * np.sqrt(np.sum(deltag_boug_mean*2))))
                        
                    # if "spectrum" in metrics_list:
                    #     spectrum_ratio.append(np.mean(spectrum_real/spectrum_synth))



                    # Store interiors parameters:
                    rho_rng_arr.append(np.loadtxt(dir+subdir+'/rho_layers.dat'))
                    radius_rng_arr.append(np.loadtxt(dir+subdir+'/radius_layers.dat')) 
                    nhalf_rng_arr.append(np.loadtxt(dir+subdir+'/n_half.dat'))


    # ------------------------------------------------------------------------------------------------------
    # If models_dir is a single directory:


        else:
            FreeMemory()
            # Loop over the sub-directories:
            for counter, subdir in tqdm(enumerate(os.listdir(models_dir)), total=len(os.listdir(models_dir))):

                # memory leak issues:
                if counter % 100 == 0:
                    FreeMemory()


                # Loading SynthGen coefficients:
                coeffs_tot=pysh.SHGravCoeffs.from_file(models_dir+subdir+'/coeffs_tot.dat')


                # Global analysis (U, H, FreeAir, Bouguer):
                U_matrix,_,deltag_freeair,deltag_boug=Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,
                                                                    r=r,rho_boug=rho_boug,
                                                                    i_max=i_max,plot_opt=None,load_opt=False,verbose_opt=False)
                # Spectrum analysis:
                spectrum_synth = Spectrum(coeffs=[coeffs_tot],n_min=n_min,n_max=n_max,
                                            save_opt=None,load_opt=load_opt,verbose_opt=False)


                # Evaluate metrics:
                if "Delta_mean" in metrics_list:
                    delta_U_mean.append(np.mean(U_matrix_real-U_matrix.data))
                    delta_FreeAir_mean.append(np.mean(deltag_freeair_real-deltag_freeair.data))
                    delta_Bouguer_mean.append(np.mean(deltag_boug_real-deltag_boug.data))
                if "Delta_std" in metrics_list:
                    delta_U_std.append(np.std(U_matrix_real-U_matrix.data))
                    delta_FreeAir_std.append(np.std(deltag_freeair_real-deltag_freeair.data))
                    delta_Bouguer_std.append(np.std(deltag_boug_real-deltag_boug.data))
                if "RMSE" in metrics_list:
                    RMSE_U.append(sklearn.metrics.root_mean_squared_error(U_matrix_real,U_matrix.data))
                    RMSE_FreeAir.append(sklearn.metrics.root_mean_squared_error(deltag_freeair_real,deltag_freeair.data))
                    RMSE_Bouguer.append(sklearn.metrics.root_mean_squared_error(deltag_boug_real,deltag_boug.data))
                if "MAE" in metrics_list:
                    MAE_U.append(np.mean(np.abs(U_matrix_real-U_matrix.data)))
                    MAE_FreeAir.append(np.mean(np.abs(deltag_freeair_real-deltag_freeair.data)))
                    MAE_Bouguer.append(np.mean(np.abs(deltag_boug_real-deltag_boug.data)))
                if "R^2" in metrics_list:
                    R2_U.append(Corr2_Edo(U_matrix_real.flatten(),U_matrix.data.flatten()))
                    R2_FreeAir.append(Corr2_Edo(deltag_freeair_real.flatten(),deltag_freeair.data.flatten()))
                    R2_Bouguer.append(Corr2_Edo(deltag_boug_real.flatten(),deltag_boug.data.flatten())) 
                if "SSIM" in metrics_list:
                    ssim_U.append(skimage.metrics.structural_similarity(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                    ssim_FreeAir.append(skimage.metrics.structural_similarity(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                    ssim_Bouguer.append(skimage.metrics.structural_similarity(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
                if "PSNR" in metrics_list:
                    psnr_U.append(skimage.metrics.peak_signal_noise_ratio(U_matrix_real,U_matrix.data, data_range=U_matrix_real.max() - U_matrix_real.min()))
                    psnr_FreeAir.append(skimage.metrics.peak_signal_noise_ratio(deltag_freeair_real,deltag_freeair.data, data_range=deltag_freeair_real.max() - deltag_freeair_real.min()))
                    psnr_Bouguer.append(skimage.metrics.peak_signal_noise_ratio(deltag_boug_real,deltag_boug.data, data_range=deltag_boug_real.max() - deltag_boug_real.min()))
                if "NCC" in metrics_list:
                    U_matrix_real_mean = U_matrix_real - np.mean(U_matrix_real)
                    U_matrix_mean = U_matrix.data - np.mean(U_matrix.data)
                    deltag_freeair_real_mean = deltag_freeair_real - np.mean(deltag_freeair_real)
                    deltag_freeair_mean = deltag_freeair.data - np.mean(deltag_freeair.data)
                    deltag_boug_real_mean = deltag_boug_real - np.mean(deltag_boug_real)
                    deltag_boug_mean = deltag_boug.data - np.mean(deltag_boug.data)
                    ncc_U.append(np.sum(U_matrix_real_mean * U_matrix_mean) / (np.sqrt(np.sum(U_matrix_real_mean*2)) * np.sqrt(np.sum(U_matrix_mean*2))))
                    ncc_FreeAir.append(np.sum(deltag_freeair_real_mean * deltag_freeair_mean) / (np.sqrt(np.sum(deltag_freeair_real_mean*2)) * np.sqrt(np.sum(deltag_freeair_mean*2))))
                    ncc_Bouguer.append(np.sum(deltag_boug_real_mean * deltag_boug_mean) / (np.sqrt(np.sum(deltag_boug_real_mean*2)) * np.sqrt(np.sum(deltag_boug_mean*2))))
                    
                # if "spectrum" in metrics_list:
                #     spectrum_ratio.append(np.mean(spectrum_real/spectrum_synth))



                # Store interiors parameters:
                rho_rng_arr.append(np.loadtxt(models_dir+subdir+'/rho_layers.dat'))
                radius_rng_arr.append(np.loadtxt(models_dir+subdir+'/radius_layers.dat')) 
                nhalf_rng_arr.append(np.loadtxt(models_dir+subdir+'/n_half.dat'))



    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------




        # Normalize the metrics:
        if "Delta_mean" in metrics_list:
            delta_U_mean         = NormalizeData_MinMax(np.abs(delta_U_mean))
            delta_FreeAir_mean   = NormalizeData_MinMax(np.abs(delta_FreeAir_mean))
            delta_Bouguer_mean   = NormalizeData_MinMax(np.abs(delta_Bouguer_mean))
            delta_mean = np.vstack([delta_U_mean, delta_FreeAir_mean, delta_Bouguer_mean])
            np.savetxt(saving_dir+'delta_mean.dat',delta_mean)

        if "Delta_std" in metrics_list:
            delta_U_std          = NormalizeData_MinMax(np.abs(delta_U_std))
            delta_FreeAir_std    = NormalizeData_MinMax(np.abs(delta_FreeAir_std))
            delta_Bouguer_std    = NormalizeData_MinMax(np.abs(delta_Bouguer_std))
            delta_std = np.vstack([delta_U_std, delta_FreeAir_std, delta_Bouguer_std])
            np.savetxt(saving_dir+'delta_std.dat',delta_std)

        if "RMSE" in metrics_list:
            RMSE_U               = NormalizeData_MinMax(np.abs(RMSE_U))
            RMSE_FreeAir         = NormalizeData_MinMax(np.abs(RMSE_FreeAir))
            RMSE_Bouguer         = NormalizeData_MinMax(np.abs(RMSE_Bouguer))
            # RMSE_U                  = 1/(1+np.array(RMSE_U))
            # RMSE_FreeAir            = 1/(1+np.array(RMSE_FreeAir))
            # RMSE_Bouguer            = 1/(1+np.array(RMSE_Bouguer))
            RMSE = np.vstack([RMSE_U, RMSE_FreeAir, RMSE_Bouguer])
            np.savetxt(saving_dir+'RMSE.dat',RMSE)

            
        if "MAE" in metrics_list:
            MAE_U               = NormalizeData_MinMax(np.abs(MAE_U))
            MAE_FreeAir         = NormalizeData_MinMax(np.abs(MAE_FreeAir))
            MAE_Bouguer         = NormalizeData_MinMax(np.abs(MAE_Bouguer))
            # MAE_U               = 1/(1+np.array(MAE_U))
            # MAE_FreeAir         = 1/(1+np.array(MAE_FreeAir))
            # MAE_Bouguer         = 1/(1+np.array(MAE_Bouguer))
            MAE = np.vstack([MAE_U, MAE_FreeAir, MAE_Bouguer])
            np.savetxt(saving_dir+'MAE.dat',MAE)

        if "R^2" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            R2_U                 = (np.array(R2_U) + 1)/2
            R2_FreeAir           = (np.array(R2_FreeAir) + 1)/2
            R2_Bouguer           = (np.array(R2_Bouguer) + 1)/2
            R2 = np.vstack([R2_U, R2_FreeAir, R2_Bouguer])
            np.savetxt(saving_dir+'R2.dat',R2)

        if "SSIM" in metrics_list:
            # ssim_U               = NormalizeData_MinMax(np.abs(ssim_U))
            # R2_FreeAir           = NormalizeData_MinMax(np.abs(ssim_FreeAir))
            # R2_Bouguer           = NormalizeData_MinMax(np.abs(ssim_Bouguer))
            SSIM = np.vstack([ssim_U, ssim_FreeAir, ssim_Bouguer])
            np.savetxt(saving_dir+'SSIM.dat',SSIM)

        if "PSNR" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            
            psnr_U                 = np.array(psnr_U)/np.max(psnr_U)
            psnr_FreeAir           = np.array(psnr_FreeAir)/np.max(psnr_FreeAir)
            psnr_Bouguer           = np.array(psnr_Bouguer)/np.max(psnr_Bouguer)
            PSNR = np.vstack([psnr_U, psnr_FreeAir, psnr_Bouguer])
            np.savetxt(saving_dir+'PSNR.dat',PSNR)

        if "NCC" in metrics_list:
            # R2_U                 = NormalizeData_MinMax(1-(np.abs(R2_U)))
            # R2_FreeAir           = NormalizeData_MinMax(1-(np.abs(R2_FreeAir)))
            # R2_Bouguer           = NormalizeData_MinMax(1-(np.abs(R2_Bouguer)))
            ncc_U                 = (np.array(ncc_U) + 1)/2
            ncc_FreeAir           = (np.array(ncc_FreeAir) + 1)/2
            ncc_Bouguer           = (np.array(ncc_Bouguer) + 1)/2
            NCC = np.vstack([ncc_U, ncc_FreeAir, ncc_Bouguer])
            np.savetxt(saving_dir+'NCC.dat',NCC)


        # if "spectrum" in metrics_list:
        #     spectrum_ratio       = 1-NormalizeData_MinMax(spectrum_ratio)

        np.savetxt(saving_dir+'/rho_layers.dat',rho_rng_arr)
        np.savetxt(saving_dir+'/radius_layers.dat',radius_rng_arr)
        np.savetxt(saving_dir+'/n_half.dat',np.array(nhalf_rng_arr, dtype=np.int64)) 



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
        if metric=="spectrum": metrics=np.vstack([metrics, spectrum_ratio])
        

    interiors_parameters = [rho_rng_arr,radius_rng_arr,nhalf_rng_arr]


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

    # Manually calculate the correlation coefficient between two matrices A and B.


    # Detrending:
    A -= np.mean(A)
    B -= np.mean(B)

    r = np.sum(np.sum(A*B))/np.sqrt(np.sum(np.sum(A*A))*np.sum(np.sum(B*B)))

    return r


##########################################################################################################################
##########################################################################################################################
