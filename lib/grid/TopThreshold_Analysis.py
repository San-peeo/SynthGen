from lib.lib_dep import *

def TopThreshold_Analysis(rho_arr,radius_arr,nhalf_arr,rigid_arr,visc_arr, final_metric, threshold_arr,truth_arr=[None]*5, layer_name=[], saving_dir=None):

    """
    Usage
    ----------
    Analyze and visualize the top% models based on a threshold percentage of the final metric.
    The function generates histograms for density, radius, and degree n_half for each layer, 
    fits Gaussian distributions, and calculates their mean and standard deviation.

    
    Parameters
    ----------
    rho_rng_sort     : numpy.ndarray
                       Sorted array of density values for each layer.
    radius_rng_sort  : numpy.ndarray
                       Sorted array of radius values for each layer.
    nhalf_rng_sort   : numpy.ndarray
                       Sorted array of degree n_half values for each layer.
    final_metric     : numpy.ndarray
                       Array of final metric values for all models.
    threshold_arr    : list
                       List of threshold percentages (e.g., [0.05, 0.1]) to select top models.
    layer_name       : list, default = []
                       List of layers name for plotting title.
    saving_dir       : str, default = None
                       Directory to save the generated histogram plots.
                       If None, plots are not saved.

                       
    Output
    ----------
    rho     : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of density for each layer.
    radius  : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of radius for each layer.
    n_half  : numpy.ndarray
              Array containing the mean (mu) and standard deviation (sigma) of degree n_half for each layer.
    fig     : matplotlib.figure.Figure
              Figure handle containing the generated histograms.
    """


    # ------------------------------------------------------------------------------------------------------

    n_layers = np.shape(rho_arr)[1]

    hist_color=matlab_colors

    rho     = np.zeros((len(threshold_arr),n_layers,2))
    radius  = np.zeros((len(threshold_arr),n_layers,2))
    n_half  = np.zeros((len(threshold_arr),n_layers,2))
    rigid   = np.zeros((len(threshold_arr),n_layers,2))
    visc    = np.zeros((len(threshold_arr),n_layers,2))
    result  = [rho,radius,n_half,rigid,visc]

    rho_true    = truth_arr[0]
    radius_true = truth_arr[1]
    n_half_true = truth_arr[2]
    rigid_true  = truth_arr[3]
    visc_true   = truth_arr[4]


    rigid_norm = 1e+10


    thresh_name=''

    # ------------------------------------------------------------------------------------------------------

    # Selecting top [threshold] %
    fig, axs = plt.subplots(n_layers, 5, figsize=(18,9))
    labels=[]
    handles = []

    bins_all1 = []
    bins_all2 = []
    bins_all3 = []
    bins_all4 = []
    bins_all5 = []


    # All simulations
    for i in range(n_layers):

        ax=axs[i, 0]

        if np.std(rho_arr[0:50,i])!=0:
            # bins1 = np.arange(np.min(rho_arr[:,i]),np.max(rho_arr[:,i])+1,10,dtype=np.int64)
            # n, bins,_ = ax.hist(rho_arr[:,i],bins=bins1, alpha=0.5,color=hist_color[1])
            n, bins,_ = ax.hist(rho_arr[:,i],bins='auto', alpha=0.5,color=hist_color[1])
            ax.grid()
            if rho_true is not None: ax.axvline(x = rho_true[i], color = 'red', label=r'$\rho_{true}=%.1f$' %(rho_true[i]), linestyle='--',linewidth=1.5)
            ax.legend()
        else:
            ax.clear()
            ax.set_axis_off()
        axs[0,0].set_title(r'Density $[kg/m^3]$')
        if layer_name==[]: ax.set_ylabel(r'Layer '+str(i+1), rotation=90, size='large')
        else: ax.set_ylabel(layer_name[i],rotation=90, size='large')
        bins_all1.append(bins)


        ax=axs[i, 1]
        if np.std(radius_arr[0:50,i])!=0:
            # bins2 = np.arange(np.min(radius_arr[:,i]),np.max(radius_arr[:,i])+1,10,dtype=np.int64)
            # n, bins,_ = ax.hist(rho_arr[:,i],bins=bins2, alpha=0.5,color=hist_color[1])
            n, bins,_ = ax.hist(radius_arr[:,i],bins='auto', alpha=0.5,color=hist_color[1])
            ax.grid()
            if radius_true is not None: ax.axvline(x = radius_true[i], color = 'red',label=r'$R_{true}=%.1f$' %(radius_true[i]), linestyle='--',linewidth=1.5)
            ax.legend()
        else:
            ax.clear()
            ax.set_axis_off()
        axs[0,1].set_title(r'Radius $[km]$')
        bins_all2.append(bins)
            
        ax=axs[i, 2]
        if np.std(nhalf_arr[0:50,i])!=0:
            bins = np.arange(np.min(nhalf_arr[:,i]),np.max(nhalf_arr[:,i])+1,dtype=np.int64)
            # n, bins,_ = ax.hist(nhalf_arr[:,i],bins=bins3, alpha=0.5,color=hist_color[1])
            n, bins,_ = ax.hist(nhalf_arr[:,i],bins=bins, alpha=0.5,color=hist_color[1])
            ax.grid()
            if n_half_true is not None: ax.axvline(x = n_half_true[i], color = 'red',label=r'$l_{half true}=%.1f$' %(n_half_true[i]), linestyle='--',linewidth=1.5)
            ax.legend()
        else:
            ax.clear()
            ax.set_axis_off()
        axs[0,2].set_title(r'Degree $l_{half}$')
        bins_all3.append(bins)
        

        ax=axs[i, 3]
        if np.std(rigid_arr[0:50,i])!=0:
            # bins4 = np.arange(np.min(rigid_arr[:,i]),np.max(rigid_arr[:,i])+1,10**11,dtype=np.int64)
            # n, bins,_ = ax.hist(rigid_arr[:,i],bins=bins4, alpha=0.5,color=hist_color[1])
            n, bins,_ = ax.hist(rigid_arr[:,i]/rigid_norm,bins='auto', alpha=0.5,color=hist_color[1])
            ax.grid()
            if rigid_true is not None :ax.axvline(x = rigid_true[i]/rigid_norm, color = 'red', label=r'$\mu_{truth}=%.1f$' %(rigid_true[i]/rigid_norm), linestyle='--',linewidth=1.5)
            ax.legend()
        else:
            ax.clear()
            ax.set_axis_off()
        axs[0,3].set_title(r' Rigidity $[10^{10}Pa]$')
        bins_all4.append(bins)


        ax=axs[i, 4]
        if np.std(visc_arr[0:50,i])!=0:
            # bins5 = 10 ** np.linspace(np.log10(np.min(visc_arr[:,i])), np.log10(np.max(visc_arr[:,i])), 20)
            # n, bins,_ = ax.hist(np.log10(visc_arr[:,i]),bins=bins5, alpha=0.5,color=hist_color[1])
            n, bins,_ = ax.hist(np.log10(visc_arr[:,i]),bins='auto', alpha=0.5,color=hist_color[1])
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25*np.log10(np.min(visc_arr[:,i]))))
            ax.set_xticks(np.arange(np.log10(np.min(visc_arr[:,i])), np.log10(np.max(visc_arr[:,i]))+1,0.5))
            ax.grid()
            if visc_true is not None: ax.axvline(x = np.log10(visc_true[i]), color = 'red',  label=r'$\eta_{truth}=%.2f$' %(np.log10(visc_true[i])), linestyle='--',linewidth=1.5)
            ax.legend()
        else:
            ax.clear()
            ax.set_axis_off()
        axs[0,4].set_title(r'$log_{10}$(Viscosity) $[Pa \cdot s]$')
        bins_all5.append(bins)


    labels.append('All ('+str(len(final_metric))+' models)')
    handles.append(Patch(edgecolor=hist_color[1], facecolor=hist_color[1], fill=True, alpha=0.5))
        




    for j,thresh in enumerate(threshold_arr):

        best_idx_arr = np.where(final_metric >= 1-thresh)
        if np.shape(best_idx_arr)[1]!=0:
            best_idx = best_idx_arr[0][0]
        else:
            print("Top "+ str(thresh*100) + "% models: ", 0, "/", len(final_metric))
            continue


        rho_rng_valid_sort_best      = rho_arr[best_idx:]
        radius_rng_valid_sort_best   = radius_arr[best_idx:]
        nhalf_rng_valid_sort_best    = nhalf_arr[best_idx:]
        rigid_rng_valid_sort_best    = rigid_arr[best_idx:]/(1e+10)
        visc_rng_valid_sort_best     = np.log10(visc_arr[best_idx:])


        print("Top "+ str(thresh*100) + "% models: ", np.shape(best_idx_arr)[1], "/", len(final_metric))


        thresh_name+= str(np.round(thresh*100))
        if j!=np.shape(threshold_arr)[0] or j!=len(threshold_arr)-1: thresh_name +='_'



        # ------------------------------------------------------------------------------------------------------

        # Histogram and Analysis
        array       = [rho_rng_valid_sort_best,radius_rng_valid_sort_best,nhalf_rng_valid_sort_best,rigid_rng_valid_sort_best,visc_rng_valid_sort_best]        
        bins_all    = [bins_all1,bins_all2,bins_all3,bins_all4,bins_all5]

        for k,arr in enumerate(array):

            for i in range(n_layers):
                ax=axs[i, k]

                # Fitting Distribution:
                if np.std(arr[0:50,i]) != 0:
                    n, bins,_ = ax.hist(arr[:,i],bins = bins_all[k][i], alpha=1,color=hist_color[j])

                    try:
                        func=Gaussian_func
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        popt_G, _ = curve_fit(func, bin_centers, n, p0=[np.max(n), np.mean(arr[:,i]), np.std(arr[:,i])],bounds=((-np.inf, np.min(arr[:,i]), -np.inf),(np.inf, np.max(arr[:,i]), np.inf)))
                        r2_G = R2_gof(n,func(bin_centers,*popt_G))
                        mu,sigma = popt_G[1],popt_G[2]
                        result[k][j,i,0] = mu
                        result[k][j,i,1] = sigma
                        popt=popt_G

                        if r2_G<0.9:
                            func=Skew_func
                            bin_centers = (bins[:-1] + bins[1:]) / 2
                            popt, _ = curve_fit(func, bin_centers, n, p0=[np.max(n), np.mean(arr[:,i]), np.std(arr[:,i]),+10],bounds=((-np.inf, np.min(arr[:,i]), -np.inf,-np.inf),(np.inf, np.max(arr[:,i]), np.inf,np.inf)))
                            r2 = R2_gof(n,func(bin_centers,*popt))
                            mu,sigma,gamma = popt[1],popt[2],popt[3]
                            delta = gamma/(np.sqrt(1+gamma**2))
                            result[k][j,i,0] = np.round(mu + sigma*(np.sqrt(2/np.pi)*delta - (1-np.pi/4)*(np.sqrt(2/np.pi)*delta)**3/(1-2/np.pi*delta**2) - np.sign(gamma)/2*np.exp(-2*np.pi/np.abs(gamma))),1)
                            result[k][j,i,1] = np.round(sigma*np.sqrt(1-2*delta**2/np.pi),1)


                            if r2<r2_G:
                                func=Gaussian_func
                                popt=popt_G
                                r2=r2_G
                                mu,sigma = popt[1],popt[2]
                                result[k][j,i,0] = mu
                                result[k][j,i,1] = sigma

                        if result[k][j,i,0]<np.max(arr[:,i]) and result[k][j,i,0]>np.min(arr[:,i]):
                            if k==0: 
                                ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\rho=%.1f \pm %.1f$' %(result[k][j,i,0], result[k][j,i,1]), color=hist_color[j])
                                ax.axvline(x = result[k][j,i,0], color = hist_color[j], linestyle='--',linewidth=1)
                            if k==1: 
                                ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $R=%.1f \pm %.1f$' %(result[k][j,i,0], result[k][j,i,1]), color=hist_color[j])
                                ax.axvline(x = result[k][j,i,0], color = hist_color[j], linestyle='--',linewidth=1)
                            if k==2:
                                ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $l_{half}=%.1f \pm %.1f$' %(result[k][j,i,0], result[k][j,i,1]), color=hist_color[j])
                                ax.axvline(x = result[k][j,i,0], color = hist_color[j], linestyle='--',linewidth=1)
                            if k==3: 
                                ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\epsilon=%.1f \pm %.1f$' %(result[k][j,i,0], result[k][j,i,1]), color=hist_color[j])
                                ax.axvline(x = result[k][j,i,0], color = hist_color[j], linestyle='--',linewidth=1)
                            if k==4: 
                                ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.2f \pm %.2f$' %(result[k][j,i,0], result[k][j,i,1]), color=hist_color[j])
                                ax.axvline(x = result[k][j,i,0], color = hist_color[j], linestyle='--',linewidth=1)
                        
                        ax.legend(fontsize=10)

                    except:
                        print('No Fit')

                    ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
                        




        labels.append(str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)')
        handles.append(Patch(edgecolor=hist_color[j], facecolor=hist_color[j], fill=True, alpha=0.5))


    fig.legend(handles=handles,labels=labels, loc = 'upper center', ncol=len(threshold_arr)+1,fontsize=13)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)    
    plt.show()


    if saving_dir is not None: plt.savefig(saving_dir+'histograms_'+thresh_name+'.png', dpi=600)


    result[3][:,:,0] *= rigid_norm
    result[4][:,:,0] = 10**result[4][:,:,0]
    

    return result, fig








##########################################################################################################################
##########################################################################################################################



def Gaussian_func(x, a, x0, sigma):


    return a * np.exp(-(x-x0)**2/(2*sigma**2))


##########################################################################################################################
##########################################################################################################################


def Skew_func(x, a, x0, sigma,gamma):

    return a * np.exp(-(x-x0)**2/(2*sigma**2)) * (1+erf(gamma*(x-x0)/(sigma*np.sqrt(2))))


##########################################################################################################################
##########################################################################################################################


def R2_gof(y, yfit):

    # residual sum of squares
    ss_res = np.sum((y - yfit) ** 2)

    # total sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)


    return r2



