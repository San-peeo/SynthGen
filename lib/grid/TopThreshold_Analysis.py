from lib.lib_dep import *

def TopThreshold_Analysis(rho_rng_sort,radius_rng_sort,nhalf_rng_sort, final_metric, threshold_arr, saving_dir=None):

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

    n_layers = np.shape(rho_rng_sort)[1]

    hist_color=matlab_colors

    rho     = np.zeros((n_layers,2))
    radius  = np.zeros((n_layers,2))
    n_half  = np.zeros((n_layers,2))
    thresh_name=''

    # ------------------------------------------------------------------------------------------------------

    # Selecting top [threshold] %
    fig, axs = plt.subplots(n_layers, 3, figsize=(12,10))
    labels=[]
    handles = []

    for j,thresh in enumerate(threshold_arr):

        best_idx_arr = np.where(final_metric >= 1-thresh)
        if np.shape(best_idx_arr)[1]!=0:
            best_idx = best_idx_arr[0][0]
        else:
            print("Top "+ str(thresh*100) + "% models: ", 0, "/", len(final_metric))
            continue
        
        rho_rng_valid_sort_best      = rho_rng_sort[best_idx:]
        radius_rng_valid_sort_best   = radius_rng_sort[best_idx:]
        nhalf_rng_valid_sort_best    = nhalf_rng_sort[best_idx:]

        print("Top "+ str(thresh*100) + "% models: ", np.shape(best_idx_arr)[1], "/", len(final_metric))


        thresh_name+= str(np.round(thresh*100))
        if j!=np.shape(threshold_arr)[0]: thresh_name +='_'

        # ------------------------------------------------------------------------------------------------------

        # Histogram and Analysis

        for i in range(n_layers):

            ax=axs[i, 0]
            n, bins,_ = ax.hist(rho_rng_valid_sort_best[:,i],bins = 100, alpha=1,color=hist_color[j])
            # Fitting normal distribution:
            if np.std(rho_rng_valid_sort_best[:,i]) != 0:
                try:
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(rho_rng_valid_sort_best[:,i]), np.std(rho_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(rho_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(rho_rng_valid_sort_best[:,i]), np.inf)))
                    mu,sigma = popt[1],popt[2]
                    ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                    ax.legend()
                    rho[i,0] = mu
                    rho[i,1] = sigma
                except:
                    print('No Gaussian fit (Density)')
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Density $[kg/m^3]$')
            ax.set_title(r'Layer '+str(i+1))




            ax=axs[i, 1]
            n, bins,_ = ax.hist(radius_rng_valid_sort_best[:,i],bins = 100, alpha=1,color=hist_color[j])
            # Fitting normal distribution:
            if np.std(radius_rng_valid_sort_best[:,i]) != 0:
                try:
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(radius_rng_valid_sort_best[:,i]), np.std(radius_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(radius_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(radius_rng_valid_sort_best[:,i]), np.inf)))
                    mu,sigma = popt[1],popt[2]
                    ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                    ax.legend()
                    radius[i,0] = mu
                    radius[i,1] = sigma
                except:
                    print('No Gaussian fit (Radius)')
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Radius $[km]$')
            ax.set_title(r'Layer '+str(i+1))


            ax=axs[i, 2]
            if nhalf_rng_valid_sort_best[:,i].all() != 0 and len(nhalf_rng_valid_sort_best[:,i]) !=0:
                n, bins,_ = ax.hist(nhalf_rng_valid_sort_best[:,i],bins = int(np.max(nhalf_rng_valid_sort_best[:,i])-np.min(nhalf_rng_valid_sort_best[:,i]))+1, alpha=1,color=hist_color[j])
                # Fitting normal distribution:
                if np.std(nhalf_rng_valid_sort_best[:,i]) != 0:
                    try:
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        popt, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(nhalf_rng_valid_sort_best[:,i]), np.std(nhalf_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(nhalf_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(nhalf_rng_valid_sort_best[:,i]), np.inf)))
                        mu,sigma = np.round(popt[1]),np.round(popt[2])
                        ax.plot(bin_centers, Gaussian_func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\mu=%.1f,\ \sigma=%.1f$' %(mu, sigma), color=hist_color[j])
                        ax.legend()
                        n_half[i,0] = mu
                        n_half[i,1] = sigma
                    except:
                        print('No Gaussian fit (n_half)')
            else:
                ax.set_xlim([-5,5])
                ax.set_xticks(np.arange(-5,6))
            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Degree n_{half}')
            ax.set_title(r'Layer '+str(i+1))



        labels.append(str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)')
        handles.append(Patch(edgecolor=hist_color[j], facecolor=hist_color[j], fill=True, alpha=0.5))


    # All simulations
    for i in range(n_layers):
        ax=axs[i, 0]
        n, bins,_ = ax.hist(rho_rng_sort[:,i],bins = 100, alpha=0.5,color=hist_color[j+1])

        ax=axs[i, 1]
        n, bins,_ = ax.hist(radius_rng_sort[:,i],bins = 100, alpha=0.5,color=hist_color[j+1])

        if nhalf_rng_sort[:,i].all() != 0 or len(nhalf_rng_sort[:,i]) !=0: 
            ax=axs[i, 2]
            n, bins,_ = ax.hist(nhalf_rng_sort[:,i],bins = int(np.max(nhalf_rng_sort[:,i])-np.min(nhalf_rng_sort[:,i]))+1, alpha=0.5,color=hist_color[j+1])

    labels.append('All ('+str(len(final_metric))+' models)')
    handles.append(Patch(edgecolor=hist_color[j+1], facecolor=hist_color[j+1], fill=True, alpha=0.5))



    fig.legend(handles=handles,labels=labels, loc = 'upper center', ncol=n_layers+1,fontsize=13)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)    
    plt.show()


    if saving_dir is not None: plt.savefig(saving_dir+'histograms_'+thresh_name+'.png', dpi=600)



    return rho, radius, n_half, fig








##########################################################################################################################
##########################################################################################################################



def Gaussian_func(x, a, x0, sigma):


    return a * np.exp(-(x-x0)**2/(2*sigma**2))


##########################################################################################################################
##########################################################################################################################
