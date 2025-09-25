from lib.lib_dep import *

def TopThreshold_Analysis_nodeg(rho_rng_sort,radius_rng_sort, final_metric, threshold_arr, layer_name=[], saving_dir=None):

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

    n_layers = np.shape(rho_rng_sort)[1]

    hist_color=matlab_colors

    rho     = np.zeros((len(threshold_arr),n_layers,2))
    radius  = np.zeros((len(threshold_arr),n_layers,2))
    thresh_name=''

    # ------------------------------------------------------------------------------------------------------

    # Selecting top [threshold] %
    fig, axs = plt.subplots(n_layers, 2, figsize=(11,9))
    labels=[]
    handles = []
    bins_all1 = []
    bins_all2 = []


    # All simulations
    for i in range(n_layers):
        ax=axs[i, 0]
        if np.std(rho_rng_sort[:,i])==0: bins =  'fd'
        else: bins =np.arange(np.min(rho_rng_sort[:,i]),np.max(rho_rng_sort[:,i])+1,1,dtype=np.int64)
        n, bins,_ = ax.hist(rho_rng_sort[:,i],bins=bins, alpha=0.5,color=hist_color[1])
        bins_all1.append(bins)
        
        ax=axs[i, 1]
        if np.std(radius_rng_sort[:,i])==0: bins =  'fd'
        else: bins =np.arange(np.min(radius_rng_sort[:,i]),np.max(radius_rng_sort[:,i])+1,1,dtype=np.int64)
        n, bins,_ = ax.hist(radius_rng_sort[:,i],bins=bins, alpha=0.5,color=hist_color[1])
        bins_all2.append(bins)


    labels.append('All ('+str(len(final_metric))+' models)')
    handles.append(Patch(edgecolor=hist_color[1], facecolor=hist_color[1], fill=True, alpha=0.5))







    for j,thresh in enumerate(threshold_arr):

        best_idx_arr = np.where(final_metric >= 1-thresh)
        if np.shape(best_idx_arr)[1]!=0:
            best_idx = best_idx_arr[0][0]
        else:
            print("Top "+ str(thresh*100) + "% models: ", 0, "/", len(final_metric))
            continue
        
        rho_rng_valid_sort_best      = rho_rng_sort[best_idx:]
        radius_rng_valid_sort_best   = radius_rng_sort[best_idx:]

        print("Top "+ str(thresh*100) + "% models: ", np.shape(best_idx_arr)[1], "/", len(final_metric))


        thresh_name+= str(np.round(thresh*100))
        if j!=np.shape(threshold_arr)[0]: thresh_name +='_'

        # ------------------------------------------------------------------------------------------------------

        # Histogram and Analysis


        for i in range(n_layers):

            ax=axs[i, 0]
            n, bins,_ = ax.hist(rho_rng_valid_sort_best[:,i],bins = bins_all1[i], alpha=1,color=hist_color[j])

            # Fitting Distribution:
            if np.std(rho_rng_valid_sort_best[:,i]) != 0:
                try:
                    func=Gaussian_func
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt_G, _ = curve_fit(func, bin_centers, n, p0=[np.max(n), np.mean(rho_rng_valid_sort_best[:,i]), np.std(rho_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(rho_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(rho_rng_valid_sort_best[:,i]), np.inf)))
                    r2_G = R2_gof(n,func(bin_centers,*popt_G))
                    mu,sigma = popt_G[1],popt_G[2]
                    rho[j,i,0] = mu
                    rho[j,i,1] = sigma
                    popt=popt_G

                    if r2_G<0.9:
                        func=Skew_func
                        popt, _ = curve_fit(func, bin_centers, n, p0=[np.max(n), np.mean(rho_rng_valid_sort_best[:,i]), np.std(rho_rng_valid_sort_best[:,i]),+10],bounds=((-np.inf, np.min(rho_rng_valid_sort_best[:,i]), -np.inf,-np.inf),(np.inf, np.max(rho_rng_valid_sort_best[:,i]), np.inf,np.inf)))
                        r2 = R2_gof(n,func(bin_centers,*popt))
                        mu,sigma,gamma = popt[1],popt[2],popt[3]
                        delta = gamma/(np.sqrt(1+gamma**2))
                        rho[j,i,0] = mu + sigma*(np.sqrt(2/np.pi)*delta - (1-np.pi/4)*(np.sqrt(2/np.pi)*delta)**3/(1-2/np.pi*delta**2) - np.sign(gamma)/2*np.exp(-2*np.pi/np.abs(gamma)) )
                        rho[j,i,1] = sigma*np.sqrt(1-2*delta**2/np.pi)


                        if r2<r2_G:
                            func=Gaussian_func
                            popt=popt_G
                            r2=r2_G
                            mu,sigma = popt[1],popt[2]
                            rho[j,i,0] = mu
                            rho[j,i,1] = sigma


                    ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $\rho=%.1f \pm %.1f,\ \textit{R}^2=%.3f$' %(rho[j,i,0], rho[j,i,1],r2), color=hist_color[j])
                    ax.legend()

                except:
                    print('No Fit (Density)')

            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Density $[kg/m^3]$')
            if layer_name==[]: ax.set_title(r'Layer '+str(i+1))
            else: ax.set_title(layer_name[i])


            # ------------------------------------------------------------------------------------------------------


            ax=axs[i, 1]
            n, bins,_ = ax.hist(radius_rng_valid_sort_best[:,i],bins = bins_all2[i], alpha=1,color=hist_color[j])
            
            # Fitting Distribution:
            if np.std(radius_rng_valid_sort_best[:,i]) != 0:
                try:
                    func = Gaussian_func
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    popt_G, _ = curve_fit(Gaussian_func, bin_centers, n, p0=[np.max(n), np.mean(radius_rng_valid_sort_best[:,i]), np.std(radius_rng_valid_sort_best[:,i])],bounds=((-np.inf, np.min(radius_rng_valid_sort_best[:,i]), -np.inf),(np.inf, np.max(radius_rng_valid_sort_best[:,i]), np.inf)))
                    r2_G = R2_gof(n,Gaussian_func(bin_centers,*popt_G))
                    mu,sigma = popt_G[1],popt_G[2]
                    radius[j,i,0] = mu
                    radius[j,i,1] = sigma
                    popt=popt_G


                    if r2_G<=0.9:
                        func = Skew_func
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        popt, _ = curve_fit(func, bin_centers, n, p0=[np.max(n), np.mean(radius_rng_valid_sort_best[:,i]), np.std(radius_rng_valid_sort_best[:,i]),-10],bounds=((-np.inf, np.min(radius_rng_valid_sort_best[:,i]), -np.inf,-np.inf),(np.inf, np.max(radius_rng_valid_sort_best[:,i]), np.inf,np.inf)))
                        r2 = R2_gof(n,func(bin_centers,*popt))
                        mu,sigma,gamma = popt[1],popt[2],popt[3]
                        delta = gamma/(np.sqrt(1+gamma**2))
                        radius[j,i,0] = mu + sigma*(np.sqrt(2/np.pi)*delta - (1-np.pi/4)*(np.sqrt(2/np.pi)*delta)**3/(1-2/np.pi*delta**2) - np.sign(gamma)/2*np.exp(-2*np.pi/np.abs(gamma)) )
                        radius[j,i,1] = sigma*np.sqrt(1-2*delta**2/np.pi) 
                                
                        if r2<r2_G:
                            func=Gaussian_func
                            popt=popt_G
                            r2=r2_G
                            mu,sigma = popt[1],popt[2]
                            radius[j,i,0] = mu
                            radius[j,i,1] = sigma

                    ax.plot(bin_centers, func(bin_centers,*popt), '--', linewidth=1.5, label=r': $R=%.1f \pm %.1f,\ \textit{R}^2=%.3f$' %(radius[j,i,0], radius[j,i,1],r2), color=hist_color[j])
                    ax.legend()

                except:
                    print('No Fit (Radius)')


            ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
            ax.set_xlabel(r'Radius $[km]$')
            if layer_name==[]: ax.set_title(r'Layer '+str(i+1))
            else: ax.set_title(layer_name[i])


        labels.append(str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)')
        handles.append(Patch(edgecolor=hist_color[j], facecolor=hist_color[j], fill=True, alpha=0.5))






    fig.legend(handles=handles,labels=labels, loc = 'upper center', ncol=n_layers+1,fontsize=13)
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)    
    plt.show()


    if saving_dir is not None: plt.savefig(saving_dir+'histograms_'+thresh_name+'_nodeg.png', dpi=600)



    return rho, radius, fig








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



