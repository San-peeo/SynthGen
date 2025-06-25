from model_lib import *


plt.ion()

FreeMemory()

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


t_start = time.time()




# Set up the parameters:

body          = 'Mercury'            # "Mercury", "Earth", "Venus", "Moon"
n_layers      = 3
n_min         = 0
n_max         = 150
radius        = 2440.0*1e+3
i_max         = 7
load_opt      = True
plot_opt      = 'all'               # 'all','top'        

# Metrics choices: "Delta_mean", "Delta_std", "MAE", "RMSE", "R^2", "SSIM", "PSNR", "NCC", "spectrum"

# metrics_list  = ["Delta_mean","Delta_std","MAE","RMSE","R^2","PSNR","SSIM","NCC"]
# metrics_list  = ["R^2","PSNR","SSIM","NCC"]                     
metrics_list  = ["PSNR","SSIM","NCC"]                     


# Decreasing order to see the overlapping histograms:
threshold_arr     = [0.20,0.15]       # n%
# threshold_arr     = [0.75,0.85,0.95]       # n%



region = None   # [lon_min, lon_max, lat_min, lat_max]
proj_opt      = ccrs.Mollweide()  # Projection option

plot_results = 'both'   # 'top', 'average','both'

# ------------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################
# ------------------------------------------------------------------------------------------------------


# Data importing
print("\n")  
print('Body: ' + body+ '\n')


# Reading Data (Gravity - Topography) from files:
param_bulk,param_body,param_int, coeffs_grav, coeffs_topo = DataReader(body, n_max, n_layers)


# Extract useful parameters
rho_boug        = param_body[7]


rho_layers      = param_int[0]
radius_layers   = param_int[1]
interface_type  = param_int[2]
interface_info  = param_int[3]


print(" ")
print("# ------------------------------------------------------------------------------------------------------")

print("Grid directories: \n")

# Models grid directory
models_dir = ["Results/Synthetic/"+ body + "/Grid/"+str(n_layers)+"_layers/models/"]
print("Models directory:")
for dir in models_dir: print(dir)

print("\n")

print("Analysis directory:")
saving_dir = "Results/Synthetic/"+ body + "/Grid/"+str(n_layers)+"_layers/analysis/"
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)
print(saving_dir)






print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")



# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Reading "Real" data:
real_dir = "Results/Real/"+body+"/"

U_matrix_real = np.loadtxt(real_dir+'U_matrix_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
deltag_freeair_real = np.loadtxt(real_dir+'deltag_freeair_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
deltag_boug_real = np.loadtxt(real_dir+'deltag_boug_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
spectrum_real = np.loadtxt(real_dir+'spectrum_grav_'+coeffs_grav.name+'.dat')


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Loading/Evaluating the metrics


metrics,interiors_parameters = MetricsAnalysis(metrics_list, load_opt, saving_dir, models_dir,
                                               [U_matrix_real,deltag_freeair_real,deltag_boug_real,coeffs_topo,spectrum_real],
                                               [n_min,n_max,radius,i_max,rho_boug])


rho_rng_arr = interiors_parameters[0]
radius_rng_arr = interiors_parameters[1]
nhalf_rng_arr = interiors_parameters[2]



if plot_opt == 'all':
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





# ------------------------------------------------------------------------------------------------------

# Final metric (ONLY IF ALL metrics are normalized [0,1])

final_metric = np.zeros([np.shape(metrics)[1]])
for i in range(np.shape(metrics)[1]):
    final_metric[i] = np.round(np.sqrt(np.sum(np.square(metrics[:,i]))/np.shape(metrics)[0]),3)





if plot_opt == 'all':
    plt.figure()
    _,bins,_ = plt.hist(final_metric,bins=500, label='All ('+str(len(final_metric))+' models)')
    plt.title('Final metric')
    for thresh in threshold_arr:
        plt.axvline(x=1-thresh, color='black', linestyle='--', linewidth=0.5)
        best_idx_arr = np.where(final_metric >= 1-thresh)
        plt.hist(final_metric[best_idx_arr],bins=bins, label=str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)')
        plt.legend()
        plt.grid()


# Sorting:
idx = np.argsort(final_metric)
final_metric = np.sort(final_metric,axis=0)


rho_rng_sort       = np.zeros(np.shape(rho_rng_arr))
radius_rng_sort    = np.zeros(np.shape(radius_rng_arr))
nhalf_rng_sort     = np.zeros(np.shape(nhalf_rng_arr))

# for i in range(len(idx)):
#     rho_rng_sort[i,:]     = rho_rng_arr[idx[i]]
#     radius_rng_sort[i,:]  = radius_rng_arr[idx[i]]
#     nhalf_rng_sort[i,:]   = (nhalf_rng_arr[idx[i]]).astype(int)





# Cutting away CORE extreme densities (>= 10e+3)

idx_rm=[]
for i in range(len(idx)):
    if rho_rng_arr[idx[i]][0] >= 10e+3: 
        idx_rm.append(i)
        continue
    rho_rng_sort[i,:]     = rho_rng_arr[idx[i]]
    radius_rng_sort[i,:]  = radius_rng_arr[idx[i]]
    nhalf_rng_sort[i,:]   = (nhalf_rng_arr[idx[i]]).astype(int)

rho_rng_sort = np.delete(rho_rng_sort,idx_rm, axis=0)
radius_rng_sort = np.delete(radius_rng_sort,idx_rm, axis=0)
nhalf_rng_sort = np.delete(nhalf_rng_sort,idx_rm, axis=0)


final_metric = np.delete(final_metric,idx_rm, axis=0)

# ------------------------------------------------------------------------------------------------------
print(" ")
print("Top thresholds analysis:\n")

# Top % threshold analysis

rho_stats,radius_stats,n_half_stats,fig = TopThreshold_Analysis(rho_rng_sort,radius_rng_sort,nhalf_rng_sort,
                                                                final_metric, threshold_arr,
                                                                plot_opt=plot_opt,saving_dir=saving_dir)
fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers')



print(" ")
print(" ")
print("# ------------------------------------------------------------------------------------------------------\n")


# ------------------------------------------------------------------------------------------------------
########################################################################################################
# ------------------------------------------------------------------------------------------------------



if plot_results == 'top' or plot_results == 'both':
   
    print(" ")
    print("Top Model:")


    # Top model plotting (comparison real-synth)
    top_rho = rho_rng_sort[-1]
    top_radius = radius_rng_sort[-1]
    top_nhalf = nhalf_rng_sort[-1]


    top_dir='TOP_'
    for i in range(n_layers):
        top_dir += 'i'+str(i+1)+'_'+interface_type[i] + '_r'+str(i+1)+'_'+str(top_radius[i]) + '_rho'+str(i+1)+'_'+str(top_rho[i])
        if interface_type[i] == 'dwnbg':
            top_dir += '_nhalf'+str(i+1)+'_'+str(top_nhalf[i])       
        if i!= n_layers-1:
            top_dir+='_'




    print(" ")
    print("Top model directory:")
    print(top_dir)
    print("\n")
    print("Top model parameters:\n")
    for i in range(n_layers):
        print('Layer ' + str(i+1) + ':')
        print('rho = ' + str(top_rho[i]) + ' kg/m^3')
        print('radius = ' + str(top_radius[i]) + ' m')
        if interface_type[i] == 'dwnbg':
            print('nhalf = ' + str(top_nhalf[i]))
        print(" ")

    if not os.path.isdir(saving_dir+top_dir):
        os.makedirs(saving_dir+top_dir)


    # Generating the top model coefficients:

    param_int[0] = top_rho
    param_int[1] = top_radius
    param_int[2] = interface_type
    param_int[3] = top_nhalf
    coeffs_tot,coeffs_layers = SynthGen(param_bulk,param_int,n_max,coeffs_grav, coeffs_topo,i_max,saving_dir+top_dir,mode='layer',
                                        save_opt=True,plot_opt=False,load_opt=False,verbose_opt=False)


    # SynthGen top model (U, H, FreeAir, Bouguer):
    U_synth,_,deltag_freeair_synth,deltag_boug_synth = Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=3-1,n_max=n_max,r=radius,rho_boug=rho_boug,
                                                                    i_max=i_max,saving_dir=saving_dir+top_dir,verbose_opt=False)



    # Real data model (U, H, FreeAir, Bouguer):
    U_real,_,deltag_freeair_real,deltag_boug_real = Global_Analysis(coeffs_grav=coeffs_grav,coeffs_topo=coeffs_topo,n_min=3-1,n_max=n_max,r=radius,rho_boug=rho_boug,
                                                                    i_max=i_max,load_opt=True,saving_dir=real_dir,verbose_opt=False)




    fig, axs = plt.subplots(3, 2, figsize =(11,8),subplot_kw={'projection': proj_opt})
    fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  (TOP)')

    MapPlotting(parent=[fig, axs[0, 0]], values=U_synth, region=region, proj_opt=proj_opt, title=r'$U\ {Synth}$', cb_label='$m^2/s^2$',cmap=cmap,clim=[np.min(U_real.data),np.max(U_real.data)])
    MapPlotting(parent=[fig, axs[0, 1]], values=U_real, region=region, proj_opt=proj_opt, title=r'$U\ {Real}$', cb_label='$m^2/s^2$',cmap=cmap)
    MapPlotting(parent=[fig, axs[1, 0]], values=deltag_freeair_synth, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_freeair_real.data),np.max(deltag_freeair_real.data)])
    MapPlotting(parent=[fig, axs[1, 1]], values=deltag_freeair_real, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Real}$', cb_label='$mGal$',cmap=cmap)
    MapPlotting(parent=[fig, axs[2, 0]], values=deltag_boug_synth, region=region, proj_opt=proj_opt,title=r'$Boug_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_boug_real.data),np.max(deltag_boug_real.data)])
    MapPlotting(parent=[fig, axs[2, 1]], values=deltag_boug_real, region=region, proj_opt=proj_opt, title=r'$Boug_{Real}$', cb_label='$mGal$',cmap=cmap)

    plt.tight_layout()
    plt.show()

    fig.savefig(saving_dir+top_dir+"/Synth_Real_top.png", dpi=600)



    # Spectrum analysis:
    _,fig = Spectrum(coeffs=[coeffs_tot,coeffs_grav],n_min=2,n_max=n_max,
                        plot_opt=True,save_opt='all',saving_dir=saving_dir+top_dir,verbose_opt=False)
    fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  (TOP)')


    print("\n")
    print("# ------------------------------------------------------------------------------------------------------")



# ------------------------------------------------------------------------------------------------------


if plot_results == 'average' or plot_results == 'both':

        
         
    print(" ")
    print(" ")
    print("Results Model:")


    avg_rho = []
    avg_radius = []
    avg_nhalf = []

    # Results model plotting (comparison real-synth)
    for i in range(n_layers):
        avg_rho.append(np.round(rho_stats[i][0],1))
        avg_radius.append(np.round(radius_stats[i][0],1))
        avg_nhalf.append(n_half_stats[i][0])

    avg_radius[-1] = radius_layers[-1]  # Last layer radius is fixed



    avg_dir='AVG_'
    for i in range(n_layers):
        avg_dir += 'i'+str(i+1)+'_'+interface_type[i] + '_r'+str(i+1)+'_'+str(avg_radius[i]) + '_rho'+str(i+1)+'_'+str(avg_rho[i])
        if interface_type[i] == 'dwnbg':
            avg_dir += '_nhalf'+str(i+1)+'_'+str(avg_nhalf[i])       
        if i!= n_layers-1:
            avg_dir+='_'




    print(" ")
    print("Results model directory:")
    print(avg_dir)
    print("\n")
    print("Results model parameters:\n")
    for i in range(n_layers):
        print('Layer ' + str(i+1) + ':')
        print('rho = ' + str(avg_rho[i]) + ' kg/m^3')
        print('radius = ' + str(avg_radius[i]) + ' m')
        if interface_type[i] == 'dwnbg':
            print('nhalf = ' + str(avg_nhalf[i]))
        print(" ")


    if not os.path.isdir(saving_dir+avg_dir):
        os.makedirs(saving_dir+avg_dir)







    # Generating the results model coefficients:

    param_int[0] = avg_rho
    param_int[1] = avg_radius
    param_int[2] = interface_type
    param_int[3] = avg_nhalf
    coeffs_tot,coeffs_layers = SynthGen(param_bulk,param_int,n_max,coeffs_grav, coeffs_topo,i_max,saving_dir+avg_dir,mode='layer',
                                        save_opt=True,plot_opt=False,load_opt=False,verbose_opt=False)


    # SynthGen results model (U, H, FreeAir, Bouguer):
    U_synth,_,deltag_freeair_synth,deltag_boug_synth = Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=3-1,n_max=n_max,r=radius,rho_boug=rho_boug,
                                                                    i_max=i_max,saving_dir=saving_dir+avg_dir,verbose_opt=False)



    # Real data model (U, H, FreeAir, Bouguer):
    U_real,_,deltag_freeair_real,deltag_boug_real = Global_Analysis(coeffs_grav=coeffs_grav,coeffs_topo=coeffs_topo,n_min=3-1,n_max=n_max,r=radius,rho_boug=rho_boug,
                                                                    i_max=i_max,load_opt=True,saving_dir=real_dir,verbose_opt=False)




    fig, axs = plt.subplots(3, 2, figsize =(11,8),subplot_kw={'projection': proj_opt})
    fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  (AVERAGE)')

    MapPlotting(parent=[fig, axs[0, 0]], values=U_synth, region=region, proj_opt=proj_opt, title=r'$U\ {Synth}$', cb_label='$m^2/s^2$',cmap=cmap,clim=[np.min(U_real.data),np.max(U_real.data)])
    MapPlotting(parent=[fig, axs[0, 1]], values=U_real, region=region, proj_opt=proj_opt, title=r'$U\ {Real}$', cb_label='$m^2/s^2$',cmap=cmap)
    MapPlotting(parent=[fig, axs[1, 0]], values=deltag_freeair_synth, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_freeair_real.data),np.max(deltag_freeair_real.data)])
    MapPlotting(parent=[fig, axs[1, 1]], values=deltag_freeair_real, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Real}$', cb_label='$mGal$',cmap=cmap)
    MapPlotting(parent=[fig, axs[2, 0]], values=deltag_boug_synth, region=region, proj_opt=proj_opt,title=r'$Boug_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_boug_real.data),np.max(deltag_boug_real.data)])
    MapPlotting(parent=[fig, axs[2, 1]], values=deltag_boug_real, region=region, proj_opt=proj_opt, title=r'$Boug_{Real}$', cb_label='$mGal$',cmap=cmap)


    plt.tight_layout()
    plt.show()

    fig.savefig(saving_dir+avg_dir+"/Synth_Real_average.png", dpi=600)



    # Spectrum analysis:
    _,fig = Spectrum(coeffs=[coeffs_tot,coeffs_grav],n_min=2,n_max=n_max,
                        plot_opt=True,save_opt='all',saving_dir=saving_dir+avg_dir,verbose_opt=False)
    fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  (AVERAGE)')


    print("\n")
    print("# ------------------------------------------------------------------------------------------------------")





# ------------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################
# ------------------------------------------------------------------------------------------------------




gc.collect()

# End timing
t_end = time.time()
print(f"Execution Time: {t_end - t_start:.2f} seconds")



########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


plt.ioff()
plt.show()
