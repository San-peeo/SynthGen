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
r             = 2440.0*1e+3
i_max         = 7
load_opt      = True
plot_opt      = 'all'               # 'all','top'        

# Metrics choices: "Delta_mean", "Delta_std", "MAE", "RMSE", "R^2", "SSIM", "PSNR", "NCC", "spectrum"

metrics_list  = ["Delta_mean","Delta_std","MAE","RMSE","R^2","PSNR","SSIM","NCC"]
# metrics_list  = ["R^2","PSNR","SSIM","NCC"]                     
# metrics_list  = ["SSIM","NCC"]                     


# Decreasing order to see the overlapping histograms:
threshold_arr     = [0.20,0.15,0.10]       # n%
# threshold_arr     = [0.75,0.85,0.95]       # n%



region = None   # [lon_min, lon_max, lat_min, lat_max]

proj_opt      = ccrs.Mollweide(central_longitude=180)  # Projection option

plot_results = 'average'   # 'top', 'average','both'

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
models_dir = "Results/Synthetic/"+ body + "/Grid/"+str(n_layers)+"_layers/models/"
print("Models directory:")


print("\n")

print("Analysis directory:")
saving_dir = "Results/Synthetic/"+ body + "/Grid/"+str(n_layers)+"_layers/analysis_test/"
if not os.path.isdir(saving_dir):
    os.makedirs(saving_dir)
print(saving_dir)



# "Real" data directory:
real_dir = "Results/Real/"+body+"/"
print("\nReal data directory:")
print(real_dir)



print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")



# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------


metrics,interiors_parameters = MetricsAnalysis2(metrics_list, models_dir, real_dir, [coeffs_topo, n_min, n_max, i_max, r], plot_opt=False)

rho_rng_arr = interiors_parameters[0]
radius_rng_arr = interiors_parameters[1]
nhalf_rng_arr = interiors_parameters[2]



# ------------------------------------------------------------------------------------------------------------------------------

# Final metric (ONLY IF ALL metrics are normalized [0,1])

final_metric = np.zeros([np.shape(metrics)[1]])
for i in range(np.shape(metrics)[1]):
    final_metric[i] = np.round(np.sqrt(np.sum(np.square(metrics[:,i]))/np.shape(metrics)[0]),3)

# Plotting the final metric:
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


# NO CUTTING:
# for i in range(len(idx)):
#     rho_rng_sort[i,:]     = rho_rng_arr[idx[i]]
#     radius_rng_sort[i,:]  = radius_rng_arr[idx[i]]
#     nhalf_rng_sort[i,:]   = (nhalf_rng_arr[idx[i]]).astype(int)




# ------------------------------------------------------------------------------------------------------
print(" ")
print("Top thresholds analysis:\n")

# Top % threshold analysis

rho_stats,radius_stats,n_half_stats,fig = TopThreshold_Analysis(rho_rng_sort,radius_rng_sort,nhalf_rng_sort,
                                                                final_metric, threshold_arr,saving_dir=saving_dir)
fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers')



print(" ")
print(" ")
print("# ------------------------------------------------------------------------------------------------------\n")





# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------



# # Top model plotting (comparison real-synth)
# rho = rho_rng_sort[-1]
# radius = radius_rng_sort[-1]
# nhalf = nhalf_rng_sort[-1]

# PlottingTopAvg(param_bulk,param_int,coeffs_grav,coeffs_topo,3,n_max,i_max,rho_boug,body,region,proj_opt,
#                 rho,radius,nhalf, real_dir,
#                 saving_dir,'TOP')




# Avg model plotting (comparison real-synth)
avg_rho = []
avg_radius = []
avg_nhalf = []

# Results model plotting (comparison real-synth)
for i in range(n_layers):
    avg_rho.append(np.round(rho_stats[i][0],1))
    avg_radius.append(np.round(radius_stats[i][0],1))
    avg_nhalf.append(n_half_stats[i][0])

avg_radius[-1] = radius_layers[-1]  # Last layer radius is fixed


PlottingTopAvg(param_bulk,param_int,coeffs_grav,coeffs_topo,3,n_max,i_max,rho_boug,body,region,proj_opt,
                avg_rho,avg_radius,avg_nhalf,real_dir,
                saving_dir,'AVG')









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
