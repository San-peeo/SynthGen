from model_lib import *


plt.ion()

FreeMemory()

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


t_start = time.time()
tracemalloc.start()



# Set up the parameters:

body          = 'Ceres'            # "Mercury", "Earth", "Venus", "Moon","Ceres"
n_layers      = 2
n_min         = 0
n_max         = 18
r             = [482.0*1e+3, 0.075]
i_max         = 7
save_opt      = None
proj_opt      = ccrs.Mollweide()

verbose_opt = False

# ------------------------------------------------------------------------------------------------------


# Results directory:
# saving_dir = "Results/Synthetic/"+body + "/Grid/"+str(n_layers)+"_layers/HgM009/"
# saving_dir = "Results/Synthetic/"+body + "/Grid/"+str(n_layers)+"_layers/HgM009_errMoI_0.00089/"
# saving_dir = "Results/Synthetic/"+body + "/Grid/"+str(n_layers)+"_layers/Validation_errMoI_0.00089/"
saving_dir = "Results/Synthetic/"+body + "/Grid/"+str(n_layers)+"_layers/"


# Real data directory:
real_dir = "Results/Real/"+body+"/"
# real_dir = "Results/Synthetic/Mercury/i1_sph_r1_666.577_rho1_8652.52_i2_sphflat_r2_2023.66_rho2_6909.98_i3_dwnbg_r3_2402.61_rho3_3343.35_nhalf3_40_i4_surf_r4_2439.4_rho4_2903.03/"


# ------------------------------------------------------------------------------------------------------


# metrics_list  = ["\Delta_{\mu}","\Delta_{\sigma}","MAE","RMSE","R^2","SSIM","PSNR","NCC"]
# metrics_list  = ["SSIM","NCC"]
metrics_list  = ["SSIM"]

# maps_list  = ["U","Free-Air","Bouguer"]
maps_list     = ["Bouguer"]


# Decreasing order to see the overlapping histograms:
threshold_arr     = [0.25,0.1,0.05]       # n%
# threshold_arr     = [0.25]       # n%


region =  None    #[[-180, 180], [0, 90]]   # [[lon_min, lon_max], [lat_min, lat_max]] or None
proj_opt     = ccrs.Mollweide(central_longitude=180)  # Projection option

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

ref_mass        = param_bulk[3]
ref_MoI         = param_bulk[6]
err_MoI         = param_bulk[7]

rho_layers          = param_int[0]
radius_layers       = param_int[1]
interface_type      = param_int[2]
interface_addinfo   = param_int[3]
layers_name         = param_int[4]



if not os.path.isdir(saving_dir):
    print("Creating directory:")
    os.makedirs(saving_dir)
    print(saving_dir)

else:
    print("Already existing directory:")
    print(saving_dir)




print(" ")
print("Selected Metrics:")
print(metrics_list)

print(" ")
print("Selected Maps:")
print(maps_list)


print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("#########################################################################################################")
print("# ------------------------------------------------------------------------------------------------------")
print(" ")

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

print("Available runs:")

list_rundir = []
for dirname in os.listdir(saving_dir):
    if re.search("run_", dirname):
            list_rundir.append(dirname)
    print(dirname)
print("\n")

run_n_counts=[]
for run_dir in list_rundir:
    run_n_counts.append(int(run_dir.split("_")[1]))
run_n_counts = np.sort(np.array(run_n_counts))


# ------------------------------------------------------------------------------------------------------

# Setting up the GRID:


n_counts,rho_range, radius_range, nhalf_range = InputRange(n_layers,param_int,run_n_counts)


# Loading Maximum value possible and correspondent ranges
if n_counts=="":
    n_counts=run_n_counts[-1]



# ------------------------------------------------------------------------------------------------------
# If n_counts already exists --> just LOAD

if any(n_counts<=run_n_counts):

    t_start_grid = time.time()

    metrics = np.zeros([8,run_n_counts[n_counts<=run_n_counts][0],3])

    run_name = "run_"+str(run_n_counts[n_counts<=run_n_counts][0])+"/"
    print("Loading metrics from: " + run_name)

    rho_range            = np.loadtxt(saving_dir+run_name+"rho_range.dat")
    radius_range         = np.loadtxt(saving_dir+run_name+"radius_range.dat")
    nhalf_range          = np.loadtxt(saving_dir+run_name+"nhalf_range.dat")

    rho_layers_arr       = np.loadtxt(saving_dir+run_name+"rho_layers_arr.dat")
    radius_layers_arr    = np.loadtxt(saving_dir+run_name+"radius_layers_arr.dat")
    nhalf_arr            = np.loadtxt(saving_dir+run_name+"nhalf_arr.dat")

    delta_mean_arr       = np.loadtxt(saving_dir+run_name+"delta_mean_arr.dat")
    delta_std_arr        = np.loadtxt(saving_dir+run_name+"delta_std_arr.dat")
    RMSE_arr             = np.loadtxt(saving_dir+run_name+"RMSE_arr.dat")
    MAE_arr              = np.loadtxt(saving_dir+run_name+"MAE_arr.dat")
    R2_arr               = np.loadtxt(saving_dir+run_name+"R2_arr.dat")
    SSIM_arr             = np.loadtxt(saving_dir+run_name+"SSIM_arr.dat")
    PSNR_arr             = np.loadtxt(saving_dir+run_name+"PSNR_arr.dat")
    NCC_arr              = np.loadtxt(saving_dir+run_name+"NCC_arr.dat")

    log = np.loadtxt(saving_dir+run_name+"log.txt")

    dbfile = open(saving_dir+run_name+'rng_state', 'rb')   
    random.setstate(pickle.load(dbfile))
    dbfile.close()

    metrics[0] = delta_mean_arr
    metrics[1] = delta_std_arr
    metrics[2] = RMSE_arr
    metrics[3] = MAE_arr
    metrics[4] = R2_arr
    metrics[5] = SSIM_arr
    metrics[6] = PSNR_arr
    metrics[7] = NCC_arr


    # End timing
    t_end_grid = time.time()
    print(f"Loading time: {t_end_grid - t_start_grid:.2f} seconds\n")



# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# If n_counts is > already exists --> LOAD + EVALUATE THE REST:

else:

    t_start_grid = time.time()

    if len(run_n_counts)!=0:

        run_name = "run_"+str(run_n_counts[-1])+"/"
        print("Partial loading metrics from: " + run_name)
        
        if rho_range==[]:
            rho_range            = np.loadtxt(saving_dir+run_name+"rho_range.dat")
            radius_range         = np.loadtxt(saving_dir+run_name+"radius_range.dat")
            nhalf_range          = np.loadtxt(saving_dir+run_name+"nhalf_range.dat")

        rho_layers_arr       = np.loadtxt(saving_dir+run_name+"rho_layers_arr.dat")
        radius_layers_arr    = np.loadtxt(saving_dir+run_name+"radius_layers_arr.dat")
        nhalf_arr            = np.loadtxt(saving_dir+run_name+"nhalf_arr.dat")

        delta_mean_arr       = np.loadtxt(saving_dir+run_name+"delta_mean_arr.dat")
        delta_std_arr        = np.loadtxt(saving_dir+run_name+"delta_std_arr.dat")
        RMSE_arr             = np.loadtxt(saving_dir+run_name+"RMSE_arr.dat")
        MAE_arr              = np.loadtxt(saving_dir+run_name+"MAE_arr.dat")
        R2_arr               = np.loadtxt(saving_dir+run_name+"R2_arr.dat")
        SSIM_arr             = np.loadtxt(saving_dir+run_name+"SSIM_arr.dat")
        PSNR_arr             = np.loadtxt(saving_dir+run_name+"PSNR_arr.dat")
        NCC_arr              = np.loadtxt(saving_dir+run_name+"NCC_arr.dat")

        log = np.loadtxt(saving_dir+run_name+"log.txt")

        dbfile = open(saving_dir+run_name+'rng_state', 'rb')   
        random.setstate(pickle.load(dbfile))
        dbfile.close()



        # Evaluating the rest of the simulations:

        n_counts = n_counts-run_n_counts[-1]
        print(" ")
        print("Evaluating remaining "+str(n_counts)+" simulations:\n")


    else:

        run_name = "run_"+str(n_counts)+"/"
        print("Calculating: " + run_name)
        

        rho_layers_arr      = np.empty([0,n_layers])
        radius_layers_arr   = np.empty([0,n_layers])
        nhalf_arr           = np.empty([0,n_layers])

        delta_mean_arr  = np.empty([0,3])
        delta_std_arr   = np.empty([0,3])
        RMSE_arr        = np.empty([0,3])
        MAE_arr         = np.empty([0,3])
        R2_arr          = np.empty([0,3])
        SSIM_arr        = np.empty([0,3])
        PSNR_arr        = np.empty([0,3])
        NCC_arr         = np.empty([0,3])


        # Evaluating the rest of the simulations:
        print(" ")
        print("Evaluating "+str(n_counts)+" simulations:\n")








    # GRID:



    # Start the evaluation of the new models:

    valid_counter=0
    counter=0
    while valid_counter < n_counts:
        counter += 1

        # -------------------------------------

        # memory leak issues:
        if counter % 100 == 0:
            FreeMemory()

        # -------------------------------------


        # Grid sample extraction:

        print(" ")
        print("Simulation: ", valid_counter+1, "/", n_counts)
        error_flag = False
        valid_flag = False



        
        rho_rng     =  np.zeros([1,n_layers])
        radius_rng  =  np.zeros([1,n_layers])
        nhalf_rng   =  np.zeros([1,n_layers])       
        interface_addinfo_rng = interface_addinfo

        # LAYERS:

        for i in range(0,n_layers):

            if interface_type[i] == 'dwnbg':
                nhalf_rng[0,i] = random.randint(nhalf_range[i][0],nhalf_range[i][1])
                interface_addinfo_rng[i] = nhalf_rng[0,i]
                rho_rng[0,i] = random.uniform(rho_range[i][0],rho_range[i][1])
                radius_rng[0,i] = random.uniform(radius_range[i][0],radius_range[i][1])

            elif interface_type[i] == 'surf':
                rho_rng[0,i] = random.uniform(rho_range[i][0],rho_range[i][1])
                radius_rng[0,i] = radius_layers[i]

            else:
                rho_rng[0,i] = random.uniform(rho_range[i][0],rho_range[i][1])
                radius_rng[0,i] = random.uniform(radius_range[i][0],radius_range[i][1])



        # CORE (Mass and MoI conservation):
        MoI_rng = random.uniform(ref_MoI-err_MoI,ref_MoI+err_MoI)

        [R_core, rho_core] = Solver_M_MoI([ref_mass,MoI_rng],[rho_rng[0,:],radius_rng[0,:]])



        if R_core is None or rho_core is None:
            error_flag=True

        # Core parameters:
        if not error_flag:
            radius_rng[0,0] = R_core
            rho_rng[0,0] = rho_core




        # ----------------------------------------------------------------------------------------------------------------


        # Rounding grid extraction results:
        rho_rng      = np.round(rho_rng)
        radius_rng   = np.round(radius_rng)


        # Checking the extracted parameters:
        print("Radii [km]:          ",radius_rng)
        print("Densities [kg/m^3]:  ",rho_rng)

        # Radius --> Ascending order:
        if all(radius_rng[0,i] <= radius_rng[0,i+1] for i in range(n_layers-1)) is False:
            print("ERROR:Radii conflict\n")
            error_flag = True

        # Density --> Descending order:
        if all(rho_rng[0,i] >= rho_rng[0,i+1] for i in range(n_layers-1)) is False:
            print("ERROR:Densities conflict\n")
            error_flag = True




        # ------------------------------------------------------------------------------------------------------

        # Checking the validity of the interiors parameters combination:


        # Rejection ERROR:
        if error_flag:
            print("PARAMETERS SET: REFUSED\n")
            print("\n")

            del rho_rng 
            del radius_rng 
            del nhalf_rng
            gc.collect()



        # Acceptance VALID
        else:
            print()
            print("PARAMETERS SET: VALID")
            valid_flag=True

        # ------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------

            # param_int_grid = [rho_rng[0,:],radius_rng[0,:],interface_type,nhalf_rng[0,:]]
            param_int_grid = [rho_rng[0,:],radius_rng[0,:],interface_type,interface_addinfo_rng]



            # Synthetic gravitational coefficients generation:

            coeffs_tot,coeffs_layers = SynthGen(param_int_grid,n_max,coeffs_grav,coeffs_topo,i_max,[],
                                                save_opt=None,load_opt=False,plot_opt=False,proj_opt=proj_opt,verbose_opt=verbose_opt)

            # Check SynthGen output:
            if coeffs_tot is None :
                print("ERROR: Layer topography conflict\n")
                continue
            elif np.isnan(coeffs_tot.coeffs).any():
                print("ERROR: NaN values\n")
                continue
            else:
                print("Generated\n")
                print("\n")
                valid_counter += 1


            # Save the interiors parameters:
            rho_layers_arr      = np.vstack((rho_layers_arr,rho_rng))
            radius_layers_arr   = np.vstack((radius_layers_arr,radius_rng))
            nhalf_arr           = np.vstack((nhalf_arr,nhalf_rng))
            



        # ------------------------------------------------------------------------------------------------------
            # Evaluate metrics:

            [delta_mean, delta_std, RMSE, MAE, R2, SSIM, PSNR, NCC] = MetricsEvaluation(["Delta_mean","Delta_std","MAE","RMSE","R^2","SSIM","PSNR","NCC"],
                                                                                        [coeffs_tot,coeffs_topo], real_dir, None,
                                                                                        n_min, n_max, i_max, r, rho_boug)

            delta_mean_arr  = np.vstack((delta_mean_arr, delta_mean))
            delta_std_arr   = np.vstack((delta_std_arr, delta_std))
            RMSE_arr        = np.vstack((RMSE_arr, RMSE))
            MAE_arr         = np.vstack((MAE_arr,MAE))
            R2_arr          = np.vstack((R2_arr, R2))
            SSIM_arr        = np.vstack((SSIM_arr, SSIM))
            PSNR_arr        = np.vstack((PSNR_arr, PSNR))
            NCC_arr         = np.vstack((NCC_arr, NCC))

        # ------------------------------------------------------------------------------------------------------



            del rho_rng 
            del radius_rng 
            del nhalf_rng
            del param_int_grid
            del coeffs_tot
            del coeffs_layers
            
            gc.collect()



    rho_layers_arr = np.squeeze(rho_layers_arr)
    radius_layers_arr = np.squeeze(radius_layers_arr)
    nhalf_arr = np.squeeze(nhalf_arr)

    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------


    print("\n")
    print("Valid simulations: ", valid_counter,"/", counter, " (",valid_counter/counter*100,"%)\n")



    if len(run_n_counts)!=0:
        n_counts = n_counts+run_n_counts[-1]


    run_name = "run_"+str(n_counts)+"/"

    metrics = np.zeros([8,n_counts,3])


    if not os.path.isdir(saving_dir+run_name):
        os.makedirs(saving_dir+run_name)
        print(saving_dir+run_name)

    np.savetxt(saving_dir+run_name+"rho_range.dat",rho_range)
    np.savetxt(saving_dir+run_name+"radius_range.dat",radius_range)
    np.savetxt(saving_dir+run_name+"nhalf_range.dat",nhalf_range)

    np.savetxt(saving_dir+run_name+"rho_layers_arr.dat",rho_layers_arr)
    np.savetxt(saving_dir+run_name+"radius_layers_arr.dat",radius_layers_arr)
    np.savetxt(saving_dir+run_name+"nhalf_arr.dat",nhalf_arr)

    np.savetxt(saving_dir+run_name+"delta_mean_arr.dat",delta_mean_arr)
    np.savetxt(saving_dir+run_name+"delta_std_arr.dat",delta_std_arr)
    np.savetxt(saving_dir+run_name+"RMSE_arr.dat",RMSE_arr)
    np.savetxt(saving_dir+run_name+"MAE_arr.dat",MAE_arr)
    np.savetxt(saving_dir+run_name+"R2_arr.dat",R2_arr)
    np.savetxt(saving_dir+run_name+"SSIM_arr.dat",SSIM_arr)
    np.savetxt(saving_dir+run_name+"PSNR_arr.dat",PSNR_arr)
    np.savetxt(saving_dir+run_name+"NCC_arr.dat",NCC_arr)

    np.savetxt(saving_dir+run_name+"log.txt",[int(valid_counter),int(counter),100*valid_counter/counter])

    state = random.getstate()
    dbfile = open(saving_dir+run_name+'rng_state', 'ab')
    pickle.dump(state, dbfile)                    
    dbfile.close()



    metrics[0] = delta_mean_arr
    metrics[1] = delta_std_arr
    metrics[2] = RMSE_arr
    metrics[3] = MAE_arr
    metrics[4] = R2_arr
    metrics[5] = SSIM_arr
    metrics[6] = PSNR_arr
    metrics[7] = NCC_arr



    # End timing
    t_end_grid = time.time()
    print(f"Grid searching time: {t_end_grid - t_start_grid:.2f} seconds\n")


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


# Selecting chosen metrics

all_metrics  = ["\Delta_{\mu}","\Delta_{\sigma}","MAE","RMSE","R^2","SSIM","PSNR","NCC"]

mask = []
for metric in all_metrics:
    if metric in metrics_list:
        mask.append(False)
    else:
        mask.append(True)

metrics = np.delete(metrics, mask, axis=0)



all_maps = ["U","Free-Air","Bouguer"]
mask = []
for map in all_maps:
    if map in maps_list:
        mask.append(False)
    else:
        mask.append(True)

metrics = np.delete(metrics, mask, axis=2)



# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

#  Metrics plotting:


fig, axs = plt.subplots(len(metrics_list), len(maps_list), figsize=(9,10))
for i,metric_name in enumerate(metrics_list):
    for j,map in enumerate(maps_list):
        if  len(metrics_list) == 1 and len(maps_list) == 1:
            ax = axs
        elif len(metrics_list) == 1:
            ax = axs[j]
        elif len(maps_list) == 1:
            ax = axs[i]
        else:
            ax = axs[i,j]
        ax.set_title(map)
        n, bins,_ = ax.hist(metrics[i,:,j],bins = 'fd')
        ax.grid()

        if j==0: ax.set_ylabel(r'$'+metric_name+'$')


plt.tight_layout()
name = "metrics"+"".join(str("_"+metric) for metric in metrics_list)
name += "".join(str("_"+map) for map in maps_list)
fig.savefig(saving_dir+run_name+name+".png", dpi=600)


# ------------------------------------------------------------------------------------------------------------------------------

# Final metric (ONLY IF ALL of the chosen metrics are normalized [0,1])

final_metric = np.zeros(int(n_counts))

for i in range(int(n_counts)):
    temp=0
    for j in range(np.shape(metrics)[0]):
        temp += np.sum(np.square(metrics[j,i,:]))
    final_metric[i] += np.round(np.sqrt(temp/(3*np.shape(metrics)[0])),3)


# Normalize between [0,1] --> score 0% - 100%
final_metric = (final_metric-np.min(final_metric))/(np.max(final_metric)-np.min(final_metric))






# Plotting the final metric:
plt.figure()
_,bins,_ = plt.hist(final_metric,bins='fd', label='All ('+str(len(final_metric))+' models)',alpha=0.5,color=matlab_colors[1])
plt.title('Final metric')
for j,thresh in enumerate(threshold_arr):
    plt.axvline(x=1-thresh, color='black', linestyle='--', linewidth=0.5)
    best_idx_arr = np.where(final_metric >= 1-thresh)
    plt.hist(final_metric[best_idx_arr],bins=bins, label=str(thresh*100)+'\% ('+str(np.shape(best_idx_arr)[1])+' models)',color=matlab_colors[j])
    plt.legend()
    plt.grid()

plt.savefig(saving_dir+run_name+"final_metrics.png", dpi=600)


# Sorting:
idx = np.argsort(final_metric)
final_metric = np.sort(final_metric,axis=0)



rho_rng_sort       = np.zeros([np.shape(rho_layers_arr)[0],np.shape(rho_layers_arr)[1]])
radius_rng_sort    = np.zeros([np.shape(radius_layers_arr)[0],np.shape(radius_layers_arr)[1]])
nhalf_rng_sort     = np.zeros([np.shape(nhalf_arr)[0],np.shape(nhalf_arr)[1]])



# Cutting away CORE extreme densities (>= 10e+3)
idx_rm=[]
for i in range(len(idx)):
    if rho_layers_arr[idx[i]][0] >= 10e+3: 
        idx_rm.append(i)
        continue
    rho_rng_sort[i,:]     = rho_layers_arr[idx[i]][:]
    radius_rng_sort[i,:]  = radius_layers_arr[idx[i]][:]
    nhalf_rng_sort[i,:]   = (nhalf_arr[idx[i]][:]).astype(int)

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
                                                                final_metric, threshold_arr,layers_name,saving_dir=saving_dir+run_name)
fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers')

# rho_stats,radius_stats,fig = TopThreshold_Analysis_nodeg(rho_rng_sort,radius_rng_sort,
#                                                             final_metric, threshold_arr,layers_name,saving_dir=saving_dir+run_name)
# fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers')


print(" ")
print(" ")
print("# ------------------------------------------------------------------------------------------------------\n")


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


if plot_results == 'top' or plot_results == 'both':
    # Top model plotting (comparison real-synth)
    rho = rho_rng_sort[-1]
    radius = radius_rng_sort[-1]
    nhalf = nhalf_rng_sort[-1]

    PlottingTopAvg(param_int,coeffs_grav,coeffs_topo,3,n_max,i_max,body,region,proj_opt,
                    rho,radius,nhalf, real_dir,
                    saving_dir+run_name,'TOP')
    
    print(" ")

    _,M = Mass(radius,rho)
    MoI = MomentofInertia(radius,rho)

    print("Total mass : " + str(format(M,'.3E')) + " [kg]")
    print("Reference mass : " + str(format(ref_mass,'.3E')) + " [kg]\n")
    print("Total MoI (norm) : " + str(MoI/(M*radius[-1]**2*1e+6)))
    print("Reference MoI (norm) : " + str(np.round(ref_MoI,3)) +" +/- " +str(np.round(err_MoI,3)) )
    print("\n")
    print("# ------------------------------------------------------------------------------------------------------\n")





if plot_results == 'average' or plot_results == 'both':
    # Avg model plotting (comparison real-synth)
    avg_rho = []
    avg_radius = []
    avg_nhalf = []

    # Results model plotting (comparison real-synth)
    for i in range(n_layers):
        avg_rho.append(np.round(rho_stats[int(len(threshold_arr)/2)][i][0],1))
        avg_radius.append(np.round(radius_stats[int(len(threshold_arr)/2)][i][0],1))
        avg_nhalf.append(n_half_stats[int(len(threshold_arr)/2)][i][0])

    avg_radius[-1] = radius_layers[-1]  # Last layer radius is fixed


    PlottingTopAvg(param_int,coeffs_grav,coeffs_topo,3,n_max,i_max,body,region,proj_opt,
                    avg_rho,avg_radius,avg_nhalf,real_dir,
                    saving_dir+run_name,'AVG')

    print(" ")

    _,M = Mass(avg_radius,avg_rho)
    MoI = MomentofInertia(avg_radius,avg_rho)

    print("Total mass : " + str(format(M,'.3E')) + " [kg]")
    print("Reference mass : " + str(format(ref_mass,'.3E')) + " [kg]\n")
    print("Total MoI (norm) : " + str(np.round(MoI/(M*avg_radius[-1]**2*1e+6),3)))
    print("Reference MoI (norm) : " + str(np.round(ref_MoI,3)) + " +/- " + str(np.round(err_MoI,3)) )


# ------------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################
# ------------------------------------------------------------------------------------------------------

print("\n")
print("# ------------------------------------------------------------------------------------------------------\n")





# End timing
t_end = time.time()
print(f"Execution Time: {t_end - t_start:.2f} seconds")




########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


plt.ioff()
plt.show()