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

body          = 'Mercury'            # "Mercury", "Earth", "Venus", "Moon"
n_layers      = 3
n_min         = 0
n_max         = 150
r             = 2440.0*1e+3
i_max         = 7
load_opt      = True
save_opt      = 'total'
proj_opt      = ccrs.Mollweide()


verbose_opt = False




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

rho_layers          = param_int[0]
radius_layers       = param_int[1]
interface_type      = param_int[2]
interface_addinfo   = param_int[3]






saving_dir = "Results/Synthetic/"+body + "/Grid/"+str(n_layers)+"_layers/models/"
if not os.path.isdir(saving_dir):
    print("Creating directory:")
    os.makedirs(saving_dir)
    print(saving_dir)

else:
    print("Already existing directory:")
    print(saving_dir)
    print("Already existing models: ", len(os.listdir(saving_dir)))





# Reading "Real" data:
real_dir = "Results/Real/"+body+"/"

U_matrix_real = np.loadtxt(real_dir+'U_matrix_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
deltag_freeair_real = np.loadtxt(real_dir+'deltag_freeair_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
deltag_boug_real = np.loadtxt(real_dir+'deltag_boug_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
spectrum_real = np.loadtxt(real_dir+'spectrum_grav_'+coeffs_grav.name+'.dat')






print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


# Setting up the GRID:


n_counts,rho_range, radius_range, nhalf_range = InputRange(n_layers,param_int)


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------



# GRID:


# Start the evaluation of the new models:

valid_counter=0
counter=0
# for counter in range(0,n_counts):
while valid_counter < n_counts:
    counter += 1

    # -------------------------------------

    # memory leak issues:
    if counter % 100 == 0:
        FreeMemory()

    # -------------------------------------


    # Grid sample extraction:


    print("Simulation: ", valid_counter, "/", n_counts)
    error_flag = False
    valid_flag = False


    nhalf_rng   =  np.zeros([1,n_layers])
    rho_rng     =  np.zeros([1,n_layers])
    radius_rng  =  np.zeros([1,n_layers])


    # LAYERS:

    for i in range(1,n_layers):

        if interface_type[i] == 'dwnbg':
            nhalf_rng[0,i] = random.randint(nhalf_range[i][0],nhalf_range[i][1])

        if interface_type[i] == 'surf':
            rho_rng[0,i] = random.uniform(rho_range[i][0],rho_range[i][1])
            radius_rng[0,i] = radius_layers[i]
        else:
            rho_rng[0,i] = random.uniform(rho_range[i][0],rho_range[i][1])
            radius_rng[0,i] = random.uniform(radius_range[i][0],radius_range[i][1])



    # CORE (Mass and MoI conservation):

    [R_core_min,R_core_max, rho_core_min,rho_core_max] = Solver_M_MoI(param_bulk,[rho_rng[0,:],radius_rng[0,:]])

    # Checking core solution validity:
    if R_core_min is None or  any(x < 0 for x in [R_core_min,R_core_max, rho_core_min,rho_core_max]):
        print("ERROR: Core parameters not valid")
        error_flag = True
        if verbose_opt:
            print("R_core (min,max) =  ", R_core_min," - ",R_core_max)
            print("rho_core (min,max) =  ", rho_core_min," - ",rho_core_max)

    if not error_flag:
        rho_rng[0,0] = random.uniform(rho_core_min,rho_core_max)
        radius_rng[0,0] = random.uniform(R_core_min,R_core_max)




    # Rounding grid extraction results:
    rho_rng      = np.round(rho_rng)
    radius_rng   = np.round(radius_rng)

    # ----------------------------------------------------------------------------------------------------------------


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


        # Name of the folder sub-directory 
        
        sub_dir=''
        for i in range(n_layers):
            sub_dir += 'i'+str(i+1)+''+interface_type[i] + '_r'+str(i+1)+''+str(radius_rng[0,i]) + 'rho'+str(i+1)+''+str(rho_rng[0,i])
            if interface_type[i] == 'dwnbg':
                sub_dir += 'nhalf'+str(i+1)+''+str(nhalf_rng[0,i])       
            if i!= n_layers-1:
                sub_dir+='_'
        

        # Create the sub-directory
        saving_dir_subdir = saving_dir+sub_dir+'/'
        if not os.path.isdir(saving_dir_subdir):
            os.makedirs(saving_dir_subdir)
        else:
            print("Already existing directory\n")
            if load_opt: continue




    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------

        param_int_grid = [rho_rng[0,:],radius_rng[0,:],interface_type,nhalf_rng[0,:]]



        # Synthetic gravitational coefficients generation:

        coeffs_tot,coeffs_layers = SynthGen(param_bulk,param_int_grid,n_max,coeffs_grav,coeffs_topo,i_max,saving_dir_subdir,
                                            save_opt=save_opt,load_opt=load_opt,plot_opt=False,proj_opt=proj_opt,verbose_opt=verbose_opt)

        # Check SynthGen output:
        if coeffs_tot is None :
            print("ERROR: Layer topography conflict\n")
            os.rmdir(saving_dir_subdir)
            continue
        elif np.isnan(coeffs_tot.coeffs).any():
            print("ERROR: NaN values\n")
            os.remove(saving_dir_subdir+'coeffs_tot.dat')
            os.rmdir(saving_dir_subdir)
            continue
        else:
            print("Generated\n")
            print("\n")
            valid_counter += 1


        # Save the interiors parameters:
        np.savetxt(saving_dir_subdir+'rho_layers.dat',rho_rng)
        np.savetxt(saving_dir_subdir+'radius_layers.dat',radius_rng)
        np.savetxt(saving_dir_subdir+'n_half.dat',nhalf_rng)



    # ------------------------------------------------------------------------------------------------------
        # Evaluate metrics:
        metrics_list  = ["Delta_mean","Delta_std","MAE","RMSE","R^2","PSNR","SSIM","NCC"]

        MetricsEvaluation(metrics_list, [coeffs_tot,coeffs_topo], real_dir, saving_dir_subdir,
                          n_min, n_max, i_max, r, rho_boug)



    # ------------------------------------------------------------------------------------------------------



        del rho_rng 
        del radius_rng 
        del nhalf_rng
        del param_int_grid
        del coeffs_tot
        del coeffs_layers
        
        gc.collect()



    


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


print("\n")
print("Valid simulations: ", valid_counter,"/", counter, " (",valid_counter/counter*100,"%)\n")





# ------------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################
# ------------------------------------------------------------------------------------------------------






# End timing
t_end = time.time()
print(f"Execution Time: {t_end - t_start:.2f} seconds")



########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


plt.ioff()
plt.show()