from main_library import *
from Planets_ConfigFiles import *

plt.ion()

FreeMemory()

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


t_start = time.time()




# Set up the parameters:

body          = 'Ganymede'            # "Mercury", "Earth", "Venus", "Moon"
n_min         = 0
n_max         = 150
r             = 2631.2*1e+3
i_max         = 7
proj_opt      = ccrs.Mollweide()
verbose_opt   = True






# ------------------------------------------------------------------------------------------------------
########################################################################################################
########################################################################################################
# ------------------------------------------------------------------------------------------------------


# Data importing
print("\n")  
print('Body: ' + body+ '\n')


# Reading Data (Gravity - Topography) from files:
param_bulk,param_body, coeffs_grav, coeffs_topo = DataReader(body, n_max)

if coeffs_grav is None or coeffs_topo is None:
    print("ERROR: Gravity/Topography are not available.")
    exit()




# Extract useful parameters
rho_boug        = param_body[7]


# Save folder ( + Checking/Making the saving directory)
saving_dir = "Results/Real/"+body + "/"    
if not os.path.isdir(saving_dir):
    print("Creating directory:")
    os.makedirs(saving_dir)
else:
    print("Already existing directory:")
print(saving_dir)


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------



# Global analysis (U, H, FreeAir, Bouguer):
Global_Analysis(coeffs_grav=coeffs_grav,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,r=r,rho_boug=rho_boug,
                i_max=i_max,saving_dir=saving_dir,plot_opt='single',proj_opt=proj_opt,verbose_opt=verbose_opt)


# Spectrum analysis:
Spectrum(coeffs=[coeffs_grav],n_min=n_min,n_max=n_max,saving_dir=saving_dir,save_opt='total',plot_opt=True,verbose_opt=verbose_opt)



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
