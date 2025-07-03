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
n_min         = 3
n_max         = 150
r             = 2439.4*1e+3
i_max         = 7
verbose_opt   = True
plot_opt      = 'multiple'            # 'single','multiple'



n_half = 80                         # Venus
delta_rho = 500                     # Density contrast for Crustal Thickness calculation   [kg/m^3]


region = None   # [lon_min, lon_max, lat_min, lat_max]
proj_opt      = ccrs.Mollweide()  # Projection option

# region = [[-180, 180], [0, 90]]   # Mercury

# region = [[184.5,230], [50,76]]   # Venus, Vinmara_Planitia
# central_lon = (region[0][0] + region[0][1]) / 2 # Region centering
# proj_opt      = ccrs.Mollweide(central_longitude=central_lon)  # Projection option




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
ref_radius      = param_bulk[0]
rho_boug        = param_body[7]


# Save folder ( + Checking/Making the saving directory)
saving_dir = "Results/Real/"+body + "/" 

if region is not None:
   region_dir = input("Enter the region name: ")
   saving_dir += region_dir + "/"

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
                i_max=i_max,region=region,saving_dir=saving_dir,plot_opt=plot_opt,proj_opt=proj_opt,verbose_opt=verbose_opt)


# Spectrum analysis:
Spectrum(coeffs=[coeffs_grav],n_min=n_min,n_max=n_max,
         saving_dir=saving_dir,save_opt='total',plot_opt=True,verbose_opt=verbose_opt)




# ------------------------------------------------------------------------------------------------------



# Global analysis (U, H, FreeAir, Bouguer):
curst_mantle_topog = CrustThickness(coeffs_topo,coeffs_grav,rho_boug,delta_rho,r/1e+3,n_max,i_max,
                                     1,filter_deg=n_half,verbose_opt=verbose_opt)
[fig, ax] = MapPlotting(values=curst_mantle_topog-ref_radius, region=region,
                         proj_opt=proj_opt,title=r'Crust thickness $h_{m-c}(\theta,\phi)$', cb_label='$km$',cmap=cmap)
fig.savefig(saving_dir+"/mantle_crust_interface.png", dpi=600)


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
