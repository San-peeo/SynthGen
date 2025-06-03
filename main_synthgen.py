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

body          = 'Ganymede'            # "Mercury", "Earth", "Venus", "Moon","Ganymede"
n_layers      = 7
n_min         = 0
n_max         = 150
r             = 2631.2*1e+3
i_max         = 7
mode          = 'interface'              # 'layer','interface'
load_opt      = False
save_opt      = 'all'            # None,'all', 'total'

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
param_bulk,param_body,param_int, coeffs_grav, coeffs_topo = DataReader(body, n_max, n_layers)


# Extract useful parameters
rho_boug        = param_body[7]
interface_info  = param_int[3]





if coeffs_grav is None or coeffs_topo is None:
    print("ERROR: Gravity/Topography are not available:")

    print(' - Gravity coefficients set to zero')
    coeffs_grav = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=param_bulk[1], r0=param_bulk[0]*1e+3)

    print(' - Topography RNG generation: DeltaH = ' + str(interface_info[-1]) + ' km')
    degrees = np.arange(n_max+1, dtype=float)
    degrees[0] = np.inf
    coeffs_topo = pysh.SHCoeffs.from_random(degrees**(-2), seed=42*n_layers)
    coeffs_topo.set_coeffs(param_bulk[0],0,0)
    surf = coeffs_topo.expand(lmax=n_max,extend=True)
    deltaH_fact = interface_info[-1]/(np.max(surf.data) - np.min(surf.data))
    coeffs_topo.coeffs *= deltaH_fact
    coeffs_topo.set_coeffs(param_bulk[0],0,0)
    print('\n')



# Save folder ( + Checking/Making the saving directory)
rho_layers      = param_int[0]
radius_layers   = param_int[1]
interface_type  = param_int[2]
sub_dir=''
for i in range(n_layers):
    sub_dir += 'i'+str(i+1)+'_'+interface_type[i] + '_r'+str(i+1)+'_'+str(radius_layers[i]) + '_rho'+str(i+1)+'_'+str(rho_layers[i])
    if interface_type[i] == 'dwnbg':
        sub_dir += '_nhalf'+str(i+1)+'_'+str(interface_info[i])
    if interface_type[i] == 'rng':
        sub_dir += '_'+str(interface_info[i])+'km'   
    if i!= n_layers-1:
        sub_dir+='_'


saving_dir = "Results/Synthetic/"+body + "/" + sub_dir    
if not os.path.isdir(saving_dir):
    print("Creating directory:")
    os.makedirs(saving_dir)
else:
    print("Already existing directory:")
print(saving_dir)




print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Synthetic gravitational coefficients generation:

t_start2 = time.time()

coeffs_tot,coeffs_layers = SynthGen(param_bulk,param_int,n_max,coeffs_grav, coeffs_topo,i_max,saving_dir,mode=mode,
                                    save_opt=save_opt,plot_opt=True,load_opt=load_opt,proj_opt=proj_opt,verbose_opt=verbose_opt)

t_end2 = time.time()
print(f"Execution Time: {t_end2 - t_start2:.2f} seconds")


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

if coeffs_tot is not None:

    # Global analysis (U, H, FreeAir, Bouguer):
    [U_matrix, topog_matrix, deltag_freeair, deltag_boug] = Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,r=r,rho_boug=rho_boug,
                                                                            i_max=i_max,saving_dir=saving_dir,plot_opt='multiple',proj_opt=proj_opt,verbose_opt=verbose_opt)


    # Spectrum analysis:
    spectrum = Spectrum(coeffs=[coeffs_tot,*coeffs_layers,coeffs_grav],n_min=n_min,n_max=n_max,
                        plot_opt=True,save_opt=save_opt,saving_dir=saving_dir,verbose_opt=verbose_opt)




# ------------------------------------------------------------------------------------------------------

# Custom plot of the results:


    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    U_matrix.plot(ax=axs[0, 0], colorbar='right',projection=proj_opt, title='Gravitational Potential', cb_label='$m^2/s^2$',cmap=cmap)
    topog_matrix.plot(ax=axs[0, 1], colorbar='right',projection=proj_opt, title='Topography', cb_label='km',cmap=cmap)
    deltag_freeair.plot(ax=axs[1, 0], colorbar='right',projection=proj_opt, title='Free-Air anomalies', cb_label='mGal',cmap=cmap)


    ax=axs[1, 1]

    coeffs=[coeffs_tot,*coeffs_layers,coeffs_grav]
    degree_grav = np.arange(0,n_max+1)

    spectrum=[]
    for coeff in coeffs:
        if coeff.coeffs[0][3,0] != 0:
            spectrum_grav = pysh.spectralanalysis.spectrum(coeff.coeffs,convention='l2norm',unit='per_lm',lmax=n_max)
            spectrum_grav /= (2*degree_grav+1) 
            ax.plot(degree_grav[n_min:],np.sqrt(spectrum_grav[n_min:]), linewidth=2, label=coeff.name)

        spectrum.append(np.sqrt(spectrum_grav))

        ax.set_xlim([0, n_max])
        ax.set_ylim([8e-9, 5e-4])
        ax.set_yscale("log")
        ax.set_xlabel("Degree $n$")
        ax.set_ylabel("Power Spectrum")
        ax.set_xticks(np.arange(0,n_max+10 ,10))
        ax.legend()
        ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5)
        ax.grid(visible=True, which='minor', linestyle='--', linewidth=0.2)



    plt.tight_layout()
    plt.show()
    fig.savefig(saving_dir+"/U_h_FreeAir_spectrum_nmin"+str(n_min)+"_nmax"+str(n_max)+".pdf", dpi=600)
    fig.savefig(saving_dir+"/U_h_FreeAir_spectrum_nmin"+str(n_min)+"_nmax"+str(n_max)+".png", dpi=600)
 





else:
    print('SynthGen ERROR: no synthetic coefficients')
    print('\n')


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
