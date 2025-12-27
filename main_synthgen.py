from model_lib import *


plt.ion()

FreeMemory()

########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


t_start = time.time()




# Set up the parameters:

body          = 'Mercury'              # "Mercury", "Earth", "Venus", "Moon","Ganymede","Ceres"
n_layers      = 4
n_min         = 0
n_max         = 150
r             = [2440.0*1e+3]
i_max         = 7
mode          = 'layer'             # 'layer','interface'
load_opt      = True
save_opt      = 'all'                   # None,'all', 'total'
sub_dir       = 'auto'                  # 'auto' for default naming (interface_rho_R_ name) 

proj_opt      = ccrs.Mollweide(central_longitude=0)


verbose_opt   = True


region = None     # None,    Mercury = [[-180, 180], [0, 90]]



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
ref_radius      = param_bulk[0]
GM_const        = param_bulk[1]
ref_mass        = param_bulk[3]
ref_MoI         = param_bulk[6]
err_MoI         = param_bulk[7]
ref_period      = param_bulk[11]

rho_boug        = param_body[7]


rho_layers          = param_int[0]
radius_layers       = param_int[1]
interface_type      = param_int[2]
interface_addinfo   = param_int[3]
rigid_layers        = param_int[4]
visc_layers         = param_int[5]
rheo_layers         = param_int[6]
rheo_addinfo        = param_int[7]
layers_name         = param_int[8]



if coeffs_grav is None or coeffs_topo is None:
    print("ERROR: Gravity/Topography are not available:")

    print(' - Gravity coefficients set to zero')
    coeffs_grav = pysh.SHGravCoeffs.from_zeros(lmax=n_max, gm=param_bulk[1], r0=param_bulk[0]*1e+3)

    print(' - Topography RNG generation: DeltaH = ' + str(interface_addinfo[-1]) + ' km')
    degrees = np.arange(n_max+1, dtype=float)
    degrees[0] = np.inf
    coeffs_topo = pysh.SHCoeffs.from_random(degrees**(-2), seed=42)
    coeffs_topo.set_coeffs(param_bulk[0],0,0)
    surf = coeffs_topo.expand(lmax=n_max,extend=True)
    deltaH_fact = interface_addinfo[-1]/(np.max(surf.data) - np.min(surf.data))
    coeffs_topo.coeffs *= deltaH_fact
    coeffs_topo.set_coeffs(param_bulk[0],0,0)
    print('\n')



# Save folder ( + Checking/Making the saving directory)
if sub_dir == 'auto':
    rho_layers      = param_int[0]
    radius_layers   = param_int[1]
    interface_type  = param_int[2]
    sub_dir=''
    for i in range(n_layers):
        sub_dir += 'i'+str(i+1)+'_'+interface_type[i] + '_r'+str(i+1)+'_'+str(radius_layers[i]) + '_rho'+str(i+1)+'_'+str(rho_layers[i])
        if interface_type[i] == 'dwnbg':
            sub_dir += '_nhalf'+str(i+1)+'_'+str(np.round(interface_addinfo[i]))
        if interface_type[i] == 'rng':
            sub_dir += '_'+str(interface_addinfo[i])+'km'   
        if i!= n_layers-1:
            sub_dir+='_'

saving_dir = "Results/Synthetic/"+body + "/" + sub_dir    
if not os.path.isdir(saving_dir):
    print("Creating directory:")
    os.makedirs(saving_dir)
else:
    print("Already existing directory:")
print(saving_dir)


DataWriter(param_bulk, param_body, param_int, saving_dir + "/config_parameters.txt")




print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------


# Synthetic gravitational coefficients generation:

t_start2 = time.time()

coeffs_tot,coeffs_layers,surf_list,M_layers = SynthGen(param_int,n_max,coeffs_grav, coeffs_topo,i_max,saving_dir,mode=mode,
                                                        save_opt=save_opt,plot_opt=True,
                                                        load_opt=load_opt,verbose_opt=verbose_opt)


M   = np.sum(M_layers)
MoI_layers,MoI = MomentofInertia(param_int[1],param_int[0])
print("Total mass : " + str(format(M,'.3E')) + " [kg]")
print("Reference mass : " + str(format(ref_mass,'.3E')) + " [kg]\n")
print("Total MoI (norm) : " + str(np.round(MoI/(M*param_int[1][-1]**2*1e+6),3)))
print("Reference MoI (norm) : " + str(np.round(ref_MoI,3)) + " +/- " + str(np.round(param_bulk[7],3)))




t_end2 = time.time()
print(f"SynthGen Execution Time: {t_end2 - t_start2:.2f} seconds")
print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")

# ------------------------------------------------------------------------------------------------------



# Love number generation:


t_start3 = time.time()


[h2,l2,k2] = Love_number_gen(radius_layers, rho_layers, rheo_layers, rigid_layers, visc_layers, rheo_addinfo, ref_period, l=2,saving_dir=saving_dir)
print("h_2 : " + mp.nstr(h2,6))
print("l_2 : " + mp.nstr(l2,6))
print("k_2 : " + mp.nstr(k2,6))



t_end3 = time.time()
print(f"Love number generation Execution Time: {t_end3 - t_start3:.2f} seconds")


print(" ")
print("# ------------------------------------------------------------------------------------------------------")
print("# ------------------------------------------------------------------------------------------------------\n")


# ------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------

# Gravitational maps:
if coeffs_tot is not None:

    # Global analysis (U, H, FreeAir, Bouguer):
    [U_matrix, topog_matrix, deltag_freeair, deltag_boug] = Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=n_min-1,n_max=n_max,r=r,rho_boug=rho_boug,
                                                                            i_max=i_max,saving_dir=saving_dir,plot_opt='single',proj_opt=proj_opt,verbose_opt=verbose_opt)


    # Spectrum analysis:
    spectrum = Spectrum(coeffs=[coeffs_tot,*coeffs_layers,coeffs_grav],n_min=n_min,n_max=n_max,
                        plot_opt=True,save_opt=save_opt,saving_dir=saving_dir,verbose_opt=verbose_opt)






    # ------------------------------------------------------------------------------------------------------

    # Comparison with real data (if available):


    if body != 'Ganymede':

        print("Comparison with real data: ")



        # Reading "Real" data:
        real_dir = "Results/Real/"+body+"/"
        # real_dir = "Data/Mercury/i1_sph_r1_666.577_rho1_8652.52_i2_sphflat_r2_2023.66_rho2_6909.98_i3_dwnbg_r3_2402.61_rho3_3343.35_nhalf3_40_i4_surf_r4_2439.4_rho4_2903.03/"



        U_matrix_real = np.loadtxt(real_dir+'U_matrix_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
        deltag_freeair_real = np.loadtxt(real_dir+'deltag_freeair_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
        deltag_boug_real = np.loadtxt(real_dir+'deltag_boug_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
        # spectrum_real = np.loadtxt(real_dir+'spectrum_grav_'+coeffs_grav.name+'.dat')



        fig, axs = plt.subplots(3, 2, figsize =(11,8),subplot_kw={'projection': proj_opt})
        fig.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers')

        MapPlotting(parent=[fig, axs[0, 0]], values=U_matrix.data, region=region, proj_opt=proj_opt, title=r'$U_{Synth}$', cb_label='$m^2/s^2$',cmap=cmap,clim=[np.min(U_matrix_real),np.max(U_matrix_real)])
        MapPlotting(parent=[fig, axs[0, 1]], values=U_matrix_real, region=region, proj_opt=proj_opt, title=r'$U_{Real}$', cb_label='$m^2/s^2$',cmap=cmap)
        MapPlotting(parent=[fig, axs[1, 0]], values=deltag_freeair.data, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_freeair_real),np.max(deltag_freeair_real)])
        MapPlotting(parent=[fig, axs[1, 1]], values=deltag_freeair_real, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Real}$', cb_label='$mGal$',cmap=cmap)
        MapPlotting(parent=[fig, axs[2, 0]], values=deltag_boug.data, region=region, proj_opt=proj_opt,title=r'$Boug_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_boug_real),np.max(deltag_boug_real)])
        MapPlotting(parent=[fig, axs[2, 1]], values=deltag_boug_real, region=region, proj_opt=proj_opt, title=r'$Boug_{Real}$', cb_label='$mGal$',cmap=cmap)


        plt.tight_layout()
        plt.show()

        fig.savefig(saving_dir+"/Synth_Real.png", dpi=600)




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
