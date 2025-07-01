from lib.lib_dep import *
from lib.synthgen.SynthGen import *
from lib.globe_analysis.Global_Analysis import *
from lib.globe_analysis.Spectrum import *


def PlottingTopAvg(param_bulk,param_int,coeffs_grav,coeffs_topo,n_min,n_max,i_max,rho_boug,body,region,proj_opt,
                   rho,radius,nhalf, real_dir,
                   saving_dir,folder_prefix:Literal['TOP','AVG']='AVG'):




# Plotting the results of the analysis:


    interface_type  = param_int[2]
    n_layers = len(param_int[0])  


    
    print(" ")
    print("Top Model:")


    # Top model plotting (comparison real-synth)
    # rho = rho_arr[-1]
    # radius = radius_arr[-1]
    # nhalf = nhalf_arr[-1]


    plot_dir=folder_prefix+'_'
    for i in range(n_layers):
        plot_dir += 'i'+str(i+1)+'_'+interface_type[i] + '_r'+str(i+1)+'_'+str(radius[i]) + '_rho'+str(i+1)+'_'+str(rho[i])
        if interface_type[i] == 'dwnbg':
            plot_dir += '_nhalf'+str(i+1)+'_'+str(nhalf[i])       
        if i!= n_layers-1:
            plot_dir+='_'




    print(" ")
    print(folder_prefix+" model directory:")
    print(plot_dir)
    print("\n")
    print(folder_prefix+" model parameters:\n")
    for i in range(n_layers):
        print('Layer ' + str(i+1) + ':')
        print('rho = ' + str(rho[i]) + ' kg/m^3')
        print('radius = ' + str(radius[i]) + ' km')
        if interface_type[i] == 'dwnbg':
            print('nhalf = ' + str(nhalf[i]))
        print(" ")

    if not os.path.isdir(saving_dir+plot_dir):
        os.makedirs(saving_dir+plot_dir)


    # Generating the top model coefficients:

    param_int[0] = rho
    param_int[1] = radius
    param_int[2] = interface_type
    param_int[3] = nhalf
    coeffs_tot,coeffs_layers = SynthGen(param_bulk,param_int,n_max,coeffs_grav, coeffs_topo,i_max,saving_dir+plot_dir,mode='layer',
                                        save_opt=True,plot_opt=False,load_opt=False,verbose_opt=False)



    # SynthGen top model (U, H, FreeAir, Bouguer):
    U_synth,_,deltag_freeair_synth,deltag_boug_synth = Global_Analysis(coeffs_grav=coeffs_tot,coeffs_topo=coeffs_topo,n_min=3-1,n_max=n_max,r=radius,rho_boug=rho_boug,
                                                                    i_max=i_max,saving_dir=saving_dir+plot_dir,verbose_opt=False)



    # Real data model (U, H, FreeAir, Bouguer):
    real_dir = "Results/Real/"+body+"/"
    U_real = np.loadtxt(real_dir+'U_matrix_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
    deltag_freeair_real = np.loadtxt(real_dir+'deltag_freeair_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')
    deltag_boug_real = np.loadtxt(real_dir+'deltag_boug_nmin'+str(n_min)+'_nmax'+str(n_max)+'.dat')


    fig_maps, axs = plt.subplots(3, 2, figsize =(11,8),subplot_kw={'projection': proj_opt})
    fig_maps.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  ('+folder_prefix+')')

    MapPlotting(parent=[fig_maps, axs[0, 0]], values=U_synth, region=region, proj_opt=proj_opt, title=r'$U\ {Synth}$', cb_label='$m^2/s^2$',cmap=cmap,clim=[np.min(U_real.data),np.max(U_real.data)])
    MapPlotting(parent=[fig_maps, axs[0, 1]], values=U_real, region=region, proj_opt=proj_opt, title=r'$U\ {Real}$', cb_label='$m^2/s^2$',cmap=cmap)
    MapPlotting(parent=[fig_maps, axs[1, 0]], values=deltag_freeair_synth, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_freeair_real.data),np.max(deltag_freeair_real.data)])
    MapPlotting(parent=[fig_maps, axs[1, 1]], values=deltag_freeair_real, region=region, proj_opt=proj_opt, title=r'$FreeAir_{Real}$', cb_label='$mGal$',cmap=cmap)
    MapPlotting(parent=[fig_maps, axs[2, 0]], values=deltag_boug_synth, region=region, proj_opt=proj_opt,title=r'$Boug_{Synth}$', cb_label='$mGal$',cmap=cmap,clim=[np.min(deltag_boug_real.data),np.max(deltag_boug_real.data)])
    MapPlotting(parent=[fig_maps, axs[2, 1]], values=deltag_boug_real, region=region, proj_opt=proj_opt, title=r'$Boug_{Real}$', cb_label='$mGal$',cmap=cmap)

    plt.tight_layout()
    plt.show()

    fig_maps.savefig(saving_dir+plot_dir+"/Synth_Real_"+folder_prefix+".png", dpi=600)



    # Spectrum analysis:
    _,fig_spectrum = Spectrum(coeffs=[coeffs_tot,*coeffs_layers,coeffs_grav],n_min=2,n_max=n_max,
                        plot_opt=True,save_opt='all',saving_dir=saving_dir+plot_dir,verbose_opt=False)
    fig_spectrum.canvas.manager.set_window_title(body + ': ' + str(n_layers) + ' layers  ('+folder_prefix+')')




    return fig_maps,fig_spectrum