from lib.lib_dep import *
from lib.misc.Volume import *


def RandomSampler_V_tot(param_int,rho_range, radius_range, rigid_range, visc_range, nhalf_range):





    radius_layers       = param_int[1]
    interface_type      = param_int[2]
    interface_addinfo   = param_int[3]


    n_layers = len(radius_layers)



    rho_rng     =  np.zeros([1,n_layers])
    radius_rng  =  np.zeros([1,n_layers])
    nhalf_rng   =  np.zeros([1,n_layers])       
    interface_addinfo_rng = interface_addinfo

    rigid_rng        = np.zeros([1,n_layers])
    visc_rng         = np.zeros([1,n_layers])





    # LAYERS:

    radius_rng[0,-1] = radius_layers[-1]

    for i in reversed(range(0,n_layers)):

        rho_rng[0,i]                = random.uniform(rho_range[i][0],rho_range[i][1])
        rigid_rng[0,i]              = random.uniform(rigid_range[i][0],rigid_range[i][1])
        visc_rng[0,i]               = 10**random.uniform(visc_range[i][0],visc_range[i][1])


        # Extraction of Volume to have an uniform sampling on the radius space:
        V_min,_ = Volume([radius_range[i-1][0],radius_rng[0,i]])
        V_min = V_min[-1]
        V_max,_ = Volume([radius_range[i-1][1],radius_rng[0,i]])
        V_max = V_max[-1]   
        V_rng               = random.uniform(V_min,V_max)
        radius_rng[0,i-1]   = (radius_rng[0,i]**3 - 3/(4*np.pi)*V_rng)**(1/3)


        if interface_type[i] == 'dwnbg':
            nhalf_rng[0,i]              = random.randint(nhalf_range[i][0],nhalf_range[i][1])
            interface_addinfo_rng[i]    = nhalf_rng[0,i]



    # ----------------------------------------------------------------------------------------------------------------


    # Rounding grid extraction results:
    rho_rng      = np.round(rho_rng,1)
    radius_rng   = np.round(radius_rng,1)



    return rho_rng, radius_rng, rigid_rng, visc_rng, interface_addinfo_rng