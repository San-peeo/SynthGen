from lib.lib_dep import *


def DataWriter(param_bulk, param_body,param_int, filename):

   
    """
    Usage
    ----------
    Write configuration parameters to a text file.

    Parameters
    ----------
    param_bulk : list
                 Bulk parameters including radius, GM constant, mass, etc.
    param_body : list
                 Data file parameters including gravity and topography file paths.
    param_int  : list
                 Interior model parameters including density, radius, and rheology information.
    filename   : str, optional
                 Name of the output text file (+complete saving path)




    """
    with open(filename, 'w') as f:

        
        # Write bulk parameters
        f.write("# Bulk parameters:\n")
        f.write(f"ref_radius = {param_bulk[0]}\t[km]\n")
        f.write(f"GM_const = {param_bulk[1]}\t[m^3/sec^2]\n")
        f.write(f"errGM_const = {param_bulk[2]}\n")
        f.write(f"ref_mass = {param_bulk[3]}\t[kg]\n")
        f.write(f"ref_rho = {param_bulk[4]}\t[kg/m^3]\n")
        f.write(f"ref_ang_vel = {param_bulk[5]}\t[rad/sec]\n")
        f.write(f"ref_MoI = {param_bulk[6]}\t(I/MR^2)\n")
        f.write(f"err_MoI = {param_bulk[7]}\t[formal error]\n")
        f.write(f"r_e_fact = {param_bulk[8]}\t[sec]\n")
        f.write(f"r_p_fact = {param_bulk[9]}\t[sec]\n")
        f.write(f"flattening f = {param_bulk[10]}\t[sec]\n")
        f.write(f"ref_period = {param_bulk[11]}\t[sec]\n")

        f.write("\n")
        f.write(f"# ------------------------------------------------\n")
        f.write("\n")

        # Write data files
        f.write("# Data files:\n")
        f.write(f"gravity_file = {param_body[0]}\n")
        f.write(f"topography_file = {param_body[1]}\n")
        f.write(f"topo_factor = {param_body[2]}\n")
        f.write(f"rho_boug = {param_body[7]}\n\n")
        

        f.write("\n")
        f.write(f"# ------------------------------------------------\n")
        f.write("\n")

        # Write interior model parameters
        f.write("# Interior model:\n")
        f.write(f"rho_layers = {param_int[0]}\t[kg/m^3]\n")
        f.write(f"radius_layers = {param_int[1]}\t[km]\n")
        f.write(f"interface_type = {param_int[2]}\n")        
        f.write(f"interface_addinfo = {param_int[3]}\n")        
        f.write(f"rigidity_layers = {param_int[4]}\t[Pa]\n")        
        f.write(f"viscosity_layers = {param_int[5]}\t[Pa s]\n")        
        f.write(f"rheology_layers = {param_int[6]}\n")        
        f.write(f"rheo_addinfo = {param_int[7]}\n")        
        f.write(f"layer_name = {param_int[8]}\n")




##########################################################################################################################
##########################################################################################################################
