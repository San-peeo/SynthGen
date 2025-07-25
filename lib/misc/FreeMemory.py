from lib.lib_dep import *

def FreeMemory(verbose_opt=False):

    """
    Usage
    ----------
    Freeing memory and cache

    Parameters
    ----------
    verbose_opt : bool, default = False
                  Verbose option
    """


    if verbose_opt: print("Memory is a free elf!")
    gc.collect()
    os.system("sudo sh -c \"echo 3 >'/proc/sys/vm/drop_caches' && swapoff -a && swapon -a \"")
    if verbose_opt: 
        os.system("printf '\n%s\n' 'Ram-cache and Swap Cleared'\"")
        print("\n")
        print("\n")



##########################################################################################################################
##########################################################################################################################
