from lib.lib_dep import *
from lib.misc.Volume import *


def LN_Checker(LN_rng, ref_LN, err_LN, error_flag):

    """
    Usage
    ----------
    Verify that a set of computed or sampled Love numbers fall within reference values +/- provided tolerances.
    Intended to validate h2, l2, k2 (index order: 0,1,2) for either real or complex values.

    Parameters
    ----------
    LN_rng      : list or array-like
                  Computed/sampled Love numbers for the tested configuration.
                  Each entry may be real or complex (mpmath/gmpy2 types or Python complex).
                  Expected length 3: [h2, l2, k2].
    ref_LN      : list or array-like
                  Reference Love numbers to compare against (same ordering as LN_rng).
    err_LN      : list or array-like
                  Tolerances for the reference Love numbers. For real values provide floats;
                  for complex values provide errors for real/imag parts as complex or tuples.
    error_flag  : bool
                  Existing global error flag. If True, it stays True; if False and any LN check fails,
                  the function sets it to True and prints diagnostic lines.

    Output
    ----------
    error_flag  : bool
                  Updated error flag (True if any check fails; otherwise unchanged).
    """

    # Checking Love numbers validity:

    for i,LN in enumerate(ref_LN):
        if LN != None:
            
            # Check if ref_LN is complex/real:

            # Real:
            if LN.imag==0:
                dig = len(str(LN))
                LN_rng_real = LN_rng[i].real
                
                if LN_rng_real > ref_LN[i]+err_LN[i] or LN_rng_real < ref_LN[i]-err_LN[i]:
                    print("ERROR: Love Numbers rejected\n")
                    # if i==0: LN_name="h_2"
                    # if i==1: LN_name="l_2"
                    # if i==2: LN_name="k_2"
                    # print(LN_name +": "+mp.nstr(LN_rng_real,dig) +"!= "+ str(ref_LN[i]) +" +/- " +str(err_LN[i]))
                    error_flag=True

            # Complex:
            else:
                if LN_rng[i].real > ref_LN[i].real+err_LN[i].real or LN_rng[i].real < ref_LN[i].real-err_LN[i].real or \
                    LN_rng[i].imag > ref_LN[i].imag+err_LN[i].imag or LN_rng[i].imag < ref_LN[i].imag-err_LN[i].imag :
                    print("ERROR: Love Numbers rejected\n")
                    # if i==0: LN_name="h_2"
                    # if i==1: LN_name="l_2"
                    # if i==2: LN_name="k_2"
                    # print(LN_name +": "+mp.nstr(LN_rng[i]) +"!= "+ str(ref_LN[i]) +" +/- " +str(err_LN[i]))
                    error_flag=True






    return error_flag