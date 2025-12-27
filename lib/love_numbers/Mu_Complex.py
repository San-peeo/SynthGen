from lib.lib_dep import *

# Credit to Dr. Consorzi Anastasia's work


def Mu_Complex(rigidity, viscosity, rheology, rheo_addinfo, frequency):

    """
    Usage
    ----------
    Compute the complex shear modulus mu(omega) for a single layer given a rheological model and driving frequency.
    This routine returns the frequency-dependent complex rigidity used in viscoelastic Love-number computations.

    Parameters
    ----------
    rigidity     : float or mp.mpf
                   Shear modulus (elastic rigidity) of the layer [Pa].
    viscosity    : float or mp.mpf
                   Viscosity of the layer [Pa·s] (used by viscoelastic rheologies).
    rheology     : str
                   Rheology identifier. Supported values include:
                   "maxwell", "andrade", "sundberg", "burgers", "newton", "ocean",
                   "kelvin", "elastic", "cole".
    rheo_addinfo : scalar or sequence
                   Additional rheology-specific parameters:
                   - "andrade": alpha (float)
                   - "sundberg": [alpha, mu2_factor, eta2_factor] (mu2 = mu2_factor*rigidity, eta2 = eta2_factor*viscosity)
                   - "burgers": [param1_factor, param2_factor] (param1 = param1_factor*rigidity, param2 = param2_factor*viscosity)
                   - "cole": (uses internal constants; rheo_addinfo can be None)
                   Provide the appropriate values expected by the chosen rheology.
    frequency    : float,
                   Angular frequency omega (rad/s); automatically converted in pure imaginary number inside this function. 
                   Use 0 or mp.inf for static/elastic limit as needed.

    Output
    ----------
    mu_complex   : complex-like (mpmath.mpc / mp.mpf)
                   Complex shear modulus at the given frequency [Pa].
    """



    s = mp.mpc(0,frequency)



    match rheology:

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "maxwell":
            mu_complex = (rigidity*s)/((s) + rigidity/viscosity)

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "andrade":
            alpha       = rheo_addinfo
            mu_complex  = 1/((1/rigidity)+ (1/(viscosity*s))+(scipy.special.gamma(alpha+1)*(1/rigidity)*((viscosity*s)/rigidity)**(-alpha)))

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "sundberg":
            alpha   = rheo_addinfo[0]
            mu2     = rheo_addinfo[1]*rigidity
            eta2    = rheo_addinfo[2]*viscosity
            b       = 1/(rigidity*(viscosity/rigidity)**alpha)
            mu_complex  = 1 / ( 1 / (s * viscosity) + (1 / rigidity) + (1 / mu2) - ((s * eta2) / (s * eta2 * mu2 + mu2**2)) + ((b* scipy.special.gamma(alpha+1)) / s**(alpha)))

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "burgers":
            param1      = rheo_addinfo[0]*rigidity
            param2      = rheo_addinfo[1]*viscosity
            mu_complex  = (rigidity*s*(s+param1/param2)) / (s**2+s*(rigidity/viscosity+(rigidity+param1)/param2)+ (rigidity*param1)/(viscosity*param2))

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "newton":
            mu_complex  = s*viscosity

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "ocean":
            mu_complex  = s*viscosity

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "kelvin":
            mu_complex  = s*viscosity+rigidity

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "elastic":
            mu_complex  = rigidity

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case "cole":
            dD      = 1.4e-09
            taud    = 3300
            alpha   = 0.53
            S       = np.log(taud*frequency)
            mu_complex  = 1/((1/rigidity + dD*(1-(2/np.pi)*np.atan(np.exp(alpha*S))) - mp.mpc(0,1)*dD*alpha/(np.exp(S*alpha)+np.exp(-S*alpha))+1/(s*viscosity)))
    

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        case _:
            raise ValueError("Rheology type not recognized.")



    return mu_complex
