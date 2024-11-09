import numpy as np
from .MHC_kfunc import MHC_kfunc


def ECIT(eta, c_sld, c_sld_2, c_lyte, k0, E_A, T, act_R=None,
         act_lyte=None, lmbda=None, alpha=None):
    # See Fraggedakis et al. 2020
    eta_f = eta + T*np.log(c_lyte/c_sld)
    ecd_extras = (1-c_sld-c_sld_2)/(2*np.sqrt(np.pi*lmbda))
    if isinstance(eta, np.ndarray):
        Rate = np.empty(len(eta), dtype=object)
        for i, etaval in enumerate(eta):
            krd = k0*MHC_kfunc(-eta_f[i], lmbda)
            kox = k0*MHC_kfunc(eta_f[i], lmbda)
            Rate[i] = ecd_extras[i]*(krd*c_lyte - kox*c_sld[i])
    else:
        krd = k0*MHC_kfunc(-eta_f, lmbda)
        kox = k0*MHC_kfunc(eta_f, lmbda)
        Rate = np.exp(-E_A/T + E_A/1) * ecd_extras*(krd*c_lyte - kox*c_sld)
    return Rate
