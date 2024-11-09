import numpy as np


def ICET(eta, c_sld, c_2sld, c_lyte, k0, E_A, T, act_R=None,
       act_lyte=None, lmbda=None, alpha=None):
    gamma_ts = (1./(1-c_sld- c_2sld))
    ecd = (k0 * c_lyte**(1-alpha)
           * c_sld**(alpha) / gamma_ts)
    Rate = ecd * np.exp(-E_A/T + E_A/1) * (np.exp(-alpha*eta/T) - np.exp((1-alpha)*eta/T))
    return Rate
