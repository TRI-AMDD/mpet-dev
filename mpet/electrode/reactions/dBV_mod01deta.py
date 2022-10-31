import numpy as np


def dBV_mod01deta(eta, c_sld, c_lyte, k0, E_A, T, act_R=None,
                  act_lyte=None, lmbda=None, alpha=None):
    ecd = (k0 * c_lyte**(1-alpha)
           * (1.0 - c_sld)**(1 - alpha) * c_sld**alpha)
    Rate = ecd * np.exp(-E_A/T + E_A/1) * (-alpha/T*np.exp(-alpha*eta/T)
                                           - (1-alpha)/T*np.exp((1-alpha)*eta/T))
    return Rate
