import numpy as np


def BV_NMC811_KaranthWeijers24(eta, c_sld, c_lyte, k0, E_A, T, act_R=None,
                    act_lyte=None, lmbda=None, alpha=None):
    """Fitted from the results of McClelland 2023"""
    # Implemented by Ombrini for Karanth and Weijers 2024
    # Ranges from 1 to 6
    # multiplied by k0 in A/m^2
    ecd = k0*(87.15*c_sld**5 + -445.05*c_sld**4 
              + 815.43*c_sld**3 + -669.95*c_sld**2 
              + 236.09*c_sld + -23.67) * (c_lyte)**0.5
    Rate = ecd * (np.exp(-alpha*eta/T) - np.exp((1-alpha)*eta/T))
    return Rate
