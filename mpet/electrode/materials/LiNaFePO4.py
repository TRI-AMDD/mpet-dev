import numpy as np

def mu(x, y, Lxs, Lys, Lxy, Lxyv):
    mu = np.log(x / (1-y-x))
    kx = 0
    for Lx in Lxs:
        mu += Lx * ((1-y-2*x)**kx) * ((1-y-2*x) - 2 * kx * (x * (1-y-x)/(1-y-2*x)))
        kx += 1
    ky = 0
    for Ly in Lys:
        mu += - Ly * y * ((1-x-2*y)**ky) * (1 + ky*(1-x-y)/(1-x-2*y))
        ky += 1
    mu += Lxy*y
    mu += Lxyv*y*(1-y-2*x)
    return mu


def LiNaFePO4(self, y, ybar, T, muR_ref):
    """ Ombrini 2024 """
    # muRtheta1 = -self.eokT*0.208
    muRtheta1 = -self.eokT*0.208
    # muRtheta2 = -self.eokT*0.075
    # muRtheta2 = -self.eokT*0.06 # from Ian fitting NFP
    # muRtheta2 = -self.eokT*0.06 # from Ian fitting NFP
    muRtheta2 = -self.eokT*self.get_trode_param('v_2')
    # muRtheta2 = -self.eokT*0.06
    y1, y2 = y
    # Omga = self.get_trode_param('Omega_a')
    Lnali = self.get_trode_param('Omega_b')
    Lnalivac = self.get_trode_param('Omega_c')
    # L_Li = [Omga]
    L_Li = [3.97034583, 0.09673699, 1.11037291, -0.15444768]
    L_Na = [0.94502646, 8.02136736, 5.35420982, -15.21264346, -4.0081790, 7.62295359] # big dV from paper of GITT 10n part
    # L_Na = [-1, 7.1, 2.9, -11.2, -1.67, 4.57] # small dV from fitting NFP Ian
    # L_Na = [  1.11101506 ,  9.0959113 ,  15.71249686 ,  1.72103692, -10.8449572, -9.38755838] # based on big difference 600 nm Part Ian

    # L_Na = [3]

    muLihom = mu(y1, y2, L_Li, L_Na, Lnali, Lnalivac)
    muNahom = mu(y2, y1, L_Na, L_Li, Lnali, Lnalivac)

    muR1nonHomog, muR2nonHomog = self.general_non_homog(y, ybar)
    muR1 = muLihom + muR1nonHomog
    muR2 = muNahom + muR2nonHomog
    # interaction between the two phases
    actR1 = np.exp(muR1/T)
    actR2 = np.exp(muR2/T)
    muR1 += muRtheta1 + muR_ref
    muR2 += muRtheta2 + muR_ref
    return (muR1, muR2), (actR1, actR2)


