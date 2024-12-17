"""These models define individual particles of active material.

This includes the equations for both 1-parameter models and 2-parameters models defining
 - mass conservation (concentration evolution)
 - reaction rate at the surface of the particles
In each model class it has options for different types of particles:
 - homogeneous
 - Fick-like diffusion
 - Cahn-Hilliard (with reaction boundary condition)
 - Allen-Cahn (with reaction throughout the particle)
These models can be instantiated from the mod_cell module to simulate various types of active
materials within a battery electrode.
"""
import daetools.pyDAE as dae
import numpy as np
import scipy.sparse as sprs
import scipy.interpolate as sintrp

import mpet.geometry as geo
import mpet.ports as ports
import mpet.props_am as props_am
import mpet.utils as utils
from mpet.daeVariableTypes import mole_frac_t

class Mod2cat(dae.daeModel):
    def __init__(self, config, trode, vInd, pInd,
                 Name, Parent=None, Description=""):
        super().__init__(Name, Parent, Description)

        self.config = config
        self.trode = trode
        self.ind = (vInd, pInd)

        # Domain
        self.Dmn = dae.daeDomain("discretizationDomain", self, dae.unit(),
                                 "discretization domain")

        # Variables
        self.c1 = dae.daeVariable(
            "c1", mole_frac_t, self,
            "Concentration in 'layer' 1 of active particle", [self.Dmn])
        self.c2 = dae.daeVariable(
            "c2", mole_frac_t, self,
            "Concentration in 'layer' 2 of active particle", [self.Dmn])
        self.cbar = dae.daeVariable(
            "cbar", mole_frac_t, self,
            "Average concentration in active particle")
        self.c1bar = dae.daeVariable(
            "c1bar", mole_frac_t, self,
            "Average concentration in 'layer' 1 of active particle")
        self.c2bar = dae.daeVariable(
            "c2bar", mole_frac_t, self,
            "Average concentration in 'layer' 2 of active particle")
        self.dcbardt = dae.daeVariable("dcbardt", dae.no_t, self, "Rate of particle filling")
        self.dcbar1dt = dae.daeVariable("dcbar1dt", dae.no_t, self, "Rate of particle 1 filling")
        self.dcbar2dt = dae.daeVariable("dcbar2dt", dae.no_t, self, "Rate of particle 2 filling")
        if self.get_trode_param("type") not in ["ACR2"]:
            self.Rxn1 = dae.daeVariable("Rxn1", dae.no_t, self, "Rate of reaction 1")
            self.Rxn2 = dae.daeVariable("Rxn2", dae.no_t, self, "Rate of reaction 2")
            if self.get_trode_param("intralayer_rxn"):
                self.interLayerRxn = dae.daeVariable("interLayerRxn", dae.no_t, self,
                                                    "Rate of reaction between layers",
                                                    [self.Dmn])
        else:
            self.Rxn1 = dae.daeVariable("Rxn1", dae.no_t, self, "Rate of reaction 1", [self.Dmn])
            self.Rxn2 = dae.daeVariable("Rxn2", dae.no_t, self, "Rate of reaction 2", [self.Dmn])

        # Get reaction rate function
        rxnType_1 = config[trode, "rxnType_1"]
        self.calc_rxn_rate_1 = utils.import_function(config[trode, "rxnType_filename"],
                                                   rxnType_1,
                                                   f"mpet.electrode.reactions.{rxnType_1}")
        rxnType_2 = config[trode, "rxnType_2"]
        self.calc_rxn_rate_2 = utils.import_function(config[trode, "rxnType_filename"],
                                                   rxnType_2,
                                                   f"mpet.electrode.reactions.{rxnType_2}")

        # Ports
        self.portInLyte = ports.portFromElyte_mcat(
            "portInLyte", dae.eInletPort, self, "Inlet port from electrolyte")
        self.portInBulk = ports.portFromBulk(
            "portInBulk", dae.eInletPort, self,
            "Inlet port from e- conducting phase")
        self.phi_lyte = self.portInLyte.phi_lyte
        self.T_lyte = self.portInLyte.T_lyte
        self.c_lyte_1 = self.portInLyte.c_lyte_1
        self.c_lyte_2 = self.portInLyte.c_lyte_2
        self.phi_m = self.portInBulk.phi_m

    def get_trode_param(self, item):
        """
        Shorthand to retrieve electrode-specific value
        """
        value = self.config[self.trode, item]
        # check if it is a particle-specific parameter
        if item in self.config.params_per_particle:
            value = value[self.ind]
        return value

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)
        N = self.get_trode_param("N")  # number of grid points in particle
        r_vec, volfrac_vec = geo.get_unit_solid_discr(self.get_trode_param('shape'), N)

        # Prepare noise
        self.noise1 = self.noise2 = None
        if self.get_trode_param("noise"):
            numnoise = self.get_trode_param("numnoise")
            noise_prefac = self.get_trode_param("noise_prefac")
            tvec = np.linspace(0., 1.05*self.config["tend"], numnoise)
            noise_data1 = noise_prefac*np.random.randn(numnoise, N)
            noise_data2 = noise_prefac*np.random.randn(numnoise, N)
            self.noise1 = sintrp.interp1d(tvec, noise_data1, axis=0,
                                          bounds_error=False, fill_value=0.)
            self.noise2 = sintrp.interp1d(tvec, noise_data2, axis=0,
                                          bounds_error=False, fill_value=0.)
        noises = (self.noise1, self.noise2)

        # Figure out mu_O, mu of the oxidized state
        mu_O_1, act_lyte_1 = calc_mu_O(
            self.c_lyte_1(), self.phi_lyte(), self.phi_m(), self.T_lyte(),
            self.config["elyteModelType"])
        
        mu_O_2, act_lyte_2 = calc_mu_O(
            self.c_lyte_2(), self.phi_lyte(), self.phi_m(), self.T_lyte(),
            self.config["elyteModelType"])
        
        mu_O = (mu_O_1, mu_O_2)
        act_lyte = (act_lyte_1, act_lyte_2)

        # Define average filling fractions in particle
        eq1 = self.CreateEquation("c1bar")
        eq2 = self.CreateEquation("c2bar")
        eq1.Residual = self.c1bar()
        eq2.Residual = self.c2bar()
        for k in range(N):
            eq1.Residual -= self.c1(k) * volfrac_vec[k]
            eq2.Residual -= self.c2(k) * volfrac_vec[k]
        eq_cbar = self.CreateEquation("cbar")

        eq_cbar.Residual = self.cbar() - (self.c1bar() + self.c2bar())

        # Define average rate of filling of particle
        eq = self.CreateEquation("dcbardt")
        eq.Residual = self.dcbardt()
        for k in range(N):
            eq.Residual -= (self.c1.dt(k) + self.c2.dt(k)) * volfrac_vec[k]

        # Define average rate of filling of particle for cbar1
        eq = self.CreateEquation("dcbar1dt")
        eq.Residual = self.dcbar1dt()
        for k in range(N):
            eq.Residual -= self.c1.dt(k) * volfrac_vec[k]

        # Define average rate of filling of particle for cbar1
        eq = self.CreateEquation("dcbar2dt")
        eq.Residual = self.dcbar2dt()
        for k in range(N):
            eq.Residual -= self.c2.dt(k) * volfrac_vec[k]

        c1 = np.empty(N, dtype=object)
        c2 = np.empty(N, dtype=object)
        c1[:] = [self.c1(k) for k in range(N)]
        c2[:] = [self.c2(k) for k in range(N)]
        if self.get_trode_param("type") in ["diffn2", "CHR2", "ACR2"]:
            # Equations for 1D particles of 2 field varible
            eta1, eta2, c_surf1, c_surf2 = self.sld_dynamics_1D2var(c1, c2, mu_O, act_lyte,
                                                                    noises)
        elif self.get_trode_param("type") in ["homog2", "homog2_sdn"]:
            # Equations for 0D particles of 2 field variables
            eta1, eta2, c_surf1, c_surf2 = self.sld_dynamics_0D2var(c1, c2, mu_O, act_lyte,
                                                                    noises)

        for eq in self.Equations:
            eq.CheckUnitsConsistency = False

    def sld_dynamics_0D2var(self, c1, c2, muO, act_lyte, noises):
        c1_surf = c1
        c2_surf = c2
        (mu1R_surf, mu2R_surf), (act1R_surf, act2R_surf) = calc_muR(
            (c1_surf, c2_surf), (self.c1bar(), self.c2bar()), self.T_lyte(), self.config,
            self.trode, self.ind)
        eta1 = calc_eta(mu1R_surf, muO)
        eta2 = calc_eta(mu2R_surf, muO)
        eta1_eff = eta1 + self.Rxn1()*self.get_trode_param("Rfilm")
        eta2_eff = eta2 + self.Rxn2()*self.get_trode_param("Rfilm")
        noise1, noise2 = noises
        if self.get_trode_param("noise"):
            eta1_eff += noise1(dae.Time().Value)
            eta2_eff += noise2(dae.Time().Value)
        Rxn1 = self.calc_rxn_rate(
            eta1_eff, c1_surf, self.c_lyte_1(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), act1R_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        Rxn2 = self.calc_rxn_rate(
            eta2_eff, c2_surf, self.c_lyte_1(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), act2R_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        eq1 = self.CreateEquation("Rxn1")
        eq2 = self.CreateEquation("Rxn2")
        eq1.Residual = self.Rxn1() - Rxn1[0]
        eq2.Residual = self.Rxn2() - Rxn2[0]

        eq1 = self.CreateEquation("dc1sdt")
        eq2 = self.CreateEquation("dc2sdt")
        eq1.Residual = self.c1.dt(0) - self.get_trode_param("delta_L")*Rxn1[0]
        eq2.Residual = self.c2.dt(0) - self.get_trode_param("delta_L")*Rxn2[0]
        return eta1[-1], eta2[-1], c1_surf[-1], c2_surf[-1]

    def sld_dynamics_1D2var(self, c1, c2, muO, act_lyte, noises):
        N = self.get_trode_param("N")
        # Equations for concentration evolution
        # Mass matrix, M, where M*dcdt = RHS, where c and RHS are vectors
        Mmat = get_Mmat(self.get_trode_param('shape'), N)
        dr, edges = geo.get_dr_edges(self.get_trode_param('shape'), N)
        muO_1 = muO[0]
        muO_2 = muO[1]
        act_lyte_1 = act_lyte[0]
        act_lyte_2 = act_lyte[1]
        # Get solid particle chemical potential, overpotential, reaction rate
        if self.get_trode_param("type") in ["diffn2", "CHR2"]:
            (mu1R, mu2R), (act1R, act2R) = calc_muR((c1, c2), (self.c1bar(), self.c2bar()),
                                                    self.T_lyte(), self.config, self.trode,
                                                    self.ind)
            c1_surf = c1[-1]
            c2_surf = c2[-1]
            mu1R_surf, act1R_surf = mu1R[-1], act1R[-1]
            mu2R_surf, act2R_surf = mu2R[-1], act2R[-1]
        elif self.get_trode_param("type") in ["ACR2"]:
            (mu1R, mu2R), (act1R, act2R) = calc_muR((c1, c2), (self.c1bar(), self.c2bar()),
                                                    self.T_lyte(), self.config, self.trode,
                                                    self.ind)
            mu1R_surf, act1R_surf = mu1R, act1R
            mu2R_surf, act2R_surf = mu2R, act2R
            
            # Gamma_1 = 0.3
            # Gamma_2 = 1
            # c1_surf = c1 -(c1**(1+Gamma_1))*((1-c1)**Gamma_1)
            # c2_surf = c2 -(c2**(1+Gamma_2))*((1-c2)**Gamma_2)

            c1_surf = c1 
            # c1_surf = 0.09*np.ones_like(c1)
            c2_surf = c2 
            # c2_surf = 0.05*np.ones_like(c2)

        eta1 = calc_eta(mu1R_surf, muO_1)
        eta2 = calc_eta(mu2R_surf, muO_2)
        if self.get_trode_param("type") in ["ACR2"]:
            eta1_eff = np.array([eta1[i]
                                 + self.Rxn1(i)*self.get_trode_param("Rfilm") for i in range(N)])
            eta2_eff = np.array([eta2[i]
                                 + self.Rxn2(i)*self.get_trode_param("Rfilm") for i in range(N)])
        else:
            eta1_eff = eta1 + self.Rxn1()*self.get_trode_param("Rfilm")
            eta2_eff = eta2 + self.Rxn2()*self.get_trode_param("Rfilm")

        Rxn1 = self.calc_rxn_rate_1(
            eta1_eff, c1_surf, c2_surf, self.c_lyte_1(), self.get_trode_param("k0_1"),
            self.get_trode_param("E_A"), self.T_lyte(), act1R_surf, act_lyte_1,
            self.get_trode_param("lambda_1"), self.get_trode_param("alpha"))
        Rxn2 = self.calc_rxn_rate_2(
            eta2_eff, c2_surf, c1_surf, self.c_lyte_2(), self.get_trode_param("k0_1")/self.get_trode_param("k012_ratio"),
            self.get_trode_param("E_A"), self.T_lyte(), act2R_surf, act_lyte_2,
            self.get_trode_param("lambda_2"), self.get_trode_param("alpha"))
        
        if self.get_trode_param("type") in ["ACR2"]:
            for i in range(N):
                eq1 = self.CreateEquation("Rxn1_{i}".format(i=i))
                eq2 = self.CreateEquation("Rxn2_{i}".format(i=i))
                eq1.Residual = self.Rxn1(i) - Rxn1[i]
                eq2.Residual = self.Rxn2(i) - Rxn2[i]
        else:
            eq1 = self.CreateEquation("Rxn1")
            eq2 = self.CreateEquation("Rxn2")
            eq1.Residual = self.Rxn1() - Rxn1
            eq2.Residual = self.Rxn2() - Rxn2

        # Get solid particle fluxes (if any) and RHS
        if self.get_trode_param("type") in ["ACR2"]:
            RHS1 = np.array([self.get_trode_param("delta_L")
                                      * self.Rxn1(i) for i in range(N)])
            RHS2 = np.array([self.get_trode_param("delta_L")
                                      * self.Rxn2(i) for i in range(N)])
        elif self.get_trode_param("type") in ["diffn2", "CHR2"]:
            # Positive reaction (reduction, intercalation) is negative
            # flux of Li at the surface.
            Flux1_bc = -self.Rxn1()
            Flux2_bc = -self.Rxn2()
            Dfunc_name = self.get_trode_param("Dfunc")
            Dfunc = utils.import_function(self.get_trode_param("Dfunc_filename"),
                                          Dfunc_name,
                                          f"mpet.electrode.diffusion.{Dfunc_name}")
            if self.get_trode_param("type") == "CHR2":
                noise1, noise2 = noises
                if self.get_trode_param("shape") == "sphere":
                    area_vec = 4*np.pi*edges**2
                elif self.get_trode_param("shape") == "cylinder":
                    area_vec = 2*np.pi*edges  # per unit height

                Flux1_vec, Flux2_vec = calc_flux_CHR2_multicat(
                    c1, c2, mu1R, mu2R, self.get_trode_param("D1"), 
                    self.get_trode_param("D2"), Dfunc,
                    self.get_trode_param("E_D"), Flux1_bc, Flux2_bc, dr, self.T_lyte(),
                    noise1, noise2)
            
                RHS1 = -np.diff(Flux1_vec * area_vec)
                RHS2 = -np.diff(Flux2_vec * area_vec)

        dc1dt_vec = np.empty(N, dtype=object)
        dc2dt_vec = np.empty(N, dtype=object)
        dc1dt_vec[0:N] = [self.c1.dt(k) for k in range(N)]
        dc2dt_vec[0:N] = [self.c2.dt(k) for k in range(N)]
        LHS1_vec = MX(Mmat, dc1dt_vec)
        LHS2_vec = MX(Mmat, dc2dt_vec)
        if self.get_trode_param("surface_diffusion") and self.get_trode_param("type") in ["ACR2"]:
            surf_diff_vec1 = calc_surf_diff(c1_surf, c2_surf, 
                                            mu1R_surf, mu2R_surf,
                                           self.get_trode_param("D_surf"),
                                           self.get_trode_param("E_D_surf"),
                                           self.T_lyte())
            surf_diff_vec2 = calc_surf_diff(c2_surf, c1_surf, 
                                            mu2R_surf, mu1R_surf,
                                           self.get_trode_param("D_surf"),
                                           self.get_trode_param("E_D_surf"),
                                           self.T_lyte())
            for k in range(N):
                eq1 = self.CreateEquation("dc1sdt_discr{k}".format(k=k))
                eq2 = self.CreateEquation("dc2sdt_discr{k}".format(k=k))
                eq1.Residual = LHS1_vec[k] - RHS1[k] - surf_diff_vec1[k]
                eq2.Residual = LHS2_vec[k] - RHS2[k] - surf_diff_vec2[k]
        else:
            for k in range(N):
                eq1 = self.CreateEquation("dc1sdt_discr{k}".format(k=k))
                eq2 = self.CreateEquation("dc2sdt_discr{k}".format(k=k))
                eq1.Residual = LHS1_vec[k] - RHS1[k]
                eq2.Residual = LHS2_vec[k] - RHS2[k]

        if self.get_trode_param("type") in ["ACR2"]:
            return eta1[-1], eta2[-1], c1_surf[-1], c2_surf[-1]
        else:
            return eta1, eta2, c1_surf, c2_surf
        

class Mod2var(dae.daeModel):
    def __init__(self, config, trode, vInd, pInd,
                 Name, Parent=None, Description=""):
        super().__init__(Name, Parent, Description)

        self.config = config
        self.trode = trode
        self.ind = (vInd, pInd)

        # Domain
        self.Dmn = dae.daeDomain("discretizationDomain", self, dae.unit(),
                                 "discretization domain")

        # Variables
        self.c1 = dae.daeVariable(
            "c1", mole_frac_t, self,
            "Concentration in 'layer' 1 of active particle", [self.Dmn])
        self.c2 = dae.daeVariable(
            "c2", mole_frac_t, self,
            "Concentration in 'layer' 2 of active particle", [self.Dmn])
        self.cbar = dae.daeVariable(
            "cbar", mole_frac_t, self,
            "Average concentration in active particle")
        self.c1bar = dae.daeVariable(
            "c1bar", mole_frac_t, self,
            "Average concentration in 'layer' 1 of active particle")
        self.c2bar = dae.daeVariable(
            "c2bar", mole_frac_t, self,
            "Average concentration in 'layer' 2 of active particle")
        self.dcbardt = dae.daeVariable("dcbardt", dae.no_t, self, "Rate of particle filling")
        self.dcbar1dt = dae.daeVariable("dcbar1dt", dae.no_t, self, "Rate of particle 1 filling")
        self.dcbar2dt = dae.daeVariable("dcbar2dt", dae.no_t, self, "Rate of particle 2 filling")
        self.q_rxn_bar = dae.daeVariable(
            "q_rxn_bar", dae.no_t, self, "Rate of heat generation in particle")
        if self.get_trode_param("type") not in ["ACR2"]:
            self.Rxn1 = dae.daeVariable("Rxn1", dae.no_t, self, "Rate of reaction 1")
            self.Rxn2 = dae.daeVariable("Rxn2", dae.no_t, self, "Rate of reaction 2")
            if self.get_trode_param("intralayer_rxn"):
                self.interLayerRxn = dae.daeVariable("interLayerRxn", dae.no_t, self,
                                                    "Rate of reaction between layers",
                                                    [self.Dmn])
        else:
            self.Rxn1 = dae.daeVariable("Rxn1", dae.no_t, self, "Rate of reaction 1", [self.Dmn])
            self.Rxn2 = dae.daeVariable("Rxn2", dae.no_t, self, "Rate of reaction 2", [self.Dmn])

        # Get reaction rate function
        rxnType = config[trode, "rxnType"]
        self.calc_rxn_rate = utils.import_function(config[trode, "rxnType_filename"],
                                                   rxnType,
                                                   f"mpet.electrode.reactions.{rxnType}")

        # Ports
        self.portInLyte = ports.portFromElyte(
            "portInLyte", dae.eInletPort, self, "Inlet port from electrolyte")
        self.portInBulk = ports.portFromBulk(
            "portInBulk", dae.eInletPort, self,
            "Inlet port from e- conducting phase")
        self.phi_lyte = self.portInLyte.phi_lyte
        self.T_lyte = self.portInLyte.T_lyte
        self.c_lyte = self.portInLyte.c_lyte
        self.phi_m = self.portInBulk.phi_m

    def get_trode_param(self, item):
        """
        Shorthand to retrieve electrode-specific value
        """
        value = self.config[self.trode, item]
        # check if it is a particle-specific parameter
        if item in self.config.params_per_particle:
            value = value[self.ind]
        return value

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)
        N = self.get_trode_param("N")  # number of grid points in particle
        r_vec, volfrac_vec = geo.get_unit_solid_discr(self.get_trode_param('shape'), N)

        # Prepare noise
        self.noise1 = self.noise2 = None
        if self.get_trode_param("noise"):
            numnoise = self.get_trode_param("numnoise")
            noise_prefac = self.get_trode_param("noise_prefac")
            tvec = np.linspace(0., 1.05*self.config["tend"], numnoise)
            noise_data1 = noise_prefac*np.random.randn(numnoise, N)
            noise_data2 = noise_prefac*np.random.randn(numnoise, N)
            self.noise1 = sintrp.interp1d(tvec, noise_data1, axis=0,
                                          bounds_error=False, fill_value=0.)
            self.noise2 = sintrp.interp1d(tvec, noise_data2, axis=0,
                                          bounds_error=False, fill_value=0.)
        noises = (self.noise1, self.noise2)

        # Figure out mu_O, mu of the oxidized state
        mu_O, act_lyte = calc_mu_O(
            self.c_lyte(), self.phi_lyte(), self.phi_m(), self.T_lyte(),
            self.config["elyteModelType"])

        # Define average filling fractions in particle
        eq1 = self.CreateEquation("c1bar")
        eq2 = self.CreateEquation("c2bar")
        eq1.Residual = self.c1bar()
        eq2.Residual = self.c2bar()
        for k in range(N):
            eq1.Residual -= self.c1(k) * volfrac_vec[k]
            eq2.Residual -= self.c2(k) * volfrac_vec[k]
        eq_cbar = self.CreateEquation("cbar")
        eq_cbar.Residual = self.cbar() - (self.c1bar() + self.c2bar())

        # Define average rate of filling of particle
        eq = self.CreateEquation("dcbardt")
        eq.Residual = self.dcbardt()
        for k in range(N):
            eq.Residual -= (self.c1.dt(k) + self.c2.dt(k)) * volfrac_vec[k]

        # Define average rate of filling of particle for cbar1
        eq = self.CreateEquation("dcbar1dt")
        eq.Residual = self.dcbar1dt()
        for k in range(N):
            eq.Residual -= self.c1.dt(k) * volfrac_vec[k]

        # Define average rate of filling of particle for cbar1
        eq = self.CreateEquation("dcbar2dt")
        eq.Residual = self.dcbar2dt()
        for k in range(N):
            eq.Residual -= self.c2.dt(k) * volfrac_vec[k]

        c1 = np.empty(N, dtype=object)
        c2 = np.empty(N, dtype=object)
        c1[:] = [self.c1(k) for k in range(N)]
        c2[:] = [self.c2(k) for k in range(N)]
        if self.get_trode_param("type") in ["diffn2", "CHR2", "ACR2"]:
            # Equations for 1D particles of 2 field varible
            eta1, eta2, c_surf1, c_surf2 = self.sld_dynamics_1D2var(c1, c2, mu_O, act_lyte,
                                                                    noises)
        elif self.get_trode_param("type") in ["homog2", "homog2_sdn"]:
            # Equations for 0D particles of 2 field variables
            eta1, eta2, c_surf1, c_surf2 = self.sld_dynamics_0D2var(c1, c2, mu_O, act_lyte,
                                                                    noises)

        # Define average rate of heat generation
        eq = self.CreateEquation("q_rxn_bar")
        if self.config["entropy_heat_gen"]:
            eq.Residual = self.q_rxn_bar() - self.dcbar1dt() * \
                (eta1 - self.T_lyte()*(np.log(c_surf1/(1-c_surf1))-1/self.c_lyte())) \
                - self.dcbar2dt() * (eta2 - self.T_lyte()
                                                * (np.log(c_surf2/(1-c_surf2))-1/self.c_lyte()))
        else:
            eq.Residual = self.q_rxn_bar() -  self.dcbar1dt() * eta1 \
                - self.dcbar2dt() * eta2

        for eq in self.Equations:
            eq.CheckUnitsConsistency = False

    def sld_dynamics_0D2var(self, c1, c2, muO, act_lyte, noises):
        c1_surf = c1
        c2_surf = c2
        (mu1R_surf, mu2R_surf), (act1R_surf, act2R_surf) = calc_muR(
            (c1_surf, c2_surf), (self.c1bar(), self.c2bar()), self.T_lyte(), self.config,
            self.trode, self.ind)
        eta1 = calc_eta(mu1R_surf, muO)
        eta2 = calc_eta(mu2R_surf, muO)
        eta1_eff = eta1 + self.Rxn1()*self.get_trode_param("Rfilm")
        eta2_eff = eta2 + self.Rxn2()*self.get_trode_param("Rfilm")
        noise1, noise2 = noises
        if self.get_trode_param("noise"):
            eta1_eff += noise1(dae.Time().Value)
            eta2_eff += noise2(dae.Time().Value)
        Rxn1 = self.calc_rxn_rate(
            eta1_eff, c1_surf, self.c_lyte(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), act1R_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        Rxn2 = self.calc_rxn_rate(
            eta2_eff, c2_surf, self.c_lyte(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), act2R_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        eq1 = self.CreateEquation("Rxn1")
        eq2 = self.CreateEquation("Rxn2")
        eq1.Residual = self.Rxn1() - Rxn1[0]
        eq2.Residual = self.Rxn2() - Rxn2[0]

        eq1 = self.CreateEquation("dc1sdt")
        eq2 = self.CreateEquation("dc2sdt")
        eq1.Residual = self.c1.dt(0) - self.get_trode_param("delta_L")*Rxn1[0]
        eq2.Residual = self.c2.dt(0) - self.get_trode_param("delta_L")*Rxn2[0]
        return eta1[-1], eta2[-1], c1_surf[-1], c2_surf[-1]

    def sld_dynamics_1D2var(self, c1, c2, muO, act_lyte, noises):
        N = self.get_trode_param("N")
        # Equations for concentration evolution
        # Mass matrix, M, where M*dcdt = RHS, where c and RHS are vectors
        Mmat = get_Mmat(self.get_trode_param('shape'), N)
        dr, edges = geo.get_dr_edges(self.get_trode_param('shape'), N)

        # Get solid particle chemical potential, overpotential, reaction rate
        if self.get_trode_param("type") in ["diffn2", "CHR2"]:
            (mu1R, mu2R), (act1R, act2R) = calc_muR((c1, c2), (self.c1bar(), self.c2bar()),
                                                    self.T_lyte(), self.config, self.trode,
                                                    self.ind)
            c1_surf = c1[-1]
            c2_surf = c2[-1]
            mu1R_surf, act1R_surf = mu1R[-1], act1R[-1]
            mu2R_surf, act2R_surf = mu2R[-1], act2R[-1]
        elif self.get_trode_param("type") in ["ACR2"]:
            (mu1R, mu2R), (act1R, act2R) = calc_muR((c1, c2), (self.c1bar(), self.c2bar()),
                                                    self.T_lyte(), self.config, self.trode,
                                                    self.ind)
            mu1R_surf, act1R_surf = mu1R, act1R
            mu2R_surf, act2R_surf = mu2R, act2R
            c1_surf = c1
            c2_surf = c2
        eta1 = calc_eta(mu1R_surf, muO)
        eta2 = calc_eta(mu2R_surf, muO)
        if self.get_trode_param("type") in ["ACR2"]:
            eta1_eff = np.array([eta1[i]
                                 + self.Rxn1(i)*self.get_trode_param("Rfilm") for i in range(N)])
            eta2_eff = np.array([eta2[i]
                                 + self.Rxn2(i)*self.get_trode_param("Rfilm") for i in range(N)])
        else:
            eta1_eff = eta1 + self.Rxn1()*self.get_trode_param("Rfilm")
            eta2_eff = eta2 + self.Rxn2()*self.get_trode_param("Rfilm")
        if self.get_trode_param("lambda_1") is not None:
            Rxn1 = self.calc_rxn_rate(
                eta1_eff, c1_surf, self.c_lyte(), self.get_trode_param("k0_1"),
                self.get_trode_param("E_A"), self.T_lyte(), act1R_surf, act_lyte,
                self.get_trode_param("lambda_1"), self.get_trode_param("alpha"))
            Rxn2 = self.calc_rxn_rate(
                eta2_eff, c2_surf, self.c_lyte(), self.get_trode_param("k0_2"),
                self.get_trode_param("E_A"), self.T_lyte(), act2R_surf, act_lyte,
                self.get_trode_param("lambda_2"), self.get_trode_param("alpha"))
        else:
            Rxn1 = self.calc_rxn_rate(
                eta1_eff, c1_surf, self.c_lyte(), self.get_trode_param("k0"),
                self.get_trode_param("E_A"), self.T_lyte(), act1R_surf, act_lyte,
                self.get_trode_param("lambda"), self.get_trode_param("alpha"))
            Rxn2 = self.calc_rxn_rate(
                eta2_eff, c2_surf, self.c_lyte(), self.get_trode_param("k0"),
                self.get_trode_param("E_A"), self.T_lyte(), act2R_surf, act_lyte,
                self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        if self.get_trode_param("type") in ["ACR2"]:
            for i in range(N):
                eq1 = self.CreateEquation("Rxn1_{i}".format(i=i))
                eq2 = self.CreateEquation("Rxn2_{i}".format(i=i))
                eq1.Residual = self.Rxn1(i) - Rxn1[i]
                eq2.Residual = self.Rxn2(i) - Rxn2[i]
        else:
            eq1 = self.CreateEquation("Rxn1")
            eq2 = self.CreateEquation("Rxn2")
            eq1.Residual = self.Rxn1() - Rxn1
            eq2.Residual = self.Rxn2() - Rxn2

        # Get solid particle fluxes (if any) and RHS
        if self.get_trode_param("type") in ["ACR2"]:
            RHS1 = np.array([self.get_trode_param("delta_L")
                                      * self.Rxn1(i) for i in range(N)])
            RHS2 = np.array([self.get_trode_param("delta_L")
                                      * self.Rxn2(i) for i in range(N)])
        elif self.get_trode_param("type") in ["diffn2", "CHR2"]:
            # Positive reaction (reduction, intercalation) is negative
            # flux of Li at the surface.
            Flux1_bc = self.Rxn1()
            Flux2_bc = self.Rxn2()
            Dfunc_name = self.get_trode_param("Dfunc")
            Dfunc = utils.import_function(self.get_trode_param("Dfunc_filename"),
                                          Dfunc_name,
                                          f"mpet.electrode.diffusion.{Dfunc_name}")
            if self.get_trode_param("type") == "CHR2":
                noise1, noise2 = noises
                if self.get_trode_param("shape") == "sphere":
                    area_vec = 4*np.pi*edges**2
                elif self.get_trode_param("shape") == "cylinder":
                    area_vec = 2*np.pi*edges  # per unit height

                if self.get_trode_param("intralayer_rxn"):
                    lambda_m = self.get_trode_param("lambda_m")
                    gamma_ts = 1/((1-c1)*(1-c2))
                    kinterlayer = self.get_trode_param("kintra")
                    delta_mu = mu2R - mu1R - np.log(c2/c1)
                    interLayerRxn = kinterlayer/gamma_ts\
                        * (c1*np.exp(-(((lambda_m+delta_mu)**2)/(4*lambda_m)))
                            - c2*np.exp(-(((lambda_m-delta_mu)**2)/(4*lambda_m))))
                    # interLayerRxn = kinterlayer/gamma_ts\
                    #     * (- c2*np.exp(-(((lambda_m-delta_mu)**2)/(4*lambda_m))))
                    # interLayerRxn = np.zeros(N)
                    RxnTerm1 = -interLayerRxn
                    RxnTerm2 = interLayerRxn
                    for i in range(N):
                        eq = self.CreateEquation("interLayerRxn{i}".format(i=i))
                        eq.Residual = self.interLayerRxn(i) - interLayerRxn[i]

                    RHS1 = RxnTerm1
                    RHS2 = RxnTerm2
                    Flux1_bc += -interLayerRxn[-1]
                    Flux2_bc += interLayerRxn[-1]

                    Flux1_vec, Flux2_vec = calc_flux_CHR2_multisub(
                        c1, c2, mu1R, mu2R, self.get_trode_param("D1"), self.get_trode_param("D2"),
                        Dfunc, self.get_trode_param("E_D"), Flux1_bc, Flux2_bc, dr, self.T_lyte(),
                        noise1, noise2)
                    RHS1 += -np.diff(Flux1_vec * area_vec)
                    RHS2 += -np.diff(Flux2_vec * area_vec)
                else:
                    Flux1_vec, Flux2_vec = calc_flux_CHR2(
                        c1, c2, mu1R, mu2R, self.get_trode_param("D"), Dfunc,
                        self.get_trode_param("E_D"), Flux1_bc, Flux2_bc, dr, self.T_lyte(),
                        noise1, noise2)
                
                    RHS1 = -np.diff(Flux1_vec * area_vec)
                    RHS2 = -np.diff(Flux2_vec * area_vec)

        dc1dt_vec = np.empty(N, dtype=object)
        dc2dt_vec = np.empty(N, dtype=object)
        dc1dt_vec[0:N] = [self.c1.dt(k) for k in range(N)]
        dc2dt_vec[0:N] = [self.c2.dt(k) for k in range(N)]
        LHS1_vec = MX(Mmat, dc1dt_vec)
        LHS2_vec = MX(Mmat, dc2dt_vec)
        if self.get_trode_param("surface_diffusion") and self.get_trode_param("type") in ["ACR2"]:
            surf_diff_vec1 = calc_surf_diff(c1_surf, mu1R_surf,
                                           self.get_trode_param("D_surf"),
                                           self.get_trode_param("E_D_surf"),
                                           self.T_lyte())
            surf_diff_vec2 = calc_surf_diff(c2_surf, mu2R_surf,
                                           self.get_trode_param("D_surf"),
                                           self.get_trode_param("E_D_surf"),
                                           self.T_lyte())
            for k in range(N):
                eq1 = self.CreateEquation("dc1sdt_discr{k}".format(k=k))
                eq2 = self.CreateEquation("dc2sdt_discr{k}".format(k=k))
                eq1.Residual = LHS1_vec[k] - RHS1[k] - surf_diff_vec1[k]
                eq2.Residual = LHS2_vec[k] - RHS2[k] - surf_diff_vec2[k]
        else:
            for k in range(N):
                eq1 = self.CreateEquation("dc1sdt_discr{k}".format(k=k))
                eq2 = self.CreateEquation("dc2sdt_discr{k}".format(k=k))
                eq1.Residual = LHS1_vec[k] - RHS1[k]
                eq2.Residual = LHS2_vec[k] - RHS2[k]

        if self.get_trode_param("type") in ["ACR","ACR2"]:
            return eta1[-1], eta2[-1], c1_surf[-1], c2_surf[-1]
        else:
            return eta1, eta2, c1_surf, c2_surf


class Mod1var(dae.daeModel):
    def __init__(self, config, trode, vInd, pInd,
                 Name, Parent=None, Description=""):
        super().__init__(Name, Parent, Description)

        self.config = config
        self.trode = trode
        self.ind = (vInd, pInd)

        # Domain
        self.Dmn = dae.daeDomain("discretizationDomain", self, dae.unit(),
                                 "discretization domain")

        # Variables
        self.c = dae.daeVariable("c", mole_frac_t, self,
                                 "Concentration in active particle",
                                 [self.Dmn])
        self.cbar = dae.daeVariable(
            "cbar", mole_frac_t, self,
            "Average concentration in active particle")
        self.dcbardt = dae.daeVariable("dcbardt", dae.no_t, self, "Rate of particle filling")
        self.q_rxn_bar = dae.daeVariable(
            "q_rxn_bar", dae.no_t, self, "Rate of heat generation in particle")
        if config[trode, "type"] not in ["ACR"]:
            self.Rxn = dae.daeVariable("Rxn", dae.no_t, self, "Rate of reaction")
        else:
            self.Rxn = dae.daeVariable("Rxn", dae.no_t, self, "Rate of reaction", [self.Dmn])

        # Get reaction rate function
        rxnType = config[trode, "rxnType"]
        self.calc_rxn_rate = utils.import_function(config[trode, "rxnType_filename"],
                                                   rxnType,
                                                   f"mpet.electrode.reactions.{rxnType}")

        # Ports
        self.portInLyte = ports.portFromElyte(
            "portInLyte", dae.eInletPort, self,
            "Inlet port from electrolyte")
        self.portInBulk = ports.portFromBulk(
            "portInBulk", dae.eInletPort, self,
            "Inlet port from e- conducting phase")
        self.phi_lyte = self.portInLyte.phi_lyte
        self.T_lyte = self.portInLyte.T_lyte
        self.c_lyte = self.portInLyte.c_lyte
        self.phi_m = self.portInBulk.phi_m

    def get_trode_param(self, item):
        """
        Shorthand to retrieve electrode-specific value
        """
        value = self.config[self.trode, item]
        # check if it is a particle-specific parameter
        if item in self.config.params_per_particle:
            value = value[self.ind]
        return value

    def DeclareEquations(self):
        dae.daeModel.DeclareEquations(self)
        N = self.get_trode_param("N")  # number of grid points in particle
        r_vec, volfrac_vec = geo.get_unit_solid_discr(self.get_trode_param('shape'), N)

        # Prepare noise
        self.noise = None
        if self.get_trode_param("noise"):
            numnoise = self.get_trode_param("numnoise")
            noise_prefac = self.get_trode_param("noise_prefac")
            tvec = np.linspace(0., 1.05*self.config["tend"], numnoise)
            noise_data = noise_prefac*np.random.randn(numnoise, N)
            self.noise = sintrp.interp1d(tvec, noise_data, axis=0,
                                         bounds_error=False, fill_value=0.)

        # Figure out mu_O, mu of the oxidized state
        mu_O, act_lyte = calc_mu_O(self.c_lyte(), self.phi_lyte(), self.phi_m(), self.T_lyte(),
                                   self.config["elyteModelType"])

        # Define average filling fraction in particle
        eq = self.CreateEquation("cbar")
        eq.Residual = self.cbar()
        for k in range(N):
            eq.Residual -= self.c(k) * volfrac_vec[k]

        # Define average rate of filling of particle
        eq = self.CreateEquation("dcbardt")
        eq.Residual = self.dcbardt()
        for k in range(N):
            eq.Residual -= self.c.dt(k) * volfrac_vec[k]

        c = np.empty(N, dtype=object)
        c[:] = [self.c(k) for k in range(N)]
        if self.get_trode_param("type") in ["ACR", "diffn", "CHR"]:
            # Equations for 1D particles of 1 field varible
            eta, c_surf = self.sld_dynamics_1D1var(c, mu_O, act_lyte, self.noise)
        elif self.get_trode_param("type") in ["homog", "homog_sdn"]:
            # Equations for 0D particles of 1 field variables
            eta, c_surf = self.sld_dynamics_0D1var(c, mu_O, act_lyte, self.noise)

        # Define average rate of heat generation
        eq = self.CreateEquation("q_rxn_bar")
        if self.config["entropy_heat_gen"]:
            eq.Residual = self.q_rxn_bar() - self.dcbardt() * \
                (eta - self.T_lyte()*(np.log(c_surf/(1-c_surf))-1/self.c_lyte()))
        else:
            eq.Residual = self.q_rxn_bar() - self.dcbardt() * eta

        for eq in self.Equations:
            eq.CheckUnitsConsistency = False

    def sld_dynamics_0D1var(self, c, muO, act_lyte, noise):
        c_surf = c
        muR_surf, actR_surf = calc_muR(c_surf, self.cbar(), self.T_lyte(),self.config,
                                       self.trode, self.ind)
        eta = calc_eta(muR_surf, muO)
        eta_eff = eta + self.Rxn()*self.get_trode_param("Rfilm")
        if self.get_trode_param("noise"):
            eta_eff += noise[0]()
        Rxn = self.calc_rxn_rate(
            eta_eff, c_surf, self.c_lyte(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), actR_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        eq = self.CreateEquation("Rxn")
        eq.Residual = self.Rxn() - Rxn[0]

        eq = self.CreateEquation("dcsdt")
        eq.Residual = self.c.dt(0) - self.get_trode_param("delta_L")*self.Rxn()
        return eta[-1], c_surf[-1]

    def sld_dynamics_1D1var(self, c, muO, act_lyte, noise):
        N = self.get_trode_param("N")
        # Equations for concentration evolution
        # Mass matrix, M, where M*dcdt = RHS, where c and RHS are vectors
        Mmat = get_Mmat(self.get_trode_param('shape'), N)
        dr, edges = geo.get_dr_edges(self.get_trode_param('shape'), N)

        # Get solid particle chemical potential, overpotential, reaction rate
        if self.get_trode_param("type") in ["ACR"]:
            if noise is None:
                c_surf = c
            else:
                c_surf = c + noise(dae.Time().Value)
            muR_surf, actR_surf = calc_muR(
                c_surf, self.cbar(), self.T_lyte(), self.config, self.trode, self.ind)
        elif self.get_trode_param("type") in ["diffn", "CHR"]:
            muR, actR = calc_muR(c, self.cbar(), self.T_lyte(),
                                 self.config, self.trode, self.ind)
            c_surf = c[-1]
            muR_surf = muR[-1]
            if actR is None:
                actR_surf = None
            else:
                actR_surf = actR[-1]
        eta = calc_eta(muR_surf, muO)
        if self.get_trode_param("type") in ["ACR"]:
            eta_eff = np.array([eta[i] + self.Rxn(i)*self.get_trode_param("Rfilm")
                                for i in range(N)])
        else:
            eta_eff = eta + self.Rxn()*self.get_trode_param("Rfilm")
        Rxn = self.calc_rxn_rate(
            eta_eff, c_surf, self.c_lyte(), self.get_trode_param("k0"),
            self.get_trode_param("E_A"), self.T_lyte(), actR_surf, act_lyte,
            self.get_trode_param("lambda"), self.get_trode_param("alpha"))
        if self.get_trode_param("type") in ["ACR"]:
            for i in range(N):
                eq = self.CreateEquation("Rxn_{i}".format(i=i))
                eq.Residual = self.Rxn(i) - Rxn[i]
        else:
            eq = self.CreateEquation("Rxn")
            eq.Residual = self.Rxn() - Rxn

        # Get solid particle fluxes (if any) and RHS
        if self.get_trode_param("type") in ["ACR"]:
            RHS = np.array([self.get_trode_param("delta_L")*self.Rxn(i) for i in range(N)])
        elif self.get_trode_param("type") in ["diffn", "CHR"]:
            # Positive reaction (reduction, intercalation) is negative
            # flux of Li at the surface.
            Flux_bc = -self.Rxn()
            Dfunc_name = self.get_trode_param("Dfunc")
            Dfunc = utils.import_function(self.get_trode_param("Dfunc_filename"),
                                          Dfunc_name,
                                          f"mpet.electrode.diffusion.{Dfunc_name}")
            if self.get_trode_param("type") == "diffn":
                Flux_vec = calc_flux_diffn(c, self.get_trode_param("D"), Dfunc,
                                           self.get_trode_param("E_D"), Flux_bc, dr,
                                           self.T_lyte(), noise)
            elif self.get_trode_param("type") == "CHR":
                Flux_vec = calc_flux_CHR(c, muR, self.get_trode_param("D"), Dfunc,
                                         self.get_trode_param("E_D"), Flux_bc, dr,
                                         self.T_lyte(), noise)
            if self.get_trode_param("shape") == "sphere":
                area_vec = 4*np.pi*edges**2
            elif self.get_trode_param("shape") == "cylinder":
                area_vec = 2*np.pi*edges  # per unit height
            RHS = -np.diff(Flux_vec * area_vec)

        dcdt_vec = np.empty(N, dtype=object)
        dcdt_vec[0:N] = [self.c.dt(k) for k in range(N)]
        LHS_vec = MX(Mmat, dcdt_vec)
        if self.get_trode_param("surface_diffusion") and self.get_trode_param("type") in ["ACR"]:
            surf_diff_vec = calc_surf_diff(c_surf, muR_surf,
                                           self.get_trode_param("D_surf"),
                                           self.get_trode_param("E_D_surf"),
                                           self.T_lyte())
            for k in range(N):
                eq = self.CreateEquation("dcsdt_discr{k}".format(k=k))
                eq.Residual = LHS_vec[k] - RHS[k] - surf_diff_vec[k]
        else:
            for k in range(N):
                eq = self.CreateEquation("dcsdt_discr{k}".format(k=k))
                eq.Residual = LHS_vec[k] - RHS[k]

        if self.get_trode_param("type") in ["ACR"]:
            return eta[-1], c_surf[-1]
        else:
            return eta, c_surf


def calc_eta(muR, muO):
    return muR - muO


def calc_surf_diff(c_surf, c2_surf, muR1_surf, muR2_surf, D_surf, E_D_surf, T):
    # the surface diffusion keeps the volume centered method of the ACR
    N = np.size(c_surf)
    dxs = 1./N
    D_eff = (D_surf * np.exp(-E_D_surf/(T)
             + E_D_surf/1))
    c_edges = utils.mean_linear(c_surf)
    c2_edges = utils.mean_linear(c2_surf)
    surf_flux = np.empty(N+1, dtype=object)
    surf_flux[1:-1] = -D_eff*c_edges*(1-c_edges-c2_edges)*np.diff(muR1_surf)/(dxs)
    # surf_flux[1:-1] = -D_eff*c_edges*(1-c_edges-c2_edges)*np.diff(muR1_surf)/(dxs)
    # surf_flux[1:-1] += D_eff*c2_edges*np.diff(muR2_surf)/(dxs)
    surf_flux[0] = 0.
    surf_flux[-1] = 0.
    surf_diff = -np.diff(surf_flux)/dxs
    return surf_diff


def get_Mmat(shape, N):
    r_vec, volfrac_vec = geo.get_unit_solid_discr(shape, N)
    if shape == "C3":
        Mmat = sprs.eye(N, N, format="csr")
    elif shape in ["sphere", "cylinder"]:
        Rs = 1.
        # For discretization background, see Zeng & Bazant 2013
        # Mass matrix is common for each shape, diffn or CHR
        if shape == "sphere":
            Vp = 4./3. * np.pi * Rs**3
        elif shape == "cylinder":
            Vp = np.pi * Rs**2  # per unit height
        vol_vec = Vp * volfrac_vec
        M1 = sprs.diags([1./8, 3./4, 1./8], [-1, 0, 1],
                        shape=(N, N), format="csr")
        M1[1,0] = M1[-2,-1] = 1./4
        M2 = sprs.diags(vol_vec, 0, format="csr")
        Mmat = M1*M2
    return Mmat


def calc_flux_diffn(c, D, Dfunc, E_D, Flux_bc, dr, T, noise):
    N = len(c)
    Flux_vec = np.empty(N+1, dtype=object)
    Flux_vec[0] = 0  # Symmetry at r=0
    Flux_vec[-1] = Flux_bc
    c_edges = utils.mean_linear(c)
    if noise is None:
        Flux_vec[1:N] = -D/T * Dfunc(c_edges) * np.exp(-E_D/T + E_D/1) * np.diff(c)/dr
    else:
        Flux_vec[1:N] = -D/T * Dfunc(c_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(c + noise(dae.Time().Value))/dr
    return Flux_vec


def calc_flux_CHR(c, mu, D, Dfunc, E_D, Flux_bc, dr, T, noise):
    N = len(c)
    Flux_vec = np.empty(N+1, dtype=object)
    Flux_vec[0] = 0  # Symmetry at r=0
    Flux_vec[-1] = Flux_bc
    c_edges = utils.mean_linear(c)
    if noise is None:
        Flux_vec[1:N] = -D/T * Dfunc(c_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu)/dr
    else:
        Flux_vec[1:N] = -D/T * Dfunc(c_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu + noise(dae.Time().Value))/dr
    return Flux_vec


def calc_flux_CHR2(c1, c2, mu1_R, mu2_R, D, Dfunc, E_D, Flux1_bc, Flux2_bc, dr, T, noise1, noise2):
    N = len(c1)
    Flux1_vec = np.empty(N+1, dtype=object)
    Flux2_vec = np.empty(N+1, dtype=object)
    Flux1_vec[0] = 0.  # symmetry at r=0
    Flux2_vec[0] = 0.  # symmetry at r=0
    Flux1_vec[-1] = Flux1_bc
    Flux2_vec[-1] = Flux2_bc
    c1_edges = utils.mean_linear(c1)
    c2_edges = utils.mean_linear(c2)
    if noise1 is None:
        Flux1_vec[1:N] = -D/T * Dfunc(c1_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu1_R)/dr
        Flux2_vec[1:N] = -D/T * Dfunc(c2_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu2_R)/dr
    else:
        Flux1_vec[1:N] = -D/T * Dfunc(c1_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu1_R+noise1(dae.Time().Value))/dr
        Flux2_vec[1:N] = -D/T * Dfunc(c2_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu2_R+noise2(dae.Time().Value))/dr
    return Flux1_vec, Flux2_vec


def calc_flux_CHR2_multisub(c1, c2, mu1_R, mu2_R, D1, D2, Dfunc, E_D,
                            Flux1_bc, Flux2_bc, dr, T, noise1, noise2):
    N = len(c1)
    Flux1_vec = np.empty(N+1, dtype=object)
    Flux2_vec = np.empty(N+1, dtype=object)
    Flux1_vec[0] = 0.  # symmetry at r=0
    Flux2_vec[0] = 0.  # symmetry at r=0
    Flux1_vec[-1] = Flux1_bc
    Flux2_vec[-1] = Flux2_bc
    c1_edges = utils.mean_linear(c1)
    c2_edges = utils.mean_linear(c2)
    if noise1 is None:
        Flux1_vec[1:N] = -D1/T * Dfunc(c1_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu1_R)/dr
        Flux2_vec[1:N] = -D2/T * Dfunc(c2_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu2_R)/dr
    else:
        Flux1_vec[1:N] = -D1/T * Dfunc(c1_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu1_R+noise1(dae.Time().Value))/dr
        Flux2_vec[1:N] = -D2/T * Dfunc(c2_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu2_R+noise2(dae.Time().Value))/dr
    return Flux1_vec, Flux2_vec

def calc_flux_CHR2_multicat(c1, c2, mu1_R, mu2_R, D1, D2, Dfunc, E_D,
                            Flux1_bc, Flux2_bc, dr, T, noise1, noise2):
    N = len(c1)
    Flux1_vec = np.empty(N+1, dtype=object)
    Flux2_vec = np.empty(N+1, dtype=object)
    Flux1_vec[0] = 0.  # symmetry at r=0
    Flux2_vec[0] = 0.  # symmetry at r=0
    Flux1_vec[-1] = Flux1_bc
    Flux2_vec[-1] = Flux2_bc
    c1_edges = utils.mean_linear(c1)
    c2_edges = utils.mean_linear(c2)
    cavg_edges = (c1_edges + c2_edges)/2
    if noise1 is None:
        Flux1_vec[1:N] = -D1/T * c1_edges * (1-c1_edges-c2_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu1_R)/dr
        Flux2_vec[1:N] = -D2/T * c2_edges * (1-c2_edges-c1_edges) * np.exp(-E_D/T + E_D/1) * np.diff(mu2_R)/dr
    else:
        Flux1_vec[1:N] = -D1/T * c1_edges * (1-c1_edges-c2_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu1_R+noise1(dae.Time().Value))/dr
        Flux2_vec[1:N] = -D2/T * c2_edges * (1-c2_edges-c1_edges) * np.exp(-E_D/T + E_D/1) * \
            np.diff(mu2_R+noise2(dae.Time().Value))/dr
    return Flux1_vec, Flux2_vec

def calc_mu_O(c_lyte, phi_lyte, phi_sld, T, elyteModelType):
    if elyteModelType == "SM":
        act_lyte = c_lyte
        mu_lyte = phi_lyte
    elif elyteModelType == "dilute":
        act_lyte = c_lyte
        mu_lyte = T*np.log(act_lyte) + phi_lyte
    mu_O = mu_lyte - phi_sld
    return mu_O, act_lyte


def calc_muR(c, cbar, T, config, trode, ind):
    muRfunc = props_am.muRfuncs(config, trode, ind).muRfunc
    muR_ref = config[trode, "muR_ref"]
    muR, actR = muRfunc(c, cbar, T, muR_ref)
    return muR, actR


def MX(mat, objvec):
    if not isinstance(mat, sprs.csr.csr_matrix):
        raise Exception("MX function designed for csr mult")
    n = objvec.shape[0]
    if isinstance(objvec[0], dae.pyCore.adouble):
        out = np.empty(n, dtype=object)
    else:
        out = np.zeros(n, dtype=float)
    # Loop through the rows
    for i in range(n):
        low = mat.indptr[i]
        up = mat.indptr[i+1]
        if up > low:
            out[i] = np.sum(
                mat.data[low:up] * objvec[mat.indices[low:up]])
        else:
            out[i] = 0.0
    return out
