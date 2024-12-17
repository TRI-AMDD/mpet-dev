"""This module defines the actual simulation to be carried out.

In this, the model(s) are created and their variables are given initial conditions (if they are
differential variables) or initial guesses (if they never appear in equations in which they have
been differentiated in time).
"""
import sys
import os.path as osp

import daetools.pyDAE as dae
import numpy as np
import h5py

import mpet.mod_cell as mod_cell
import mpet.daeVariableTypes
import mpet.utils as utils
from mpet.config import constants


class SimMPET(dae.daeSimulation):
    def __init__(self, config, tScale=None):
        dae.daeSimulation.__init__(self)
        self.config = config
        self.tScale = tScale
        config["currPrev"] = 0.
        config["phiPrev"] = 0.
        if config["prevDir"] and config["prevDir"] != "false":
            # Get the data mat file from prevDir
            self.dataPrev = osp.join(config["prevDir"], "output_data")
            data = utils.open_data_file(self.dataPrev)
            config["currPrev"] = utils.get_dict_key(data, "current", final=True)
            config["phiPrev"] = utils.get_dict_key(data, "phi_applied", final=True)

            # close file if it is a h5py file
            if isinstance(data, h5py._hl.files.File):
                data.close()

        # Set absolute tolerances for variableTypes
        mpet.daeVariableTypes.mole_frac_t.AbsoluteTolerance = config["absTol"]
        mpet.daeVariableTypes.conc_t.AbsoluteTolerance = config["absTol"]
        mpet.daeVariableTypes.elec_pot_t.AbsoluteTolerance = config["absTol"]

        # Define the model we're going to simulate
        if config['MultiCation']:
            self.m = mod_cell.ModCell_multication(config, "mpet")
        else:
            self.m = mod_cell.ModCell(config, "mpet")

    def SetUpParametersAndDomains(self):
        # Domains
        config = self.config
        if config["have_separator"]:
            self.m.DmnCell["s"].CreateArray(config["Nvol"]["s"])
        for tr in config["trodes"]:
            self.m.DmnCell[tr].CreateArray(config["Nvol"][tr])
            self.m.DmnPart[tr].CreateArray(config["Npart"][tr])
            for i in range(config["Nvol"][tr]):
                for j in range(config["Npart"][tr]):
                    self.m.particles[tr][i, j].Dmn.CreateArray(
                        int(config["psd_num"][tr][i,j]))

    def SetUpVariables(self):
        config = self.config
        Nvol = config["Nvol"]
        Npart = config["Npart"]
        phi_cathode = config["phi_cathode"]
        if not config["prevDir"] or config["prevDir"] == "false":
            # Solids
            for tr in config["trodes"]:
                cs1 = config['cs1'][tr]
                cs2 = config['cs2'][tr]
                self.m.ffrac[tr].SetInitialGuess(cs1+cs2)
                # Guess initial filling fractions
                mosaic = config["mosaic_1"]
                if mosaic:
                    # list of int(cs1) particles with c1 = 0.9 among the matrix Nvol x Npart
                    part_list = [(i, j) for i in range(Nvol[tr]) for j in range(Npart[tr])]
                    np.random.shuffle(part_list)
                    for i, j in part_list[:int(cs1*len(part_list))]:
                        self.m.particles[tr][i, j].c1bar.SetInitialGuess(0.9)
                        self.m.particles[tr][i, j].c2bar.SetInitialGuess(cs2)
                        Nij = config["psd_num"][tr][i,j]
                        epsrnd = 0
                        rnd1 = epsrnd*(np.random.rand(Nij) - 0.5)
                        rnd2 = epsrnd*(np.random.rand(Nij) - 0.5)
                        rnd1 -= np.mean(rnd1)
                        rnd2 -= np.mean(rnd2)
                        for k in range(Nij):
                            self.m.particles[tr][i,j].c1.SetInitialCondition(k, 0.9+rnd1[k])
                            self.m.particles[tr][i, j].c2.SetInitialCondition(k, cs2)

                    for i, j in part_list[int(cs1*len(part_list)):]:
                        self.m.particles[tr][i, j].c1bar.SetInitialGuess(0.1)
                        self.m.particles[tr][i, j].c2bar.SetInitialGuess(cs2)
                        Nij = config["psd_num"][tr][i,j]
                        epsrnd = 0
                        rnd1 = epsrnd*(np.random.rand(Nij) - 0.5)
                        rnd2 = epsrnd*(np.random.rand(Nij) - 0.5)
                        rnd1 -= np.mean(rnd1)
                        rnd2 -= np.mean(rnd2)
                        for k in range(Nij):
                            self.m.particles[tr][i,j].c1.SetInitialCondition(k, 0.1+rnd1[k])
                            self.m.particles[tr][i, j].c2.SetInitialCondition(k, cs2)
                    
                for i in range(Nvol[tr]):
                    # Guess initial volumetric reaction rates
                    if config["MultiCation"]:
                        self.m.R_Vp_1[tr].SetInitialGuess(i, 0.0)
                        self.m.R_Vp_2[tr].SetInitialGuess(i, 0.0)
                    else:
                        self.m.R_Vp[tr].SetInitialGuess(i, 0.0)
                    # set initial temperature condition
                    self.m.T_lyte[tr].SetInitialCondition(i, config["T"])
                    # Guess initial value for the potential of the
                    # electrodes
                    if tr == "a":  # anode
                        self.m.phi_bulk[tr].SetInitialGuess(i, config["a", "phiRef"])
                    else:  # cathode
                        self.m.phi_bulk[tr].SetInitialGuess(i, phi_cathode)
                    for j in range(Npart[tr]):
                        Nij = config["psd_num"][tr][i,j]
                        part = self.m.particles[tr][i,j]
                        # Guess initial value for the average solid
                        # concentrations and set initial value for
                        # solid concentrations
                        solidType = self.config[tr, "type"]
                        if mosaic == False:
                            part.c1bar.SetInitialGuess(cs1)
                            part.c2bar.SetInitialGuess(cs2)
                            epsrnd = 0
                            rnd1 = epsrnd*(np.random.rand(Nij) - 0.5)
                            rnd2 = epsrnd*(np.random.rand(Nij) - 0.5)
                            rnd1 -= np.mean(rnd1)
                            rnd2 -= np.mean(rnd2)
                            for k in range(Nij):
                                if self.config[tr,'type'] not in ['ACR2']:
                                    if self.config[tr,'intralayer_rxn']:
                                        part.interLayerRxn.SetInitialGuess(k, 0.0)
                                if config['cs1'][tr] is not None:
                                    part.c1.SetInitialCondition(k, cs1+rnd1[k])
                                    part.c2.SetInitialCondition(k, cs2+rnd2[k])

            # Cell potential initialization
            if config['tramp'] > 0:
                phi_guess = 0
            elif config['profileType'] == 'CV':
                phi_guess = config['Vset']
            elif config['profileType'] == 'CVsegments':
                phi_guess = config['segments'][0][0]
            else:
                phi_guess = 0
            self.m.phi_applied.SetInitialGuess(phi_guess)
            self.m.phi_cell.SetInitialGuess(phi_guess)

            # Initialize the ghost points used for boundary conditions
            if not self.m.SVsim:
                if self.config["MultiCation"]:
                    self.m.c_lyte_1_GP_L.SetInitialGuess(config["c01"])
                    self.m.c_lyte_2_GP_L.SetInitialGuess(config["c02"])
                    self.m.phi_lyteGP_L.SetInitialGuess(0)
                    self.m.T_lyteGP_L.SetInitialGuess(config["T"])
                    self.m.T_lyteGP_R.SetInitialGuess(config["T"])
                else:    
                    self.m.c_lyteGP_L.SetInitialGuess(config["c0"])
                    self.m.phi_lyteGP_L.SetInitialGuess(0)
                    self.m.T_lyteGP_L.SetInitialGuess(config["T"])
                    self.m.T_lyteGP_R.SetInitialGuess(config["T"])

            # Separator electrolyte initialization
            if config["have_separator"]:
                if self.config["MultiCation"]:
                    for i in range(Nvol["s"]):
                        self.m.c_lyte["s"].SetInitialCondition(i, config['c01'])
                        self.m.c_lyte["s"].SetInitialCondition(i, config['c02'])
                        self.m.T_lyte["s"].SetInitialCondition(i, config['T'])
                        self.m.phi_lyte["s"].SetInitialGuess(i, 0)
                else:    
                    for i in range(Nvol["s"]):
                        self.m.c_lyte["s"].SetInitialCondition(i, config['c0'])
                        self.m.T_lyte["s"].SetInitialCondition(i, config['T'])
                        self.m.phi_lyte["s"].SetInitialGuess(i, 0)

            # Anode and cathode electrolyte initialization
            for tr in config["trodes"]:
                if self.config["MultiCation"]:
                    for i in range(Nvol[tr]):
                        self.m.c_lyte_1[tr].SetInitialCondition(i, config['c01'])
                        self.m.c_lyte_2[tr].SetInitialCondition(i, config['c02'])
                        self.m.T_lyte[tr].SetInitialCondition(i, config['T'])
                        self.m.phi_lyte[tr].SetInitialGuess(i, 0)

                        for j in range(Npart[tr]):
                            self.m.particles[tr][i,j].c_lyte_1.SetInitialGuess(config["c01"])
                            self.m.particles[tr][i,j].c_lyte_2.SetInitialGuess(config["c01"])
                            self.m.particles[tr][i,j].T_lyte.SetInitialGuess(config["T"])
                else:
                    for i in range(Nvol[tr]):
                        self.m.c_lyte[tr].SetInitialCondition(i, config['c0'])
                        self.m.T_lyte[tr].SetInitialCondition(i, config['T'])
                        self.m.phi_lyte[tr].SetInitialGuess(i, 0)

                        # Set electrolyte concentration in each particle
                        for j in range(Npart[tr]):
                            self.m.particles[tr][i,j].c_lyte.SetInitialGuess(config["c0"])
                            self.m.particles[tr][i,j].T_lyte.SetInitialGuess(config["T"])

        else:
            dPrev = self.dataPrev
            data = utils.open_data_file(dPrev)
            for tr in config["trodes"]:
                self.m.ffrac[tr].SetInitialGuess(
                    utils.get_dict_key(data, "ffrac_" + tr, final=True))
                for i in range(Nvol[tr]):
                    self.m.R_Vp[tr].SetInitialGuess(
                        i, data["R_Vp_" + tr][-1,i])
                    self.m.phi_bulk[tr].SetInitialGuess(
                        i, data["phi_bulk_" + tr][-1,i])
                    for j in range(Npart[tr]):
                        Nij = config["psd_num"][tr][i,j]
                        part = self.m.particles[tr][i,j]
                        solidType = self.config[tr, "type"]
                        partStr = "partTrode{l}vol{i}part{j}_".format(
                            l=tr, i=i, j=j)

                        # Set the inlet port variables for each particle
                        part.c_lyte.SetInitialGuess(data["c_lyte_" + tr][-1,i])
                        part.T_lyte.SetInitialGuess(data["T_lyte_" + tr][-1,i])
                        part.phi_lyte.SetInitialGuess(data["phi_lyte_" + tr][-1,i])
                        part.phi_m.SetInitialGuess(data["phi_bulk_" + tr][-1,i])

                        if solidType in constants.one_var_types:
                            part.cbar.SetInitialGuess(
                                utils.get_dict_key(data, partStr + "cbar", final=True))
                            for k in range(Nij):
                                part.c.SetInitialCondition(
                                    k, data[partStr + "c"][-1,k])
                        elif solidType in constants.two_var_types:
                            part.c1bar.SetInitialGuess(
                                utils.get_dict_key(data, partStr + "c1bar", final=True))
                            part.c2bar.SetInitialGuess(
                                utils.get_dict_key(data, partStr + "c2bar", final=True))
                            part.cbar.SetInitialGuess(
                                utils.get_dict_key(data, partStr + "cbar", final=True))
                            for k in range(Nij):
                                part.c1.SetInitialCondition(
                                    k, data[partStr + "c1"][-1,k])
                                part.c2.SetInitialCondition(
                                    k, data[partStr + "c2"][-1,k])
            if config["have_separator"]:
                for i in range(Nvol["s"]):
                    self.m.c_lyte["s"].SetInitialCondition(
                        i, data["c_lyte_s"][-1,i])
                    self.m.T_lyte["s"].SetInitialCondition(
                        i, data["T_lyte_s"][-1,i])
                    self.m.phi_lyte["s"].SetInitialGuess(
                        i, data["phi_lyte_s"][-1,i])
            for tr in config["trodes"]:
                for i in range(Nvol[tr]):
                    self.m.c_lyte[tr].SetInitialCondition(
                        i, data["c_lyte_" + tr][-1,i])
                    self.m.T_lyte[tr].SetInitialCondition(
                        i, data["T_lyte_" + tr][-1,i])
                    self.m.phi_lyte[tr].SetInitialGuess(
                        i, data["phi_lyte_" + tr][-1,i])

            # Read in the ghost point values
            if not self.m.SVsim:
                if self.config["MultiCation"]:
                    self.m.c_lyte_1_GP_L.SetInitialGuess(
                        utils.get_dict_key(data, "c_lyte_1_GP_L", final=True))
                    self.m.c_lyte_2_GP_L.SetInitialGuess(
                        utils.get_dict_key(data, "c_lyte_2_GP_L", final=True))
                    self.m.phi_lyteGP_L.SetInitialGuess(
                        utils.get_dict_key(data, "phi_lyteGP_L", final=True))
                else:
                    self.m.c_lyteGP_L.SetInitialGuess(
                        utils.get_dict_key(data, "c_lyteGP_L", final=True))
                    self.m.phi_lyteGP_L.SetInitialGuess(
                        utils.get_dict_key(data, "phi_lyteGP_L", final=True))

            # Guess the initial cell voltage
            self.m.phi_applied.SetInitialGuess(utils.get_dict_key(data, "phi_applied", final=True))
            self.m.phi_cell.SetInitialGuess(utils.get_dict_key(data, "phi_cell", final=True))

            # close file if it is a h5py file
            if isinstance(data, h5py._hl.files.File):
                data.close()

        # The simulation runs when the endCondition is 0
        self.m.endCondition.AssignValue(0)

    def Run(self):
        """
        Overload the simulation "Run" function so that the simulation
        terminates when the specified condition is satisfied.
        """
        tScale = self.tScale
        for nextTime in self.ReportingTimes:

            # Print logging information
            progressStr = "{0} {1}".format(self.Log.PercentageDone,self.Log.ETA)
            message = "Integrating from {t0:.2f} to {t1:.2f} s ...".format(
                t0=self.CurrentTime*tScale, t1=nextTime*tScale)
            sys.stdout.write(f"\r{progressStr[:-1]} {message}")
            sys.stdout.flush()

            # Integrate the equations
            self.IntegrateUntilTime(nextTime, dae.eStopAtModelDiscontinuity, True)
            self.ReportData(self.CurrentTime)
            self.Log.SetProgress(int(100. * self.CurrentTime/self.TimeHorizon))

            # Break when an end condition has been met
            if self.m.endCondition.npyValues:
                description = mod_cell.endConditions[int(self.m.endCondition.npyValues)]
                sys.stdout.write("\nEnding condition: " + description)
                break
