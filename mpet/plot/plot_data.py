import os

import matplotlib as mpl
import matplotlib.animation as manim
import matplotlib.collections as mcollect
import matplotlib.pyplot as plt
import numpy as np

import mpet.geometry as geom
import mpet.mod_cell as mod_cell
import mpet.utils as utils
from mpet.config import Config, constants

"""Set list of matplotlib rc parameters to make more readable plots."""
# axtickfsize = 18
# labelfsize = 20
# mpl.rcParams['xtick.labelsize'] = axtickfsize
# mpl.rcParams['ytick.labelsize'] = axtickfsize
# mpl.rcParams['axes.labelsize'] = labelfsize
# mpl.rcParams['font.size'] = labelfsize - 2
# mpl.rcParams['legend.fontsize'] = labelfsize - 2
# mpl.rcParams['lines.linewidth'] = 3
# mpl.rcParams['lines.markersize'] = 10
# mpl.rcParams['lines.markeredgewidth'] = 0.1
# mpl.rcParams['text.usetex'] = True


def show_data(indir, plot_type, print_flag, save_flag, data_only, vOut=None, pOut=None, tOut=None):
    pfx = 'mpet.'
    sStr = "_"
    ttl_fmt = "% = {perc:2.1f}"
    # Read in the simulation results and calcuations data
    dataFileName = "output_data"
    dataFile = os.path.join(indir, dataFileName)
    data = utils.open_data_file(dataFile)
    try:
        utils.get_dict_key(data, pfx + 'current')
    except KeyError:
        pfx = ''
    try:
        utils.get_dict_key(data, pfx + "partTrodecvol0part0" + sStr + "cbar")
    except KeyError:
        sStr = "."
    # Read in the parameters used to define the simulation
    config = Config.from_dicts(indir)
    # simulated (porous) electrodes
    trodes = config["trodes"]
    # Pick out some useful calculated values
    limtrode = config["limtrode"]
    k = constants.k                      # Boltzmann constant, J/(K Li)
    Tref = constants.T_ref               # Temp, K
    e = constants.e                      # Charge of proton, C
    F = constants.F                      # C/mol
    c_ref = constants.c_ref
    td = config["t_ref"]
    Etheta = {"a": 0.}
    cap = config[limtrode, "cap"]
    for trode in trodes:
        Etheta[trode] = -(k*Tref/e) * config[trode, "phiRef"]
    Vstd = Etheta["c"] - Etheta["a"]
    dataReporter = config["dataReporter"]
    Nvol = config["Nvol"]
    Npart = config["Npart"]
    psd_len = config["psd_len"]
    # Discretization (and associated porosity)
    Lfac = 1e6
    Lunit = r"$\mu$m"
    dxc = config["L"]["c"]/Nvol["c"]
    dxvec = np.array(Nvol["c"] * [dxc])
    porosvec = np.array(Nvol["c"] * [config["poros"]["c"]])
    cellsvec = dxc*np.arange(Nvol["c"]) + dxc/2.
    if config["have_separator"]:
        dxs = config["L"]["s"]/Nvol["s"]
        dxvec_s = np.array(Nvol["s"] * [dxs])
        dxvec = np.hstack((dxvec_s, dxvec))
        poros_s = np.array(Nvol["s"] * [config["poros"]["s"]])
        porosvec = np.hstack((poros_s, porosvec))
        cellsvec += config["L"]["s"] / config["L"]["c"]
        cellsvec_s = dxs*np.arange(Nvol["s"]) + dxs/2.
        cellsvec = np.hstack((cellsvec_s, cellsvec))
    if "a" in trodes:
        dxa = config["L"]["a"]/Nvol["a"]
        dxvec_a = np.array(Nvol["a"] * [dxa])
        dxvec = np.hstack((dxvec_a, dxvec))
        poros_a = np.array(Nvol["a"] * [config["poros"]["a"]])
        porosvec = np.hstack((poros_a, porosvec))
        cellsvec += config["L"]["a"] / config["L"]["c"]
        cellsvec_a = dxa*np.arange(Nvol["a"]) + dxa/2.
        cellsvec = np.hstack((cellsvec_a, cellsvec))
    cellsvec *= config["L_ref"] * Lfac
    facesvec = np.insert(np.cumsum(dxvec), 0, 0.) * config["L_ref"] * Lfac
    # Extract the reported simulation times
    times = utils.get_dict_key(data, pfx + 'phi_applied_times')
    numtimes = len(times)
    tmin = np.min(times)
    tmax = np.max(times)
    # Simulation type
    profileType = config['profileType']
    # Colors for plotting concentrations
    to_yellow = 0.3
    to_red = 0.7
    scl = 1.0  # static
#    scl = 1.2  # movies
    figsize = (scl*6, scl*4)

    # Print relevant simulation info
    if print_flag:
        print("profileType:", profileType)
#        for i in trodes:
#            print "PSD_{l}:".format(l=l)
#            print psd_len[l].transpose()
#            print "Actual psd_mean [nm]:", np.mean(psd_len[l])
#            print "Actual psd_stddev [nm]:", np.std(psd_len[l])
        print("Cell structure:")
        print(("porous anode | " if "a" in config["trodes"] else "flat anode | ")
              + ("sep | " if config["have_separator"] else "") + "porous cathode")
        if "a" in config["trodes"]:
            print("capacity ratio cathode:anode, 'z':", config["z"])
        for trode in trodes:
            print("solidType_{t}:".format(t=trode), config[trode, 'type'])
            print("solidShape_{t}".format(t=trode), config[trode, 'shape'])
            print("rxnType_{t}:".format(t=trode), config[trode, 'rxnType'])
        if profileType == "CC":
            print("C_rate:", config['Crate'])
            theoretical_1C_current = config[config['limtrode'], 'cap'] / 3600.
            currset_dim = config['currset'] * theoretical_1C_current * config['curr_ref']
            print("current:", currset_dim, "A/m^2")
        else:  # CV
            Vref = config['c', 'phiRef']
            if 'a' in config["trodes"]:
                Vref -= config['a', 'phiRef']
            Vset_dim = -(config['Vset'] * k * Tref / e - Vref)
            print("Vset:", Vset_dim)
        print("Specified psd_mean, c [{unit}]:".format(unit=Lunit),
              np.array(config['mean']["c"])*Lfac)
        print("Specified psd_stddev, c [{unit}]:".format(unit=Lunit),
              np.array(config['stddev']["c"])*Lfac)
        print("ndim B_c:", config["c", "B"])
        if config["have_separator"]:
            print("Nvol_s:", Nvol["s"])
        print("Nvol_c:", Nvol["c"])
        if 'a' in config["trodes"]:
            print("Nvol_a:", Nvol["a"])
        print("Npart_c:", Npart["c"])
        if 'a' in config["trodes"]:
            print("Npart_a:", Npart["a"])
        print("td [s]:", config["t_ref"])
        for trode in trodes:
            if trode == "a":
                k0 = config.D_a["k0"]
            elif trode == "c":
                k0 = config.D_c["k0"]
            else:
                raise Exception(f"Unknown trode: {trode}")
            print("k0_{t} [A/m^2]:".format(t=trode), k0)
            rxnType = config[trode, 'rxnType']
            if rxnType == "BV":
                print("alpha_" + trode + ":", config[trode, 'alpha'])
            elif rxnType in ["Marcus", "MHC"]:
                print("lambda_" + trode + "/(kTref):", config[trode, "lambda"]
                      * k * Tref)
            if config['simBulkCond'][trode]:
                print(trode + " bulk conductivity loss: Yes -- "
                      + "sigma_s [S/m]: " + str(config['sigma_s'][trode] * config['sigma_s_ref']))
            else:
                print(trode + " bulk conductivity loss: No")
            try:
                simSurfCond = config[trode, 'simSurfCond']
                if simSurfCond:
                    print(trode + " surface conductivity loss: Yes -- "
                          + "dim_scond [S]: " + str(config[trode, 'scond']))
                else:
                    print(trode + " surface conductivity loss: No")
            except Exception:
                pass
#            if ndD['simSurfCond'][l]:
#                print (l + " surface conductivity loss: Yes -- " +
#                        "dim_scond [S]: " + str(dD['scond'][l]))
#            else:
#                print l + " surface conductivity loss: No"

    if plot_type in ["params"]:
        # return ndD_s, dD_s, ndD_e, dD_e
        return config
    if plot_type in ["discData"]:
        return cellsvec/Lfac, facesvec/Lfac

    # Plot voltage profile
    if plot_type in ["v", "vt"]:
        voltage = (Vstd
                   - (k*Tref/e)*utils.get_dict_key(data, pfx + 'phi_applied'))
        ffvec = utils.get_dict_key(data, pfx + 'ffrac_c')
        fig, ax = plt.subplots(figsize=figsize)
        if plot_type == "v":
            if data_only:
                plt.close(fig)
                return ffvec, voltage
            ax.plot(ffvec, voltage)
            xmin = 0.
            xmax = 1.
            ax.set_xlim((xmin, xmax))
            ax.set_xlabel("Cathode Filling Fraction [dimensionless]")
        elif plot_type == "vt":
            if data_only:
                plt.close(fig)
                return times*td, voltage
            ax.plot(times*td, voltage)
            ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
        if save_flag:
            fig.savefig("mpet_v.pdf", bbox_inches="tight")
        return fig, ax

    # Plot surface conc.
    if plot_type[:-2] in ["surf"]:
        if dataReporter == "hdf5Fast":
            # hdf5Fast does not print internal particle concentrations
            raise Exception("hdf5Fast dataReporter does not print internal particle "
                            "concentrations, rerun simulation with another data reporter")
        trode = plot_type[-1]
        str_base = (pfx
                    + "partTrode{trode}vol{{vInd}}part{{pInd}}".format(trode=trode)
                    + sStr + "c")
        if data_only:
            sol_str = str_base.format(pInd=pOut, vInd=vOut)
            datay = utils.get_dict_key(data, sol_str, squeeze=False)[:,-1]
            return times*td, datay
        fig, ax = plt.subplots(Npart[trode], Nvol[trode], squeeze=False, sharey=True,
                               figsize=figsize)
        ylim = (0, 1.01)
        datax = times
        for pInd in range(Npart[trode]):
            for vInd in range(Nvol[trode]):
                sol_str = str_base.format(pInd=pInd, vInd=vInd)
                # Remove axis ticks
                ax[pInd,vInd].xaxis.set_major_locator(plt.NullLocator())
                datay = utils.get_dict_key(data, sol_str, squeeze=False)[:,-1]
                line, = ax[pInd,vInd].plot(times, datay)
        return fig, ax

    # Plot SoC profile
    if plot_type[:-2] in ["soc"]:
        trode = plot_type[-1]
        ffvec = utils.get_dict_key(data, pfx + 'ffrac_{trode}'.format(trode=trode))
        if data_only:
            return times*td, ffvec
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times*td, ffvec)
        xmin = np.min(ffvec)
        xmax = np.max(ffvec)
        ax.set_ylim((0, 1.05))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Filling Fraciton [dimless]")
        if save_flag:
            fig.savefig("mpet_soc.pdf", bbox_inches="tight")
        return fig, ax

    # Check to make sure mass is conserved in elyte
    if plot_type == "elytecons":
        fig, ax = plt.subplots(figsize=figsize)
        eps = 1e-2
        ymin = 1-eps
        ymax = 1+eps
#        ax.set_ylim((ymin, ymax))
        ax.set_ylabel('Avg. Concentration of electrolyte [nondim]')
        sep = pfx + 'c_lyte_s'
        anode = pfx + 'c_lyte_a'
        cath = pfx + 'c_lyte_c'
        ax.set_xlabel('Time [s]')
        cvec = utils.get_dict_key(data, cath)
        if Nvol["s"]:
            cvec_s = utils.get_dict_key(data, sep)
            cvec = np.hstack((cvec_s, cvec))
        if "a" in trodes:
            cvec_a = utils.get_dict_key(data, anode)
            cvec = np.hstack((cvec_a, cvec))
        cavg = np.sum(porosvec*dxvec*cvec, axis=1)/np.sum(porosvec*dxvec)
        if data_only:
            plt.close(fig)
            return times*td, cavg
        np.set_printoptions(precision=8)
        ax.plot(times*td, cavg)
        return fig, ax

    # Plot current profile
    if plot_type == "curr":
        theoretical_1C_current = config[config['limtrode'], "cap"] / 3600.  # A/m^2
        current = (utils.get_dict_key(data, pfx + 'current')
                   * theoretical_1C_current / config['1C_current_density'] * config['curr_ref'])
        ffvec = utils.get_dict_key(data, pfx + 'ffrac_c')
        if data_only:
            return times*td, current
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times*td, current)
        xmin = np.min(ffvec)
        xmax = np.max(ffvec)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Current [C-rate]")
        if save_flag:
            fig.savefig("mpet_current.png", bbox_inches="tight")
        return fig, ax

    # Plot maximum temperature profile
    if plot_type == "max_temp":
        T_sep, T_anode, T_cath = pfx + 'T_lyte_s', pfx + 'T_lyte_a', pfx + 'T_lyte_c'
        datay_T = utils.get_dict_key(data, T_cath, squeeze=False)
        if config["have_separator"]:
            datay_s_T = utils.get_dict_key(data, T_sep, squeeze=False)
            datay_T = np.hstack((datay_s_T, datay_T))
        if "a" in trodes:
            datay_a_T = utils.get_dict_key(data, T_anode, squeeze=False)
            datay_T = np.hstack((datay_a_T, datay_T))
        Tmax = np.max(datay_T * Tref, axis=1)
        if data_only:
            return times*td, Tmax
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times*td, Tmax)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Maximum Temperature [K]")
        if save_flag:
            fig.savefig("mpet_max_temp.png", bbox_inches="tight")
        return fig, ax

    if plot_type == "power":
        current = utils.get_dict_key(data, pfx + 'current') * (3600/td) * (cap/3600)  # in A/m^2
        voltage = (Vstd - (k*Tref/e)*utils.get_dict_key(data, pfx + 'phi_applied'))  # in V
        power = np.multiply(current, voltage)
        if data_only:
            return times*td, power
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times*td, power)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(r"Power [W/m$^2$]")
        if save_flag:
            fig.savefig("mpet_power.png", bbox_inches="tight")
        return fig, ax

    # Plot electrolyte concentration or potential
    elif plot_type in ["elytec", "elytep", "elytecf", "elytepf", "elytei",
                       "elyteif", "elytedivi", "elytedivif", "temp"]:
        fplot = (True if plot_type[-1] == "f" else False)
        t0ind = (0 if not fplot else -1)
        datax = cellsvec
        c_sep, p_sep, T_sep = pfx + 'c_lyte_s', pfx + 'phi_lyte_s', pfx + 'T_lyte_s'
        c_anode, p_anode, T_anode = pfx + 'c_lyte_a', pfx + 'phi_lyte_a', pfx + 'T_lyte_a'
        c_cath, p_cath, T_cath = pfx + 'c_lyte_c', pfx + 'phi_lyte_c', pfx + 'T_lyte_c'
        datay_c = utils.get_dict_key(data, c_cath, squeeze=False)
        datay_p = utils.get_dict_key(data, p_cath, squeeze=False)
        datay_T = utils.get_dict_key(data, T_cath, squeeze=False)
        L_c = config['L']["c"] * config['L_ref'] * Lfac
        Ltot = L_c
        if config["have_separator"]:
            datay_s_c = utils.get_dict_key(data, c_sep, squeeze=False)
            datay_s_p = utils.get_dict_key(data, p_sep, squeeze=False)
            datay_s_T = utils.get_dict_key(data, T_sep, squeeze=False)
            datay_c = np.hstack((datay_s_c, datay_c))
            datay_p = np.hstack((datay_s_p, datay_p))
            datay_T = np.hstack((datay_s_T, datay_T))
            L_s = config['L']["s"] * config['L_ref'] * Lfac
            Ltot += L_s
        else:
            L_s = 0
        if "a" in trodes:
            datay_a_c = utils.get_dict_key(data, c_anode, squeeze=False)
            datay_a_p = utils.get_dict_key(data, p_anode, squeeze=False)
            datay_a_T = utils.get_dict_key(data, T_anode, squeeze=False)
            datay_c = np.hstack((datay_a_c, datay_c))
            datay_p = np.hstack((datay_a_p, datay_p))
            datay_T = np.hstack((datay_a_T, datay_T))
            L_a = config['L']["a"] * config['L_ref'] * Lfac
            Ltot += L_a
        else:
            L_a = 0
        xmin = 0
        xmax = Ltot
        if plot_type in ["elytec", "elytecf"]:
            ylbl = 'Concentration of electrolyte [M]'
            datay = datay_c * c_ref / 1000.
        elif plot_type in ["elytep", "elytepf"]:
            ylbl = 'Potential of electrolyte [V]'
            datay = datay_p*(k*Tref/e) - Vstd
        elif plot_type in ["temp"]:
            ylbl = 'Temperature [K]'
            datay = datay_T * constants.T_ref
        elif plot_type in ["elytei", "elyteif", "elytedivi", "elytedivif"]:
            cGP_L = utils.get_dict_key(data, "c_lyteGP_L")
            pGP_L = utils.get_dict_key(data, "phi_lyteGP_L")
            TGP_L = utils.get_dict_key(data, "T_lyteGP_L")
            cmat = np.hstack((cGP_L.reshape((-1,1)), datay_c, datay_c[:,-1].reshape((-1,1))))
            pmat = np.hstack((pGP_L.reshape((-1,1)), datay_p, datay_p[:,-1].reshape((-1,1))))
            Tmat = np.hstack((TGP_L.reshape((-1,1)), datay_T, datay_T[:,-1].reshape((-1,1))))
            disc = geom.get_elyte_disc(
                Nvol, config["L"], config["poros"], config["BruggExp"], config["k_h"])
            i_edges = np.zeros((numtimes, len(facesvec)))
            for tInd in range(numtimes):
                # no heat flux at boundary
                i_edges[tInd, :] = mod_cell.get_lyte_internal_fluxes(
                    cmat[tInd, :], pmat[tInd, :], Tmat[tInd, :], disc, config, Nvol)[1]
            if plot_type in ["elytei", "elyteif"]:
                ylbl = r'Current density of electrolyte [A/m$^2$]'
                datax = facesvec
                datay = i_edges * (F*constants.c_ref*config["D_ref"]/config["L_ref"])
            elif plot_type in ["elytedivi", "elytedivif"]:
                ylbl = r'Divergence of electrolyte current density [A/m$^3$]'
                datax = cellsvec
                datay = np.diff(i_edges, axis=1) / disc["dxvec"]
                datay *= (F*constants.c_ref*config["D_ref"]/config["L_ref"]**2)
        if fplot:
            datay = datay[t0ind]
        if data_only:
            return datax, datay, L_a, L_s
        dataMin, dataMax = np.min(datay), np.max(datay)
        dataRange = dataMax - dataMin
        ymin = dataMin - 0.05*dataRange
        ymax = dataMax + 0.05*dataRange
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Battery Position [{unit}]'.format(unit=Lunit))
        ax.set_ylabel(ylbl)
        ttl = ax.text(
            0.5, 1.05, ttl_fmt.format(perc=0),
            transform=ax.transAxes, verticalalignment="center",
            horizontalalignment="center")
        ax.set_ylim((ymin, ymax))
        ax.set_xlim((xmin, xmax))
        # returns tuble of line objects, thus comma
        if fplot:
            line1, = ax.plot(datax, datay, '-')
        else:
            line1, = ax.plot(datax, datay[t0ind,:], '-')
        ax.axvline(x=L_a, linestyle='--', color='g')
        ax.axvline(x=(L_a+L_s), linestyle='--', color='g')
        if fplot:
            print("time =", times[t0ind]*td, "s")
            if save_flag:
                fig.savefig("mpet_{pt}.png".format(pt=plot_type),
                            bbox_inches="tight")
            return fig, ax

        def init():
            line1.set_ydata(np.ma.array(datax, mask=True))
            ttl.set_text('')
            return line1, ttl

        def animate(tind):
            line1.set_ydata(datay[tind])
            t_current = times[tind]
            tfrac = (t_current - tmin)/(tmax - tmin) * 100
            ttl.set_text(ttl_fmt.format(perc=tfrac))
            return line1, ttl

    # Plot solid particle-average concentrations
    elif plot_type[:-2] in ["cbarLine", "dcbardtLine"]:
        trode = plot_type[-1]
        fig, ax = plt.subplots(Npart[trode], Nvol[trode], squeeze=False, sharey=True,
                               figsize=figsize)
        partStr = "partTrode{trode}vol{{vInd}}part{{pInd}}".format(trode=trode) + sStr
        type2c = False
        if config[trode, "type"] in constants.one_var_types:
            if plot_type[:-2] in ["cbarLine"]:
                str_base = pfx + partStr + "cbar"
            elif plot_type[:-2] in ["dcbardtLine"]:
                str_base = pfx + partStr + "dcbardt"
        elif config[trode, "type"] in constants.two_var_types:
            type2c = True
            if plot_type[:-2] in ["cbarLine"]:
                str1_base = pfx + partStr + "c1bar"
                str2_base = pfx + partStr + "c2bar"
            elif plot_type[:-2] in ["dcbardtLine"]:
                str1_base = pfx + partStr + "dc1bardt"
                str2_base = pfx + partStr + "dc2bardt"
        ylim = (0, 1.01)
        datax = times*td
        if data_only:
            plt.close(fig)
            if type2c:
                sol1_str = str1_base.format(pInd=pOut, vInd=vOut)
                sol2_str = str2_base.format(pInd=pOut, vInd=vOut)
                datay1 = utils.get_dict_key(data, sol1_str)
                datay2 = utils.get_dict_key(data, sol2_str)
                datay = (datay1, datay2)
            else:
                sol_str = str_base.format(pInd=pOut, vInd=vOut)
                datay = utils.get_dict_key(data, sol_str)
            return datax, datay
        xLblNCutoff = 4
        xLbl = "Time [s]"
        yLbl = "Particle Average Filling Fraction"
        for pInd in range(Npart[trode]):
            for vInd in range(Nvol[trode]):
                if type2c:
                    sol1_str = str1_base.format(pInd=pInd, vInd=vInd)
                    sol2_str = str2_base.format(pInd=pInd, vInd=vInd)
                    if Nvol[trode] > xLblNCutoff:
                        # Remove axis ticks
                        ax[pInd,vInd].xaxis.set_major_locator(plt.NullLocator())
                    else:
                        ax[pInd,vInd].set_xlabel(xLbl)
                        ax[pInd,vInd].set_ylabel(yLbl)
                    datay1 = utils.get_dict_key(data, sol1_str)
                    datay2 = utils.get_dict_key(data, sol2_str)
                    line1, = ax[pInd,vInd].plot(times, datay1)
                    line2, = ax[pInd,vInd].plot(times, datay2)
                else:
                    sol_str = str_base.format(pInd=pInd, vInd=vInd)
                    if Nvol[trode] > xLblNCutoff:
                        # Remove axis ticks
                        ax[pInd,vInd].xaxis.set_major_locator(plt.NullLocator())
                    else:
                        ax[pInd,vInd].set_xlabel(xLbl)
                        ax[pInd,vInd].set_ylabel(yLbl)
                    datay = utils.get_dict_key(data, sol_str)
                    line, = ax[pInd,vInd].plot(times, datay)
        return fig, ax

    # Plot all solid concentrations or potentials
    elif plot_type[:-2] in ["csld"]:
        if dataReporter == "hdf5Fast":
            # hdf5Fast does not print internal particle concentrations
            raise Exception("hdf5Fast dataReporter does not print internal particle "
                            "concentrations, rerun simulation with another data reporter")

        timettl = False  # Plot the current simulation time as title
        # Plot title in seconds
        ttlscl, ttlunit = 1, "s"
        t0ind = 0
        trode = plot_type[-1]
        if plot_type[0] == "c":
            plt_cavg = True
        else:
            plt_cavg = False
        plt_axlabels = True
        if config[trode, "type"] in constants.one_var_types:
            type2c = False
        elif config[trode, "type"] in constants.two_var_types:
            type2c = True
        Nv, Np = Nvol[trode], Npart[trode]
        partStr = "partTrode{trode}vol{vInd}part{pInd}" + sStr
        fig, ax = plt.subplots(Np, Nv, squeeze=False, sharey=True, figsize=figsize)
        if not type2c:
            cstr_base = pfx + partStr + "c"
            cbarstr_base = pfx + partStr + "cbar"
            cstr = np.empty((Np, Nv), dtype=object)
            cbarstr = np.empty((Np, Nv), dtype=object)
            lines = np.empty((Np, Nv), dtype=object)
        elif type2c:
            c1str_base = pfx + partStr + "c1"
            c2str_base = pfx + partStr + "c2"
            c1barstr_base = pfx + partStr + "c1bar"
            c2barstr_base = pfx + partStr + "c2bar"
            c1str = np.empty((Np, Nv), dtype=object)
            c2str = np.empty((Np, Nv), dtype=object)
            c1barstr = np.empty((Np, Nv), dtype=object)
            c2barstr = np.empty((Np, Nv), dtype=object)
            lines1 = np.empty((Np, Nv), dtype=object)
            lines2 = np.empty((Np, Nv), dtype=object)
            lines3 = np.empty((Np, Nv), dtype=object)
        lens = np.zeros((Np, Nv))
        if data_only:
            print("tInd_{}".format(tOut), "time =", times[tOut]*td, "s")
            lenval = psd_len[trode][vOut, pOut]
            if type2c:
                c1str = c1str_base.format(trode=trode, pInd=pOut, vInd=vOut)
                c2str = c2str_base.format(trode=trode, pInd=pOut, vInd=vOut)
                c1barstr = c1barstr_base.format(trode=trode, pInd=pOut, vInd=vOut)
                c2barstr = c2barstr_base.format(trode=trode, pInd=pOut, vInd=vOut)
                datay1 = utils.get_dict_key(data, c1str[pOut,vOut])
                datay2 = utils.get_dict_key(data, c2str[pOut,vOut])
                datay = (datay1, datay2)
                numy = len(datay1)
            else:
                cstr = cstr_base.format(trode=trode, pInd=pOut, vInd=vOut)
                cbarstr = cbarstr_base.format(trode=trode, pInd=pOut, vInd=vOut)
                datay = utils.get_dict_key(data, cstr)[tOut]
                numy = len(datay)
            datax = np.linspace(0, lenval * Lfac, numy)
            plt.close(fig)
            return datax, datay
        ylim = (0, 1.01)
        for pInd in range(Np):
            for vInd in range(Nv):
                lens[pInd,vInd] = psd_len[trode][vInd,pInd]
                if type2c:
                    c1str[pInd,vInd] = c1str_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    c2str[pInd,vInd] = c2str_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    c1barstr[pInd,vInd] = c1barstr_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    c2barstr[pInd,vInd] = c2barstr_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    datay1 = utils.get_dict_key(data, c1str[pInd,vInd])[t0ind]
                    datay2 = utils.get_dict_key(data, c2str[pInd,vInd])[t0ind]
                    datay3 = (datay1 + datay2)
                    lbl1, lbl2 = r"$\widetilde{c}_1$", r"$\widetilde{c}_2$"
                    lbl3 = r"$\overline{c}$"
                    numy = len(datay1) if isinstance(datay1, np.ndarray) else 1
                    datax = np.linspace(0, lens[pInd,vInd] * Lfac, numy)
                    line1, = ax[pInd,vInd].plot(datax, datay1, label=lbl1, linewidth=3)
                    line2, = ax[pInd,vInd].plot(datax, datay2, label=lbl2, linewidth=3)
                    if plt_cavg:
                        line3, = ax[pInd,vInd].plot(datax, datay3, '--', label=lbl3)
                        lines3[pInd,vInd] = line3
                    lines1[pInd,vInd] = line1
                    lines2[pInd,vInd] = line2
                else:
                    cstr[pInd,vInd] = cstr_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    cbarstr[pInd,vInd] = cbarstr_base.format(trode=trode, pInd=pInd, vInd=vInd)
                    datay = utils.get_dict_key(data, cstr[pInd,vInd])[t0ind]
                    numy = len(datay)
                    datax = np.linspace(0, lens[pInd,vInd] * Lfac, numy)
                    line, = ax[pInd,vInd].plot(datax, datay)
                    lines[pInd,vInd] = line
                ax[pInd,vInd].set_ylim(ylim)
                ax[pInd,vInd].set_xlim((0, lens[pInd,vInd] * Lfac))
                if plt_axlabels:
                    ax[pInd, vInd].set_xlabel(r"$x$ [{Lunit}]".format(Lunit=Lunit))
                    if plot_type[0] == "c":
                        ax[pInd, vInd].set_ylabel(r"$\widetilde{{c}}$")
                    elif plot_type[:2] == "mu":
                        ax[pInd, vInd].set_ylabel(r"$\mu/k_\mathrm{B}T$")
                if timettl:
                    ttl = ax[pInd, vInd].text(
                        0.5, 1.04, "t = {tval:3.3f} {ttlu}".format(
                            tval=times[t0ind]*td*ttlscl, ttlu=ttlunit),
                        verticalalignment="center", horizontalalignment="center",
                        transform=ax[pInd, vInd].transAxes)

        def init():
            toblit = []
            for pInd in range(Npart[trode]):
                for vInd in range(Nvol[trode]):
                    if type2c:
                        data_c1str = utils.get_dict_key(data, c1str[pInd,vInd])[t0ind]
                        # check if it is array, then return length. otherwise return 1
                        numy = len(data_c1str) if isinstance(data_c1str, np.ndarray) else 1
                        maskTmp = np.zeros(numy)
                        lines1[pInd,vInd].set_ydata(np.ma.array(maskTmp, mask=True))
                        lines2[pInd,vInd].set_ydata(np.ma.array(maskTmp, mask=True))
                        lines_local = np.vstack((lines1, lines2))
                        if plt_cavg:
                            lines3[pInd,vInd].set_ydata(np.ma.array(maskTmp, mask=True))
                            lines_local = np.vstack((lines_local, lines3))
                    else:
                        data_cstr = utils.get_dict_key(data, cstr[pInd,vInd])[t0ind]
                        numy = len(data_cstr) if isinstance(data_cstr, np.ndarray) else 1
                        maskTmp = np.zeros(numy)
                        lines[pInd,vInd].set_ydata(np.ma.array(maskTmp, mask=True))
                        lines_local = lines.copy()
                    toblit.extend(lines_local.reshape(-1))
                    if timettl:
                        ttl.set_text("")
                        toblit.extend([ttl])
            return tuple(toblit)

        def animate(tind):
            toblit = []
            for pInd in range(Npart[trode]):
                for vInd in range(Nvol[trode]):
                    if type2c:
                        datay1 = utils.get_dict_key(data, c1str[pInd,vInd])[tind]
                        datay2 = utils.get_dict_key(data, c2str[pInd,vInd])[tind]
                        datay3 = (datay1 + datay2)
                        lines1[pInd,vInd].set_ydata(datay1)
                        lines2[pInd,vInd].set_ydata(datay2)
                        lines_local = np.vstack((lines1, lines2))
                        if plt_cavg:
                            lines3[pInd,vInd].set_ydata(datay3)
                            lines_local = np.vstack((lines_local, lines3))
                    else:
                        datay = utils.get_dict_key(data, cstr[pInd,vInd])[tind]
                        lines[pInd,vInd].set_ydata(datay)
                        lines_local = lines.copy()
                    toblit.extend(lines_local.reshape(-1))
                    if timettl:
                        ttl.set_text("t = {tval:3.3f} {ttlu}".format(
                            tval=times[tind]*td*ttlscl, ttlu=ttlunit))
                        toblit.extend([ttl])
            return tuple(toblit)

    # Plot average solid concentrations
    elif plot_type in ["cbar_c", "cbar_a", "cbar_full"]:
        if plot_type[-4:] == "full":
            trvec = ["a", "c"]
        elif plot_type[-1] == "a":
            trvec = ["a"]
        else:
            trvec = ["c"]
        dataCbar = {}
        for trode in trodes:
            dataCbar[trode] = np.zeros((numtimes, Nvol[trode], Npart[trode]))
            for tInd in range(numtimes):
                for vInd in range(Nvol[trode]):
                    for pInd in range(Npart[trode]):
                        dataStr = (
                            pfx
                            + "partTrode{t}vol{vInd}part{pInd}".format(
                                t=trode, vInd=vInd, pInd=pInd)
                            + sStr + "cbar")
                        dataCbar[trode][tInd,vInd,pInd] = (
                            np.squeeze(utils.get_dict_key(data, dataStr))[tInd])
        if data_only:
            return dataCbar
        # Set up colors.
        # Define if you want smooth or discrete color changes
        # Option: "smooth" or "discrete"
        # color_changes = "discrete"
        color_changes = "smooth"
        # Discrete color changes:
        if color_changes == "discrete":
            # Make a discrete colormap that goes from green to yellow
            # to red instantaneously
            cdict = {
                "red": [(0.0, 0.0, 0.0),
                        (to_yellow, 0.0, 1.0),
                        (1.0, 1.0, 1.0)],
                "green": [(0.0, 0.502, 0.502),
                          (to_yellow, 0.502, 1.0),
                          (to_red, 1.0, 0.0),
                          (1.0, 0.0, 0.0)],
                "blue": [(0.0, 0.0, 0.0),
                         (1.0, 0.0, 0.0)]
                }
            cmap = mpl.colors.LinearSegmentedColormap(
                "discrete", cdict)
        # Smooth colormap changes:
        if color_changes == "smooth":
            # generated with colormap.org
            cmaps = np.load(r"C:\Users\pierfrancescoo\Documents\Phase-field\mpet-LFMP\mpet\mpet\plot\colormaps_custom.npz")
            cmap_data = cmaps["GnYlRd_3"]
            cmap = mpl.colors.ListedColormap(cmap_data/255.)

        size_frac_min = 0.10
        fig, axs = plt.subplots(1, len(trvec), squeeze=False, figsize=figsize)
        ttlx = 0.5 if len(trvec) < 2 else 1.1
        ttl = axs[0,0].text(
            ttlx, 1.05, ttl_fmt.format(perc=0),
            transform=axs[0,0].transAxes, verticalalignment="center",
            horizontalalignment="center")
        collection = np.empty(len(trvec), dtype=object)
        for indx, trode in enumerate(trvec):
            ax = axs[0,indx]
            # Get particle sizes (and max size) (length-based)
            lens = psd_len[trode]
            len_max = np.max(lens)
            len_min = np.min(lens)
            ax.patch.set_facecolor('white')
            # Don't stretch axes to fit figure -- keep 1:1 x:y ratio.
            ax.set_aspect('equal', 'box')
            # Don't show axis ticks
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.set_xlim(0, 1.)
            ax.set_ylim(0, float(Npart[trode])/Nvol[trode])
            # Label parts of the figure
#            ylft = ax.text(-0.07, 0.5, "Separator",
#                    transform=ax.transAxes, rotation=90,
#                    verticalalignment="center",
#                    horizontalalignment="center")
#            yrht = ax.text(1.09, 0.5, "Current Collector",
#                    transform=ax.transAxes, rotation=90,
#                    verticalalignment="center",
#                    horizontalalignment="center")
#            xbtm = ax.text(.50, -0.05, "Electrode Depth -->",
#                    transform=ax.transAxes, rotation=0,
#                    verticalalignment="center",
#                    horizontalalignment="center")
            # Geometric parameters for placing the rectangles on the axes
            spacing = 1.0 / Nvol[trode]
            size_fracs = 0.4*np.ones((Nvol[trode], Npart[trode]))
            if len_max != len_min:
                size_fracs = (lens - len_min)/(len_max - len_min)
            sizes = (size_fracs*(1-size_frac_min) + size_frac_min) / Nvol[trode]
            # Create rectangle "patches" to add to figure axes.
            rects = np.empty((Nvol[trode], Npart[trode]), dtype=object)
            color = 'green'  # value is irrelevant -- it will be animated
            for (vInd, pInd), c in np.ndenumerate(sizes):
                size = sizes[vInd,pInd]
                center = np.array([spacing*(vInd + 0.5), spacing*(pInd + 0.5)])
                bottom_left = center - size / 2
                rects[vInd,pInd] = plt.Rectangle(
                    bottom_left, size, size, color=color)
            # Create a group of rectange "patches" from the rects array
            collection[indx] = mcollect.PatchCollection(rects.reshape(-1))
            # Put them on the axes
            ax.add_collection(collection[indx])
        # Have a "background" image of rectanges representing the
        # initial state of the system.

        def init():
            for indx, trode in enumerate(trvec):
                cbar_mat = dataCbar[trode][0,:,:]
                colors = cmap(cbar_mat.reshape(-1))
                collection[indx].set_color(colors)
                ttl.set_text('')
            out = [collection[i] for i in range(len(collection))]
            out.append(ttl)
            out = tuple(out)
            return out

        def animate(tind):
            for indx, trode in enumerate(trvec):
                cbar_mat = dataCbar[trode][tind,:,:]
                colors = cmap(cbar_mat.reshape(-1))
                collection[indx].set_color(colors)
            t_current = times[tind]
            tfrac = (t_current - tmin)/(tmax - tmin) * 100
            ttl.set_text(ttl_fmt.format(perc=tfrac))
            out = [collection[i] for i in range(len(collection))]
            out.append(ttl)
            out = tuple(out)
            return out

    # Plot cathode potential
    elif plot_type[0:5] in ["bulkp"]:
        trode = plot_type[-1]
        fplot = (True if plot_type[-3] == "f" else False)
        t0ind = (0 if not fplot else -1)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel('Position in electrode [{unit}]'.format(unit=Lunit))
        ax.set_ylabel('Potential of cathode [nondim]')
        ttl = ax.text(0.5, 1.05, ttl_fmt.format(perc=0),
                      transform=ax.transAxes, verticalalignment="center",
                      horizontalalignment="center")
        bulkp = pfx + 'phi_bulk_{trode}'.format(trode=trode)
        datay = utils.get_dict_key(data, bulkp)
        ymin = np.min(datay) - 0.2
        ymax = np.max(datay) + 0.2
        if trode == "a":
            datax = cellsvec[:Nvol["a"]]
        elif trode == "c":
            datax = cellsvec[-Nvol["c"]:]
        if data_only:
            plt.close(fig)
            return datax, datay
        # returns tuble of line objects, thus comma
        line1, = ax.plot(datax, datay[t0ind])

        def init():
            line1.set_ydata(np.ma.array(datax, mask=True))
            ttl.set_text('')
            return line1, ttl

        def animate(tind):
            line1.set_ydata(datay[tind])
            t_current = times[tind]
            tfrac = (t_current - tmin)/(tmax - tmin) * 100
            ttl.set_text(ttl_fmt.format(perc=tfrac))
            return line1, ttl

    else:
        raise Exception("Unexpected plot type argument. See README.md.")

    ani = manim.FuncAnimation(
        fig, animate, frames=numtimes, interval=50, blit=True, repeat=False, init_func=init)
    if save_flag:
        fig.tight_layout()
        ani.save("mpet_{type}.mp4".format(type=plot_type), fps=25, bitrate=5500)

    return fig, ax, ani
