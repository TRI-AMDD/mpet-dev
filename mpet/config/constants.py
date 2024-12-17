# global constants

#: Reference temperature, K
T_ref = 298.
#: Boltzmann constant, J/(K particle)
k = 1.381e-23
#: Electron charge, C
e = 1.602e-19
#: Avogadro constant, particle / mol
N_A = 6.022e23
#: Reference flux, C/mol
F = e * N_A
#: General particle classification (1 var)
two_var_types = ["diffn2", "ACR2", "CHR2", "homog2", "homog2_sdn"]
#: General particle classification (2 var)
one_var_types = ["ACR", "diffn", "CHR", "homog", "homog_sdn"]
#: Reference concentration, mol/m^3 = 1M
c_ref = 1000.
#: Reaction rate epsilon for values close to zero
reactions_epsilon = 1e-12

#: parameter that are defined per electrode with a ``_{electrode}`` suffix
PARAMS_PER_TRODE = ['Nvol', 'Npart', 'mean', 'stddev','cs1', 'cs2', 'simBulkCond', 'sigma_s',
                    'simPartCond','carbon_coating' ,'G_carb', 'G_bulk', 'G_mean_cont', 'G_std_cont',
                    'G_bulk_1','G_bulk_2', 
                    'simPartNet', 'avg_num_cont', 'std_num_cont', 
                    'penalty_grid_cont', 'penalty_factor', 'perc_grid',
                    'L', 'P_L', 'poros', 'BruggExp',
                    'specified_psd', 'rhom', 'cp', 'k_h', 'c_dep_exp']
#: subset of ``PARAMS_PER_TRODE``` that is defined for the separator as well
PARAMS_SEPARATOR = ['Nvol', 'L', 'poros', 'BruggExp', 'k_h', 'cp', 'rhom']
#: parameters that are defined for each particle, and their type
PARAMS_PARTICLE = {'N': int, 'kappa': float, 
                   'kappa1': float, 'kappa2': float, 'kappa12': float,
                   'beta_s': float, 'beta_s1': float, 'beta_s2': float,
                   'D': float, 'D1': float, 'D2': float, 'D_surf': float,
                   'k0': float, 'k0_1': float, 'k0_2': float,
                   'Rfilm': float, 'delta_L': float, 'Omega_a': float,
                   'E_D_surf': float, 'E_D': float,'E_A': float}
