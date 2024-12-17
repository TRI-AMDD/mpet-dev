# definition of config file sections, parameters, types

import ast
from distutils.util import strtobool

from schema import Schema, Use, Optional, And, Or
import numpy as np

from mpet.config import constants


def parse_segments(key):
    """
    Parse the segments key of the configuration file and
    validate it

    :param str key: The raw key from the config file
    :return: segments (tuple)
    """
    segments = ast.literal_eval(key)
    assert isinstance(segments, list), "segments must be a list"
    assert len(segments) > 0, "There must be at least one segment"
    for item in segments:
        assert (isinstance(item, tuple)) and (len(item) == 2), \
            "Each segment must be a tuple of (setpoint, time)"
    return segments


def check_allowed_values(value, allowed_values):
    """
    Check if value was chosen from a set of allowed values

    :param value: Value to verify
    :param list allowed_values: Possible values
    :return: True if value appears in allowed_values, else and AssertionError is raised
    """
    assert value in allowed_values, f"{value} is invalid, options are: {allowed_values}"
    # Schema needs a True return value if the check passes
    return True


def tobool(value):
    """
    Convert string value (y/yes/t/true/1/on, n/no/f/false/0/off) to boolean

    :param str value: Value to convert to bool
    :return: Boolean representation of value
    """
    assert isinstance(value, str), f"{value} must be a string"
    # strtobool returns 0 or 1, use bool() to convert to actual boolean type
    return bool(strtobool(value))


#: System parameters, per section
system = {'Sim Params': {'profileType': lambda x:
                         check_allowed_values(x, ["CC", "CV", "CP", "CCsegments", "CVsegments"]),
                         'Crate': Use(float),
                         Optional('power', default=None): Use(float),
                         Optional('1C_current_density', default=None): Use(float),
                         Optional('tramp', default=0.): Use(float),
                         'Vmax': Use(float),
                         'Vmin': Use(float),
                         Optional('Vset', default=None): Use(float),
                         Optional('capFrac', default=1.0): Use(float),
                         Optional('segments', default=[]): Use(parse_segments),
                         Optional('prevDir', default=''): str,
                         'tend': And(Use(float), lambda x: x > 0),
                         'tsteps': And(Use(int), lambda x: x > 0),
                         'relTol': And(Use(float), lambda x: x > 0),
                         'absTol': And(Use(float), lambda x: x > 0),
                         'T': Use(float),
                         Optional('nonisothermal', default=False): Use(tobool),
                         'randomSeed': Use(tobool),
                         Optional('seed'): And(Use(int), lambda x: x >= 0),
                         Optional('dataReporter', default='mat'): str,
                         'Rser': Use(float),
                         'Nvol_c': And(Use(int), lambda x: x > 0),
                         'Nvol_s': And(Use(int), lambda x: x >= 0),
                         'Nvol_a': And(Use(int), lambda x: x >= 0),
                         'Npart_c': And(Use(int), lambda x: x >= 0),
                         'Npart_a': And(Use(int), lambda x: x >= 0)},
          'Electrodes': {'cathode': str,
                         'anode': str,
                         'k0_foil': Use(float),
                         'Rfilm_foil': Use(float)},
          'Particles': {'mean_c': Use(float),
                        'stddev_c': Use(float),
                        Optional('mean_a', default=0): Use(float),
                        Optional('stddev_a', default=0): Use(float),
                        Optional('cs1_c', default=0): Use(float),
                        Optional('cs2_c', default=0): Use(float),
                        Optional('cs0_a', default=0): Use(float),
                        Optional('cs1_a', default=0): Use(float),
                        Optional('cs2_a', default=0): Use(float),
                        Optional('mosaic_1', default=False): Use(tobool),
                        Optional('mosaic_2', default=False): Use(tobool),
                        Optional('specified_psd_c', default=False):
                            Or(Use(tobool), Use(lambda x: np.array(ast.literal_eval(x)))),
                        Optional('specified_psd_a', default=False):
                            Or(Use(tobool), Use(lambda x: np.array(ast.literal_eval(x))))},
          'Conductivity': {'simBulkCond_c': Use(tobool),
                           'simBulkCond_a': Use(tobool),
                           'sigma_s_c': Use(float),
                           'sigma_s_a': Use(float),
                           Optional('simPartCond_c', default=False): Use(tobool),
                           Optional('simPartCond_a', default=False): Use(tobool),
                           Optional('carbon_coating_c', default=False): Use(tobool),
                           Optional('carbon_coating_a', default=False): Use(tobool),
                           Optional('simPartNet_c', default=False): Use(tobool),
                           Optional('simPartNet_a', default=False): Use(tobool),
                           Optional('avg_num_cont_c', default=5): Use(float),
                           Optional('avg_num_cont_a', default=5): Use(float),
                           Optional('std_num_cont_c', default=5): Use(float),
                           Optional('std_num_cont_a', default=5): Use(float),
                           Optional('penalty_grid_cont_c', default=False): Use(tobool),
                           Optional('penalty_grid_cont_a', default=False): Use(tobool),
                           Optional('penalty_factor_c', default=1): Use(float),
                           Optional('penalty_factor_a', default=1): Use(float),
                           Optional('perc_grid_c', default=1): Use(float),
                           Optional('perc_grid_a', default=1): Use(float),
                           Optional('G_bulk_c', default=0): Use(float),
                           Optional('G_bulk_a', default=0): Use(float),
                           Optional('G_carb_c', default=0): Use(float),
                           Optional('c_dep_exp_c', default=1): Use(float),
                           Optional('c_dep_exp_a', default=1): Use(float),
                           Optional('G_carb_a', default=0): Use(float),
                           Optional('G_mean_cont_c', default=0): Use(float),
                           Optional('G_mean_cont_a', default=0): Use(float),
                           Optional('G_std_cont_c', default=0): Use(float),
                           Optional('G_std_cont_a', default=0): Use(float),
                           Optional('G_bulk_1_c', default=0): Use(float),
                           Optional('G_bulk_1_a', default=0): Use(float),
                           Optional('G_bulk_2_c', default=0): Use(float),
                           Optional('G_bulk_2_a', default=0): Use(float),
                           },
          'Geometry': {'L_c': Use(float),
                       'L_a': Use(float),
                       'L_s': Use(float),
                       'P_L_c': Use(float),
                       'P_L_a': Use(float),
                       'poros_c': Use(float),
                       'poros_a': Use(float),
                       'poros_s': Use(float),
                       'BruggExp_c': Use(float),
                       'BruggExp_a': Use(float),
                       'BruggExp_s': Use(float)},
          'Thermal Parameters': {Optional('cp_c', default=0): Use(float),
                                 Optional('cp_a', default=0): Use(float),
                                 Optional('cp_s', default=0): Use(float),
                                 Optional('rhom_c', default=0): Use(float),
                                 Optional('rhom_a', default=0): Use(float),
                                 Optional('rhom_s', default=0): Use(float),
                                 Optional('k_h_c', default=0): Use(float),
                                 Optional('k_h_a', default=0): Use(float),
                                 Optional('k_h_s', default=0): Use(float),
                                 Optional('h_h', default=0): Use(float),
                                 Optional('entropy_heat_gen', default=False): Use(tobool)},
          'Electrolyte': {Optional('c0', default=1000): Use(float),
                          'zp': Use(int),
                          'zm': And(Use(int), lambda x: x < 0),
                          'nup': Use(int),
                          'num': Use(int),
                          'elyteModelType': str,
                          Optional('SMset_filename', default=None): str,
                          'SMset': str,
                          'n': Use(int),
                          'sp': Use(int),
                          Optional('Dp', default=None): Use(float),
                          Optional('Dm', default=None): Use(float),
                          Optional('MultiCation', default=False): Use(tobool),
                          Optional('c01', default=1000): Use(float),
                          Optional('c02', default=1000): Use(float),}}

#: Electrode parameters, per section
electrode = {'Particles': {'type': lambda x: check_allowed_values(x,
                                                                  constants.one_var_types
                                                                  + constants.two_var_types),
                           'discretization': Use(float),
                           'shape': lambda x:
                               check_allowed_values(x, ["C3", "sphere", "cylinder", "homog_sdn"]),
                           Optional('thickness', default=30e-9): Use(float),
                           Optional('std_thickness', default=0e-9): Use(float)},
             'Material': {Optional('muRfunc_filename', default=None): str,
                          'muRfunc': str,
                          'noise': Use(tobool),
                          'noise_prefac': Use(float),
                          'numnoise': Use(int),
                          Optional('v_2', default=None): Use(float),
                          Optional('Omega_a', default=None): Use(float),
                          Optional('Omega_b', default=None): Use(float),
                          Optional('Omega_c', default=None): Use(float),
                          Optional('kappa', default=None): Use(float),
                          Optional('kappa1', default=None): Use(float),
                          Optional('kappa2', default=None): Use(float),
                          Optional('kappa12', default=None): Use(float),
                          Optional('B', default=None): Use(float),
                          Optional('B1', default=None): Use(float),
                          Optional('B2', default=None): Use(float),
                          Optional('EvdW', default=None): Use(float),
                          Optional('rho_s', default=None): Use(float),
                          Optional('rho_s_1', default=None): Use(float),
                          Optional('rho_s_2', default=None): Use(float),
                          Optional('D', default=None): Use(float),
                          Optional('D1', default=None): Use(float),
                          Optional('D2', default=None): Use(float),
                          Optional('D_surf', default=0.): Use(float),
                          Optional('Dfunc_filename', default=None): str,
                          'Dfunc': str,
                          Optional('E_D', default=0.): Use(float),
                          Optional('E_D_surf', default=0.): Use(float),
                          Optional('dgammadc', default=None): Use(float),
                          Optional('dgammadc_1', default=None): Use(float),
                          Optional('dgammadc_2', default=None): Use(float),
                          Optional('natural_bc', default=False): Use(tobool),
                          Optional('natural_bc_c', default=False): Use(tobool),
                          Optional('cwet', default=None): Use(float),
                          Optional('cwet_1', default=None): Use(float),
                          Optional('cwet_2', default=None): Use(float)},
             'Reactions': {Optional('rxnType_filename', default=None): str,
                           Optional('surface_diffusion', default=None): Use(tobool),
                           Optional('intralayer_rxn', default=False): Use(tobool),
                           Optional('rxnType', default=None): str,
                           Optional('rxnType_1', default=None): str,
                           Optional('rxnType_2', default=None): str,
                           Optional('k0', default=None): Use(float),
                           Optional('k0_1', default=None): Use(float),
                           Optional('k012_ratio', default=None): Use(float),
                           Optional('k0_2', default=None): Use(float),
                           Optional('E_A', default=0.): Use(float),
                           'alpha': Use(float),
                           Optional('lambda', default=None): Use(float),
                           Optional('lambda_1', default=None): Use(float),
                           Optional('lambda_2', default=None): Use(float),
                           Optional('lambda_m', default=None): Use(float),
                           Optional('kintra', default=None): Use(float),
                           'Rfilm': Use(float)}}


# convert the dictionaries to actual schemas
for d in [system, electrode]:
    for key, value in d.items():
        d[key] = Schema(value, ignore_extra_keys=False)
