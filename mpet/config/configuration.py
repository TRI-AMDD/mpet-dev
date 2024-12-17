"""
This module provides functions for various data format exchanges:
 - config files (on disk) <--> dictionaries of parameters (in memory)
 - dictionaries of parameters (in memory) <--> dictionaries of parameters ('pickled' on disk)

It also has various other functions used in processes for things such as generating
distributions from input means and standard deviations.
"""
import os
import pickle

import numpy as np

from mpet.config import constants
from mpet.config.derived_values import DerivedValues
from mpet.config.parameterset import ParameterSet


class Config:
    def __init__(self, paramfile='params.cfg', from_dicts=False):
        """
        Hold values from system and electrode configuration files, as well as
        derived values. When initializing a new Config object, the invididual
        particle distributions are generated and any non-dimensionalization is
        performed *in-place*.

        The actual parameters are stored in up to four separate dictionary-like objects:

        * ``D_s`` holds the system config
        * ``D_c`` holds the cathode config
        * ``D_a`` holds the anode config (only if anode is simulated)
        * ``derived_values`` holds the values that can be derived from (a combination of) the other
          configs

        All parameter values can be accessed directly from the Config object
        with the ``[]`` operator, like one would access values from a dictionary.
        For parameters that are defined for every individual particle, the values for
        all particles are returned, i.e. an array with shape ``(Nvol, Npart)``.
        See :meth:`__getitem__`
        for example usage of the ``[]`` operator.

        Note that if needed, the underlying dictionaries can be accessed directly,
        e.g. ``config.D_s``.

        :param str paramfile: Path to .cfg file on disk, or folder with dictionaries
            on disk if from_dicts=True
        :param bool from_dicts: Whether to read existing config dicts from disk.
            Instead of using from_dicts=True, consider using the Config.from_dicts function instead

        :return: Config object

        Examples:

        To read the config from a set of .cfg files:

        >>> from mpet.config import Config
        >>> config = Config('configs/params_system.cfg')

        To create config from a previous run of MPET
        (See also :meth:`from_dicts`):

        >>> from mpet.config import Config
        >>> config = Config.from_dicts('/path/to/previous/run/sim_output')

        .. make sure to document methods related to [] operator
        .. automethod:: __getitem__
        .. automethod:: __setitem__
        .. automethod:: __delitem__
        """
        # initialize class to calculate and hold derived values
        self.derived_values = DerivedValues()
        # Parameters defined per particle in an (Nvol, Npart) array.
        # initially this list is empty. When the individual particle values
        # are calculated, the list is populated.
        self.params_per_particle = []

        if from_dicts:
            # read existing dictionaries instead of parameter file
            # paramfile is now folder with input dicts
            self.path = os.path.normpath(paramfile)
            self._init_from_dicts()
        else:
            # store path to config file
            self.path = os.path.dirname(paramfile)
            self._init_from_cfg(paramfile)

    @classmethod
    def from_dicts(cls, path):
        """
        Create a config instance from a set of dictionaries on disk, instead
        of from config files.

        :param str path: folder containing previously-saved config dicts

        :return: Config object

        Example usage:

        >>> from mpet.config import Config
        >>> config = Config.from_dicts('/path/to/previous/run/sim_output')
        """
        return cls(path, from_dicts=True)

    def _init_from_dicts(self):
        """
        Initialize configuration from a set of dictionaries on disk, generated
        from a previous run. This method should only be called from the
        ``__init__`` of :class:`Config`.
        """
        # create empty system parameter set
        self.D_s = ParameterSet(None, 'system', self.path)
        # set which electrodes there are based on which dict files exist
        trodes = ['c']
        # set up types of materials (cathode, anode electrolyte) for thermal parameters
        materials = ['c', 'l']
        if os.path.isfile(os.path.join(self.path, 'input_dict_anode.p')):
            trodes.append('a')
            materials.append('a')
        self['trodes'] = trodes
        self['materials'] = materials

        # create empty electrode parametersets
        self.D_c = ParameterSet(None, 'electrode', self.path)
        if 'a' in self['trodes']:
            self.D_a = ParameterSet(None, 'electrode', self.path)
        else:
            self.D_a = None
        # now populate the dicts
        self.read(self.path, full=True)
        self.params_per_particle = list(constants.PARAMS_PARTICLE.keys())
        self.config_processed = True

    def _init_from_cfg(self, paramfile):
        """
        Initialize configuration from a set of .cfg files on disk.
        This method should only be called from the
        ``__init__`` of :class:`Config`.

        :param str paramfile: Path to system parameter .cfg file
        """
        # load system parameters file
        self.D_s = ParameterSet(paramfile, 'system', self.path)
        # the anode and separator are optional: only if there are volumes to simulate
        trodes = ['c']
        # set up types of materials (cathode, anode electrolyte) for thermal parameters
        materials = ['c', 'l']
        if self.D_s['Nvol_a'] > 0:
            trodes.append('a')
            materials.append('a')
        self['trodes'] = trodes
        self['materials'] = materials
        # to check for separator, directly access underlying dict of system config;
        # self['Nvol']['s'] would not work because that requires have_separator to
        # be defined already
        self['have_separator'] = self.D_s['Nvol_s'] > 0

        # load electrode parameter file(s)
        self.paramfiles = {}

        cathode_paramfile = self.D_s['cathode']
        if not os.path.isabs(cathode_paramfile):
            cathode_paramfile = os.path.join(self.path, cathode_paramfile)
            self.paramfiles['c'] = cathode_paramfile
        self.D_c = ParameterSet(cathode_paramfile, 'electrode', self.path)

        if 'a' in self['trodes']:
            anode_paramfile = self.D_s['anode']
            if not os.path.isabs(anode_paramfile):
                anode_paramfile = os.path.join(self.path, anode_paramfile)
            self.paramfiles['a'] = anode_paramfile
            self.D_a = ParameterSet(anode_paramfile, 'electrode', self.path)
        else:
            self.D_a = None

        # set defaults and scale values that should be non-dim
        self.config_processed = False
        # either process the config, or read already processed config from disk
        prevDir = self['prevDir']
        if prevDir and prevDir != 'false':
            if not os.path.isabs(prevDir):
                # assume it is relative to the input parameter files
                self['prevDir'] = os.path.normpath(os.path.join(self.path, prevDir))
            self._process_config(self['prevDir'])
        else:
            self._process_config()

        self._verify_config()

    def _retrieve_config(self, items):
        """
        Select system or electrode config based on reqested parameter(s) and
        return the parameter and selected electrode

        :param str/tuple items: tuple of (electrode, item) to retrieve `item` from
            `electrode` config, or a single string to retrieve item from system config.
            `electrode` must be one of a, c.

        :return: config subset (system/chosen electrode/individual particle in electrode),
            item, electrode (None if system config)
        """
        if isinstance(items, tuple):
            # electrode config
            try:
                trode, item = items
            except ValueError:
                raise ValueError(f'Reading from electrode config requires two arguments, but '
                                 f'got {len(items)}')
            # select correct config
            if trode == 'a':
                assert self.D_a is not None, 'Anode parameter requested but ' \
                                             'anode is not simulated'
                d = self.D_a
            elif trode == 'c':
                d = self.D_c
            else:
                raise ValueError(f'Provided electrode must be a or c, got {trode}')
        else:
            # system config
            trode = None
            item = items
            d = self.D_s

        # select individual particle settings if requested
        if item in self.params_per_particle:
            assert trode is not None, 'Particle-specific parameter requested but ' \
                                      'electrode not specified'
            d = self[trode, 'indvPart']
        return d, item, trode

    def write(self, folder=None, filenamebase='input_dict'):
        """
        Write config to disk in pickled format.

        :param str folder: Folder in which to store the config (default: current folder)
        :param str filenamebase: prefix of filenames. These are appended with _system,
            _cathode, _anode, and _derived_values to create four config dicts on disk.
        """
        if folder:
            filenamebase = os.path.join(folder, filenamebase)

        # system, derived values, cathode, optionally anode
        dicts = {'system': self.D_s.params, 'derived_values': self.derived_values.values,
                 'cathode': self.D_c.params}
        if 'a' in self['trodes']:
            dicts['anode'] = self.D_a.params

        for section, d in dicts.items():
            with open(f'{filenamebase}_{section}.p', 'wb') as f:
                pickle.dump(d, f)

    def read(self, folder=None, filenamebase='input_dict', full=False):
        """
        Read previously processed config from disk. This also sets the numpy random seed
        if enabled in the config.

        :param str folder: Folder from which to read the config (default: current folder)
        :param str filenamebase: prefix of filenames. These are appended with _system,
            _cathode, _anode, and _derived_values to read the four config dicts from disk
        :param bool full: If true, all values from the dictionaries on disk are read.
            If false, only the generated particle distributions are read from the config dicts,
            i.e. the ``psd_*`` and ``G`` values in the system config, and the ``indvPart``
            section of the electrode configs
        """

        if folder:
            filenamebase = os.path.join(folder, filenamebase)

        # system, derived values, cathode, optionally anode
        sections = ['system', 'derived_values', 'cathode']
        if 'a' in self['trodes']:
            sections.append('anode')

        for section in sections:
            with open(f'{filenamebase}_{section}.p', 'rb') as f:
                try:
                    d = pickle.load(f)
                except UnicodeDecodeError:
                    d = pickle.load(f, encoding='latin1')

            if full:
                # update all config
                if section == 'system':
                    self.D_s.params = d
                elif section == 'derived_values':
                    self.derived_values.values = d
                elif section == 'cathode':
                    self.D_c.params = d
                elif section == 'anode':
                    self.D_a.params = d
                else:
                    raise Exception(f'Unknown section: {section}')
            else:
                # only update generated distributions
                if section == 'system':
                    for key in ['psd_num', 'psd_len', 'psd_area', 'psd_vol',
                                'psd_vol_FracVol', 'G','G_2']:
                        self[key] = d[key]
                elif section in ['anode', 'cathode']:
                    trode = section[0]
                    self[trode, 'indvPart'] = d['indvPart']

        # make sure to set the numpy random seed, as is usually done in process_config
        if self['randomSeed']:
            np.random.seed(self.D_s['seed'])

        self.params_per_particle = list(constants.PARAMS_PARTICLE.keys())
        self.config_processed = True

    def __getitem__(self, items):
        """
        ``__getitem__`` is the ``[]`` operator. It is used to retrieve the value of a parameter.
        If the parameter is found in the available derived values, it is extracted from there.
        Else, it is read from the config files.
        :meth:`_retrieve_config` is called to automatically
        selected the right config file. If the requested parameter does not exist,
        an ``UnknownParameterError`` is raised.

        :param str/tuple items: tuple of (electrode, item) to retrieve `item` from
            `electrode` config, or a single string to retrieve item from system config.
            `electrode` must be one of a, c.

        :return: parameter value

        Example usage:

        Extract the profileType parameter from the system config:

        >>> config['profileType']
        'CC'

        To access an electrode parameter, add 'c' for cathode or 'a' for anode as first argument.
        The requested parameter should be the second argument. To read the type parameter from the
        cathode config:

        >>> config['c', 'type']
        'ACR'

        The same parameter for the anode:

        >>> config['a', 'type']
        'CHR'

        Note that if the anode is not simulated, trying to access an anode parameter will result
        in an error:

        >>> config['a', 'type']  # from a config where Nvol_a = 0
        AssertionError: Anode parameter requested but anode is not simulated

        A parameter defined for each particle, delta_L:

        >>> config['c', 'delta_L']
        array([[10.1, 10.1],
               [10. , 10. ],
               [10.1,  9.9],
               [10.1,  9.9],
               [ 9.9, 10.2],
               [ 9.9, 10.2],
               [10.3, 10. ],
               [10. , 10.1],
               [10.1, 10.1],
               [ 9.9,  9.9]])

        Finally, derived values are handled transparently. The csmax parameter from
        both the cathode and anode:

        >>> config['c', 'csmax']
        22904.35071404849
        >>> config['a', 'csmax']
        28229.8239787446
        """
        # select config
        d, item, trode = self._retrieve_config(items)

        # check if the item is a derived value
        if item in self.derived_values.available_values:
            value = self.derived_values[self, item, trode]
        else:
            # not a derived value, can be read from config
            # raises UnknownParameterError if not found
            value = d[item]

        return value

    def __setitem__(self, items, value):
        """
        ``__setitem__`` is the ``[]`` operator when used in an assignment, e.g.
        ``config['parameter'] = 2``.
        It can be used to store a parameter in one of the config dicts. The dictionary is selected
        automatically in the same way as ``__getitem__`` does.

        :param str/tuple items: tuple of (electrode, item) to retrieve `item` from
            `electrode` config, or a single string to retrieve item from system config.
            `electrode` must be one of a, c.
        """
        # select config, ignore returned trode value as we don't need it
        d, item, _ = self._retrieve_config(items)
        # make sure to update derived value if that is where the value originally came from
        if item in self.derived_values.available_values:
            self.derived_values.values[item] = value
        else:
            d[item] = value

    def __delitem__(self, items):
        """
        ``__delitem__`` is the ``[]`` operator when used to delete a value,
        e.g. ``del config['parameter']``. The specified item is automatically deleted from
        the config dictionary where it was originally stored.

        :param str/tuple items: tuple of (electrode, item) to retrieve `item` from
            `electrode` config, or a single string to retrieve item from system config.
            `electrode` must be one of a, c.
        """
        # select config, ignore returned trode value as we don't need it
        d, item, _ = self._retrieve_config(items)

        del d[item]

    def _process_config(self, prevDir=None):
        """
        Process raw config after loading files from disk. This can only be done once per
        instance of ``Config``, as parameters are scaled in-place. Attempting to run it a
        second time results in an error. The processing consists of several steps:

        #. Set numpy random seed (if enabled in config)
        #. Set default values
        #. Scale to non-dimensional values
        #. Parse current/voltage segments
        #. Either generate particle distributions or load from previous run

        :param bool prevDir: if True, load particle distributions from previous run,
            otherwise generate them
        """
        if self.config_processed:
            raise Exception('The config can be processed only once as values are scaled in-place')
        self.config_processed = True

        # ensure paths are absolute
        self._make_paths_absolute()

        # set the random seed (do this before generating any random distributions)
        if self.D_s['randomSeed']:
            np.random.seed(self.D_s['seed'])

        # set default 1C_current_density
        limtrode = self['limtrode']
        theoretical_1C_current = self[limtrode, 'cap'] / 3600.  # A/m^2
        # print('Theoretical 1C current density: ', theoretical_1C_current)
        param = '1C_current_density'
        if self[param] is None:
            # set to theoretical value
            self[param] = theoretical_1C_current

        # apply scalings
        self._scale_system_parameters(theoretical_1C_current)
        self._scale_electrode_parameters()  # includes separator
        # reference voltage, should only be calculated after scaling of system and trode parameters
        Vref = self['c', 'phiRef']
        if 'a' in self['trodes']:
            Vref -= self['a', 'phiRef']
        self._scale_macroscopic_parameters(Vref)
        self._scale_current_voltage_segments(theoretical_1C_current, Vref)

        if prevDir:
            # load particle distrubtions etc. from previous run
            self.read(prevDir, full=False)
            # Set params per particle as would be done in _invdPart when generating distributions
            self.params_per_particle = list(constants.PARAMS_PARTICLE.keys())
        else:
            # particle distributions
            self._distr_part()
            # contact conductivty
            self._G_cont()
            # netwerk conductivty
            self._G_contNet()
            # connectivity matrix
            self._conn_matrix()
            # carbon coating conductivty
            self._G_carb()
            # bulk conductivty
            self._G()
            # Electrode parameters that depend on invidividual particle
            self._indvPart()

    def _scale_system_parameters(self, theoretical_1C_current):
        """
        Scale system parameters to non-dimensional values. This method should be called only once,
        from :meth:`_process_config`.
        """
        # non-dimensional scalings
        self['T'] = self['T'] / constants.T_ref
        self['Rser'] = self['Rser'] / self['Rser_ref']
        self['h_h'] = self['h_h'] * self['L_ref'] / self['k_h_ref']
        if self['Dp'] is not None:
            self['Dp'] = self['Dp'] / self['D_ref']
        if self['Dm'] is not None:
            self['Dm'] = self['Dm'] / self['D_ref']
        self['c0'] = self['c0'] / constants.c_ref
        if self['c01'] is not None:
            self['c01'] = self['c01'] / constants.c_ref
        if self['c02'] is not None:
            self['c02'] = self['c02'] / constants.c_ref
        self['phi_cathode'] = 0.  # TODO: why is this defined if always 0?
        self['currset'] = self['currset'] / (theoretical_1C_current * self['curr_ref'])
        if self['power'] is not None:
            self['power'] = self['power'] / (self['power_ref'])
        self['k0_foil'] = self['k0_foil'] / (self['1C_current_density'] * self['curr_ref'])
        self['Rfilm_foil'] = self['Rfilm_foil'] / self['Rser_ref']

    def _scale_electrode_parameters(self):
        """
        Scale electrode and separator parameters to non-dimensional values.
        This method should be called only once, from :meth:`_process_config`.
        """
        kT = constants.k * constants.T_ref

        # scalings per electrode
        self['beta'] = {}
        for trode in self['trodes']:
            self['L'][trode] = self['L'][trode] / self['L_ref']
            self['beta'][trode] = self[trode, 'csmax'] / constants.c_ref
            self['sigma_s'][trode] = self['sigma_s'][trode] / self['sigma_s_ref']
            self['k_h'][trode] = self['k_h'][trode] / self['k_h_ref']
            self['cp'][trode] = self['cp'][trode] / \
                (self['k_h_ref'] * self['t_ref'] / (self['rho_ref']*self['L_ref']**2))
            self['rhom'][trode] = self['rhom'][trode] / self['rho_ref']
            if self['nonisothermal']:
                if self['rhom'][trode] == 0 or self['cp'][trode] == 0 or self['k_h'][trode] == 0:
                    raise Exception(
                        "Please provide all nonisothermal parameters for the "
                        + trode
                        + " electrode and ensure they are nonzero")

            if self[trode, 'lambda'] is not None:
                self[trode, 'lambda'] = self[trode, 'lambda'] / kT
            if self[trode, 'lambda_1'] is not None:
                self[trode, 'lambda_1'] = self[trode, 'lambda_1'] / kT
            if self[trode, 'lambda_2'] is not None:
                self[trode, 'lambda_2'] = self[trode, 'lambda_2'] / kT
            if self[trode, 'intralayer_rxn']:
                self[trode, 'lambda_m'] = self[trode, 'lambda_m'] / kT
            if self[trode, 'intralayer_rxn']:
                self[trode, 'kintra'] = self[trode, 'kintra'] * self['t_ref']
            if self[trode, 'B'] is not None:
                self[trode, 'B'] = self[trode, 'B'] / (kT * constants.N_A * self[trode, 'cs_ref'])
            if self[trode, 'B1'] is not None:
                self[trode, 'B1'] = self[trode, 'B1'] / (kT * constants.N_A * self[trode, 'csmax'])
            if self[trode, 'B2'] is not None:
                self[trode, 'B2'] = self[trode, 'B2'] / (kT * constants.N_A * self[trode, 'csmax'])
            if self[trode, 'v_2'] is not None:
                self[trode, 'v_2'] = self[trode, 'v_2']
            for param in ['Omega_a', 'Omega_b', 'Omega_c', 'EvdW']:
                value = self[trode, param]
                if value is not None:
                    self[trode, param] = value / kT

        # scalings on separator
        if self['have_separator']:
            self['L']['s'] /= self['L_ref']
            self['k_h']['s'] = self['k_h']['s'] / self['k_h_ref']
            self['cp']['s'] = self['cp']['s'] / \
                (self['k_h_ref'] * self['t_ref'] / (self['rho_ref']*self['L_ref']**2))
            self['rhom']['s'] = self['rhom']['s'] / self['rho_ref']
            if self['nonisothermal']:
                if self['rhom']['s'] == 0 or self['cp']['s'] == 0 or self['k_h']['s'] == 0:
                    raise Exception(
                        "Please provide all nonisothermal parameters for the "
                        + " separator and ensure they are nonzero")

    def _scale_macroscopic_parameters(self, Vref):
        """
        Scale macroscopic input parameters to non-dimensional values and add
        reference values.
        This method should be called only once, from :meth:`_process_config`.
        """

        # scaling/addition of macroscopic input information
        factor = constants.e / (constants.k * constants.T_ref)
        if self['Vset'] is not None:
            self['Vset'] = -(factor * self['Vset'] + Vref)
        self['phimin'] = -(factor * self['Vmax'] + Vref)
        self['phimax'] = -(factor * self['Vmin'] + Vref)

    def _scale_current_voltage_segments(self, theoretical_1C_current, Vref):
        """
        Process current or voltage segments. This method should be called only once,
        from :meth:`_process_config`.
        """
        kT = constants.k * constants.T_ref

        # Scaling of current and voltage segments
        segments = []
        if self['profileType'] == 'CCsegments':
            for i in range(len(self['segments'])):
                segments.append((self["segments"][i][0]*self["1C_current_density"]
                                / theoretical_1C_current/self['curr_ref'],
                                self["segments"][i][1]*60/self['t_ref']))
        elif self['profileType'] == 'CVsegments':
            for i in range(len(self['segments'])):
                segments.append((-((constants.e/kT)*self['segments'][i][0]+Vref),
                                self['segments'][i][1]*60/self['t_ref']))

        # Current or voltage segments profiles
        segments_tvec = np.zeros(2 * self['numsegments'] + 1)
        segments_setvec = np.zeros(2 * self['numsegments'] + 1)
        if self['profileType'] == 'CCsegments':
            segments_setvec[0] = 0.
        elif self['profileType'] == 'CVsegments':
            segments_setvec[0] = -(kT / constants.e) * Vref
        tPrev = 0.
        for segIndx in range(self['numsegments']):
            tNext = tPrev + self['tramp']
            segments_tvec[2*segIndx+1] = tNext
            tPrev = tNext
            # Factor of 60 here to convert to s
            tNext = tPrev + (self['segments'][segIndx][1] * 60 - self["tramp"])
            segments_tvec[2*segIndx+2] = tNext
            tPrev = tNext
            setNext = self['segments'][segIndx][0]
            segments_setvec[2*segIndx+1] = setNext
            segments_setvec[2*segIndx+2] = setNext
        segments_tvec /= self['t_ref']
        if self['profileType'] == 'CCsegments':
            segments_setvec *= self["1C_current_density"]/theoretical_1C_current/self['curr_ref']
        elif self['profileType'] == 'CVsegments':
            segments_setvec = -((constants.e/kT)*segments_setvec + Vref)
        if 'segments' in self['profileType']:
            self['tend'] = segments_tvec[-1]
            # Pad the last segment so no extrapolation occurs
            segments_tvec[-1] *= 1.01
        else:
            self['tend'] /= self['t_ref']
        self['segments'] = segments
        self['segments_tvec'] = segments_tvec
        self['segments_setvec'] = segments_setvec
        if self['profileType'] == 'CC' and not np.allclose(self['currset'], 0., atol=1e-12):
            self['tend'] = np.abs(self['capFrac'] / self['currset'])

    def _make_paths_absolute(self):
        """
        Make paths in config absolute. This only applies to paths used in MPET simulations,
        not to the cathode/anode parameter files and prevDir.
        """
        # filenames in global config
        for key in ["SMset_filename"]:
            path = self[key]
            if path is not None and not os.path.isabs(path):
                self[key] = os.path.abspath(os.path.join(self.path, path))
        # filenames in trode config
        for key in ["rxnType_filename", "muRfunc_filename", "Dfunc_filename"]:
            for trode in self['trodes']:
                path = self[trode, key]
                if path is not None and not os.path.isabs(path):
                    self[trode, key] = os.path.abspath(os.path.join(self.path, path))

    def _distr_part(self):
        """
        Generate particle distributions and store in config.
        """
        # intialize dicts in config
        self['psd_num'] = {}
        self['psd_len'] = {}
        self['psd_area'] = {}
        self['psd_vol'] = {}
        self['psd_vol_FracVol'] = {}

        for trode in self['trodes']:
            solidType = self[trode, 'type']
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]

            # check if PSD is specified. If so, it is an ndarray so use np.all
            if not np.all(self['specified_psd'][trode]):
                # If PSD is not specified, make a length-sampled particle size distribution
                # Log-normally distributed
                mean_l = self['mean'][trode]
                stddev_l = self['stddev'][trode]
                mean_t = self[trode, 'thickness']
                stddev_t = self[trode, 'std_thickness']
                if np.allclose(stddev_l, 0., atol=1e-12):
                    raw_l = mean_l * np.ones((Nvol, Npart))
                else:
                    var_l = stddev_l**2
                    mu = np.log((mean_l**2) / np.sqrt(var_l + mean_l**2))
                    sigma = np.sqrt(np.log(var_l/(mean_l**2) + 1))
                    raw_l = np.random.lognormal(mu, sigma, size=(Nvol, Npart))
                if np.allclose(stddev_t, 0., atol=1e-12):
                    raw_t = mean_t * np.ones((Nvol, Npart))
                else:
                    var_t = stddev_t**2
                    mu = np.log((mean_t**2) / np.sqrt(var_t + mean_t**2))
                    sigma = np.sqrt(np.log(var_t/(mean_t**2) + 1))
                    raw_t = np.random.lognormal(mu, sigma, size=(Nvol, Npart))
            else:
                # use user-defined PSD
                raw_l = self['specified_psd'][trode]
                if raw_l.shape != (Nvol, Npart):
                    raise ValueError('Specified particle size distribution discretization '
                                     'of volumes inequal to the one specified in the config file')

            # For particles with internal profiles, convert psd to
            # integers -- number of steps
            solidDisc = self[trode, 'discretization']
            if solidType in ['ACR', 'ACR2']:
                psd_num = np.ceil(raw_l / solidDisc).astype(int)
                psd_len = solidDisc * psd_num
            elif solidType in ['CHR', 'diffn', 'CHR2', 'diffn2']:
                psd_num = np.ceil(raw_l / solidDisc).astype(int) + 1
                psd_len = solidDisc * (psd_num - 1)
            # For homogeneous particles (only one 'volume' per particle)
            elif solidType in ['homog', 'homog_sdn', 'homog2', 'homog2_sdn']:
                # Each particle is only one volume
                psd_num = np.ones(raw_l.shape, dtype=int)
                # The lengths are given by the original length distr.
                psd_len = raw_l
            else:
                raise NotImplementedError(f'Unknown solid type: {solidType}')

            # Calculate areas and volumes
            solidShape = self[trode, 'shape']
            if solidShape == 'sphere':
                psd_area = 4 * np.pi * psd_len**2
                psd_vol = (4. / 3) * np.pi * psd_len**3
            elif solidShape == 'C3':
                psd_area = 2 * 1.2263 * psd_len**2
                # psd_vol = 1.2263 * psd_len**2 * self[trode, 'thickness']
                psd_vol = 1.2263 * psd_len**2 * raw_t
            elif solidShape == 'cylinder':
                psd_area = 2 * np.pi * psd_len * self[trode, 'thickness']
                psd_vol = np.pi * psd_len**2 * self[trode, 'thickness']
            else:
                raise NotImplementedError(f'Unknown solid shape: {solidShape}')

            # Fraction of individual particle volume compared to total
            # volume of particles _within the simulated electrode
            # volume_
            psd_frac_vol = psd_vol / psd_vol.sum(axis=1, keepdims=True)

            # store values to config
            self['psd_num'][trode] = psd_num
            self['psd_len'][trode] = psd_len
            self['psd_area'][trode] = psd_area
            self['psd_vol'][trode] = psd_vol
            self['psd_vol_FracVol'][trode] = psd_frac_vol

    def _G_cont(self):
        """
        Generate inter-particle connectivity distribution of a one-dim wire and store in config.
        """
        self['G_cont'] = {}
        for trode in self['trodes']:
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]
            mean = self['G_mean_cont'][trode]
            stddev = self['G_std_cont'][trode]
            if np.allclose(stddev, 0, atol=1e-17):
                G_cont = mean * np.ones((Nvol, Npart))
            else:
                var = stddev**2
                mu = np.log((mean**2) / np.sqrt(var + mean**2))
                sigma = np.sqrt(np.log(var / (mean**2) + 1))
                G_cont = np.random.lognormal(mu, sigma, size=(Nvol, Npart))
            # scale and store
            self['G_cont'][trode] = G_cont * constants.k * constants.T_ref * self['t_ref'] \
                / (constants.e * constants.F * self[trode, 'csmax'] * self['psd_vol'][trode])
            
    def _G_contNet(self):
        self['G_contNet'] = {}
        for trode in self['trodes']:
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]
            mean = self['G_mean_cont'][trode]
            stddev = self['G_std_cont'][trode]
            # N*(N-1)/2 are the number of contacts
            # + N for the self-contact that is the contact with carbon black
            Nconn = int(Npart * (Npart - 1) / 2) + int(Npart)
            if np.allclose(stddev, 0, atol=1e-17):
                G_contNet = mean * np.ones((Nvol, Nconn))
            else:
                var = stddev**2
                mu = np.log((mean**2) / np.sqrt(var + mean**2))
                sigma = np.sqrt(np.log(var / (mean**2) + 1))
                G_contNet = np.random.lognormal(mu, sigma, size=(Nvol, Nconn))
            # note G_contNet is not normalized by the particle volume
            # the normalization is done in mod_cell 
            self['G_contNet'][trode] = G_contNet * constants.k * constants.T_ref * self['t_ref'] \
                / (constants.e * constants.F * self[trode, 'csmax'])
    
    def _conn_matrix(self):
        self['conn_matrix'] = {}
        
        for trode in self['trodes']:
            if self['simPartNet'][trode]:
                Nvol = self['Nvol'][trode]
                Npart = self['Npart'][trode]
                mean_n = self['avg_num_cont'][trode]
                # standard deviation of the contacts per particle
                std_n = self['std_num_cont'][trode]
                # penalty for the carbon black contact
                penalty_grid_cont = self['penalty_grid_cont'][trode]
                penalty_factor = self['penalty_factor'][trode]
                perc_grid = self['perc_grid'][trode]
                # number of lost contact if a particle is connected to carbon black
                self['conn_matrix'][trode] = {}
                for vInd in range(Nvol):
                    if Npart == 1:
                        self['conn_matrix'][trode][vInd] = [1]
                    else:
                        indeces = np.array([])
                        conn_mat = np.zeros((Npart, Npart))
                        if mean_n > Npart: # to be sure the while loop is not infinite
                            mean_n = Npart - 1
                        Numb_conn_vec = create_num_conn(mean_n, std_n, Npart)
                        for i in range(int(Npart*perc_grid)):
                            conn_mat[i,i] = 1

                        for i in range(Npart):
                            # generate random number of connections
                            orig_conn = int(Numb_conn_vec[i])
                            if i in range(int(Npart*perc_grid)):
                                # if we consider a penalty for the grid connection
                                # reduce the number of connections
                                if penalty_grid_cont:
                                    orig_conn = int(orig_conn*penalty_factor)
                            # max connection = n_part (considering self connection as conn to carbon)
                            # check how many connections are in that row
                            existing_conn = np.where(conn_mat[i, :] == 1)[0]
                            existing_conn = existing_conn[existing_conn != i]
                            # if there are already connections, subtract them from the total
                            conn_todo = orig_conn - len(existing_conn)
                            if conn_todo < 0:
                                conn_todo = 0
                            if conn_todo == 0:
                                continue
                            empty_positions = np.where(conn_mat[i, :] == 0)[0]
                            if len(empty_positions) == 0:
                                continue
                            # sort empty positions
                            empty_positions = np.sort(empty_positions)
                            # available positions = empty positions - already passed positions
                            available_positions = np.setdiff1d(empty_positions, indeces)
                            # if there are not enough empty positions, take the minimum
                            if conn_todo > len(available_positions):
                                random_positions = available_positions
                            else:
                                random_positions = np.random.choice(available_positions, size=conn_todo, replace=False)
                            conn_mat[i, random_positions] = 1
                            conn_mat[random_positions, i] = 1
                            indeces = np.append(indeces, i)

                        self['conn_matrix'][trode][vInd] = conn_mat
            else:
                self['conn_matrix'][trode] = {}
                for vInd in range(self['Nvol'][trode]):
                    self['conn_matrix'][trode][vInd] = np.ones((self['Npart'][trode], self['Npart'][trode]))
            
    def _G_carb(self):
        """
        Generate Carbon conductivity distribution and store in config.
        """
        self['G_car'] = {}
        for trode in self['trodes']:
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]
            mean = self['G_carb'][trode]
            G_carb = mean * np.ones((Nvol, Npart))
            # scale and store
            self['G_car'][trode] = G_carb * constants.k * constants.T_ref * self['t_ref'] \
                / (constants.e * constants.F * self[trode, 'csmax'] * self['psd_vol'][trode])

    def _G(self):
        """
        Generate Bulk connectivity and store in config.
        """
        self['G'] = {}
        for trode in self['trodes']:
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]
            mean = self['G_bulk'][trode]
            G = mean * np.ones((Nvol, Npart))
            # scale and store
            self['G'][trode] = G * constants.k * constants.T_ref * self['t_ref'] \
                / (constants.e * constants.F * self[trode, 'csmax'] * self['psd_vol'][trode])


    def _indvPart(self):
        """
        Generate particle-specific parameter values and store in config.
        """
        for trode in self['trodes']:
            Nvol = self['Nvol'][trode]
            Npart = self['Npart'][trode]
            self[trode, 'indvPart'] = {}

            # intialize parameters
            for param, dtype in constants.PARAMS_PARTICLE.items():
                self[trode, 'indvPart'][param] = np.empty((Nvol, Npart), dtype=dtype)

            # reference scales per trode
            cs_ref_part = constants.N_A * self[trode, 'cs_ref']  # part/m^3

            # calculate the values for each particle in each volume
            for i in range(Nvol):
                for j in range(Npart):
                    # This specific particle dimensions
                    self[trode, 'indvPart']['N'][i, j] = self['psd_num'][trode][i,j]
                    plen = self['psd_len'][trode][i,j]
                    parea = self['psd_area'][trode][i,j]
                    pvol = self['psd_vol'][trode][i,j]
                    # Define a few reference scales
                    F_s_ref = plen * cs_ref_part / self['t_ref']  # part/(m^2 s)
                    i_s_ref = constants.e * F_s_ref  # A/m^2
                    kappa_ref = constants.k * constants.T_ref * cs_ref_part * plen**2  # J/m
                    gamma_S_ref = kappa_ref / plen  # J/m^2
                    # non-dimensional quantities
                    if self[trode, 'kappa'] is not None:
                        self[trode, 'indvPart']['kappa'][i, j] = self[trode, 'kappa'] / kappa_ref
                    if self[trode, 'kappa1'] is not None:
                        kappa_ref1 = (constants.k * constants.T_ref * constants.N_A
                                      * self[trode, 'csmax'] * plen**2)
                        self[trode, 'indvPart']['kappa1'][i, j] = (self[trode, 'kappa1']
                                                                   / kappa_ref1)
                    if self[trode, 'kappa2'] is not None:
                        kappa_ref2 = (constants.k * constants.T_ref * constants.N_A
                                      * self[trode, 'csmax'] * plen**2)
                        self[trode, 'indvPart']['kappa2'][i, j] = (self[trode, 'kappa2']
                                                                   / kappa_ref2)
                    if self[trode, 'kappa12'] is not None:
                        kappa_ref12 = (constants.k * constants.T_ref * constants.N_A
                                      * self[trode, 'csmax'] * plen**2)
                        self[trode, 'indvPart']['kappa12'][i, j] = (self[trode, 'kappa12']
                                                                   / kappa_ref12)
                    if self[trode, 'dgammadc'] is not None:
                        nd_dgammadc = self[trode, 'dgammadc'] * cs_ref_part / gamma_S_ref
                        if self[trode, 'kappa'] is not None:
                            self[trode, 'indvPart']['beta_s'][i, j] = nd_dgammadc \
                                / self[trode, 'indvPart']['kappa'][i, j]
                            
                    if self[trode, 'dgammadc_1'] is not None:
                        nd_dgammadc1 = self[trode, 'dgammadc_1'] * cs_ref_part / gamma_S_ref
                        self[trode, 'indvPart']['beta_s1'][i, j] = nd_dgammadc1 \
                            / self[trode, 'indvPart']['kappa1'][i, j]

                        nd_dgammadc2 = self[trode, 'dgammadc_2'] * cs_ref_part / gamma_S_ref
                        self[trode, 'indvPart']['beta_s2'][i, j] = nd_dgammadc2 \
                            / self[trode, 'indvPart']['kappa2'][i, j]
                        
                    if self[trode, 'D'] is not None:
                        self[trode, 'indvPart']['D'][i, j] = self[trode, 'D'] \
                            * self['t_ref'] / plen**2
                    if self[trode,'surface_diffusion'] is not None:
                        self[trode,'indvPart']['D_surf'][i, j] = self[trode,'D_surf'] \
                            * self['t_ref'] / plen**2
                    if self[trode, 'E_D_surf'] is not None:
                        self[trode, 'indvPart']['E_D_surf'][i, j] = self[trode, 'E_D_surf'] \
                            / (constants.k * constants.N_A * constants.T_ref)
                    # self[trode, 'indvPart']['D'][i, j] = self[trode, 'D'] 
                    # * self['t_ref'] / plen**2
                    if self[trode, 'D1'] is not None:
                        self[trode, 'indvPart']['D1'][i, j] = self[trode, 'D1'] \
                            * self['t_ref'] / plen**2
                    if self[trode, 'D2'] is not None:
                        self[trode, 'indvPart']['D2'][i, j] = self[trode, 'D2'] \
                            * self['t_ref'] / plen**2
                    self[trode, 'indvPart']['E_D'][i, j] = self[trode, 'E_D'] \
                        / (constants.k * constants.N_A * constants.T_ref)
                    if self[trode, 'k0'] is not None:
                        self[trode, 'indvPart']['k0'][i, j] = self[trode, 'k0'] \
                            / (constants.e * F_s_ref)
                    if self[trode, 'k0_1'] is not None:
                        random_factor_1 = np.random.normal(1, 0.1)
                        if random_factor_1 < 0:
                            random_factor_1 = 0.01
                        F_s_1ref = plen * constants.N_A * self[trode, 'csmax'] / self['t_ref']
                        self[trode, 'indvPart']['k0_1'][i, j] = self[trode, 'k0_1']*random_factor_1 \
                            / (constants.e * F_s_1ref)
                    if self[trode, 'k0_2'] is not None:
                        random_factor_2 = np.random.normal(1, 0.1)
                        if random_factor_2 < 0:
                            random_factor_2 = 0.01
                        F_s_2ref = plen * constants.N_A * self[trode, 'csmax'] / self['t_ref']
                        self[trode, 'indvPart']['k0_2'][i, j] = self[trode, 'k0_2']*random_factor_2 \
                            / (constants.e * F_s_2ref)
                    self[trode, 'indvPart']['E_A'][i, j] = self[trode, 'E_A'] \
                        / (constants.k * constants.N_A * constants.T_ref)
                    self[trode, 'indvPart']['Rfilm'][i, j] = self[trode, 'Rfilm'] \
                        / (constants.k * constants.T_ref / (constants.e * i_s_ref))
                    self[trode, 'indvPart']['delta_L'][i, j] = (parea * plen) / pvol
                    # If we're using the model that varies Omg_a with particle size,
                    # overwrite its value for each particle
                    if self[trode, 'type'] in ['homog_sdn', 'homog2_sdn']:
                        self[trode, 'indvPart']['Omega_a'][i, j] = self.size2regsln(plen)
                    else:
                        # just use global value
                        self[trode, 'indvPart']['Omega_a'][i, j] = self[trode, 'Omega_a']

        # store which items are defined per particle, so in the future they are retrieved
        # per particle instead of from the values per electrode
        self.params_per_particle = list(constants.PARAMS_PARTICLE.keys())

    def _verify_config(self):
        """
        Verify configuration parameters.
        """
        # solid type
        for trode in self['trodes']:
            solidShape = self[trode, 'shape']
            solidType = self[trode, 'type']
            if solidType in ["ACR", "ACR2", "homog_sdn"] and solidShape != "C3":
                raise Exception("ACR, ACR2 and homog_sdn req. C3 shape")
            if (solidType in ["CHR", "diffn"] and solidShape not in ["sphere", "cylinder"]):
                raise NotImplementedError("CHR and diffn req. sphere or cylinder")

    @staticmethod
    def size2regsln(size):
        """
        This function returns the non-dimensional regular solution
        parameter which creates a barrier height that corresponds to
        the given particle size (C3 particle, measured in nm in the
        [100] direction). The barrier height vs size is taken from
        Cogswell and Bazant 2013, and the reg sln vs barrier height
        was done by TRF 2014 (Ferguson and Bazant 2014).

        :param float/array size: Size of the particle(s) (m)

        :return: regular solution parameter (float/array)
        """
        # First, this function wants the argument to be in [nm]
        size *= 1e+9
        # Parameters for polynomial curve fit
        p1 = -1.168e4
        p2 = 2985
        p3 = -208.3
        p4 = -8.491
        p5 = -10.25
        p6 = 4.516
        # The nucleation barrier depends on the ratio of the particle
        # wetted area to total particle volume.
        # *Wetted* area to volume ratio for C3 particles (Cogswell
        # 2013 or Kyle Smith)
        AV = 3.6338/size
        # Fit function (TRF, "SWCS" paper 2014)
        param = p1*AV**5 + p2*AV**4 + p3*AV**3 + p4*AV**2 + p5*AV + p6
        # replace values less than 2 with 2.
        if isinstance(param, np.ndarray):
            param[param < 2] = 2.
        elif param < 2:
            param = 2.
        return param

def lognorm(mean, std, N):
    var = std**2
    mu = np.log((mean**2) / np.sqrt(var + mean**2))
    sigma = np.sqrt(np.log((var)/(mean**2) + 1))
    return np.random.lognormal(mu, sigma, N)

def create_num_conn(mean_n, std_n, Npart):
    lognormal_values = lognorm(mean_n, std_n, Npart)
    outside_bounds = (lognormal_values < 1) | (lognormal_values > (Npart-1))
    while outside_bounds.any():
        lognormal_values[outside_bounds] = lognorm(mean_n, std_n, Npart)[outside_bounds]
        outside_bounds = (lognormal_values < 1) | (lognormal_values > (Npart-1))

    rounded_values = np.round(lognormal_values).astype(int)
    return rounded_values