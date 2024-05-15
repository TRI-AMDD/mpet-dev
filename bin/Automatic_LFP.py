import os
import subprocess
import itertools
import configparser


def run_MPET(cwd, config):
    os.chdir(cwd)
    subprocess.call(["python", os.path.join(cwd, "bin", "mpetrun.py"), config])


def ensemble_definitions(parameters):
    keys, vals = zip(*parameters)
    return keys, vals


def run_params_mpet(config_file, material_file,
                    system_properties, material_properties, output_folder):
    cwd = os.getcwd()
    os.chdir(cwd)
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    os.chdir("configs")

    keys_system, val_system = ensemble_definitions(system_properties)
    cfg_sys = configparser.ConfigParser()
    cfg_sys.optionxform = str
    cfg_sys.read(config_file)
    combinations_system = list(itertools.product(*val_system))
    num_sys = len(combinations_system)

    keys_mat, val_mat = ensemble_definitions(material_properties)
    cfg_mat = configparser.ConfigParser()
    cfg_mat.optionxform = str
    cfg_mat.read(material_file)
    combinations_material = list(itertools.product(*val_mat))
    num_mat = len(combinations_material)
    ind = 0

    for comb_mat in combinations_material:
        param_mat = dict(zip(keys_mat, comb_mat))
        new_mat = cfg_mat
        nicename_mat = []
        for key, val in param_mat.items():
            new_mat[key[0]][key[1]] = val
            nicename_mat.append(f"{key[1]}={val}")
        with open(material_file, "w") as f:
            new_mat.write(f)

        for combin_sys in combinations_system:
            params_sys = dict(zip(keys_system, combin_sys))
            new_sys = cfg_sys
            nicename_sys = []
            for key, val in params_sys.items():
                new_sys[key[0]][key[1]] = val
                nicename_sys.append(f"{key[1]}={val}")
            with open(config_file, "w") as f:
                new_sys.write(f)

            os.chdir(cwd)
            run_MPET(cwd, os.path.join(cwd, "configs", config_file))

            os.chdir("history")
            folders = os.listdir()
            folders.sort()
            last_folder = folders[0]
            new_folder_name = "-".join(nicename_mat + nicename_sys)
            new_folder_path = os.path.join(cwd, output_folder, new_folder_name)
            if os.path.exists(new_folder_path):
                os.rename(new_folder_path, os.path.join(cwd, output_folder,
                                                        (new_folder_name + 'old')))
            os.rename(last_folder, new_folder_path)
            os.chdir(cwd)
            os.chdir("configs")
            ind += 1
            print(f"Simulation {ind} of {num_mat * num_sys} completed")


system_properties = [
    # [("Conductivity","E_G_c"), ["0.4"]],
    # [("Conductivity","avg_num_cont_c"), ["2"]],
    # [("Conductivity","std_num_cont_c"), ["3"]],
    # [("Conductivity","sigma_s_c"), ["0.03"]],
    # [("Conductivity","perc_grid_c"), ["0.1","0.5","0.8"]],
    # [("Conductivity","c_dep_exp_c"), ["1"]],
    # [("Conductivity","simPartNet_c"), ["true"]],
    # [("Sim Params","seed"), ["0"]],
    # [("Conductivity","sigma_s_c"), ["0.5"]],
    # [("Sim Params","Npart_c"), ["20"]],
    # [("Sim Params","Nvol_c"), ["2"]],
    # [("Particles","mean_c"), ["25e-9"]],
    # [("Particles","stddev_c"), ["15e-9"]],
    # [("Sim Params","T"), ["268","283","298"]],
    [("Sim Params","Crate"), ["0.5","2"]],
    ]

material_properties = [
    # [("Particles", 'type'), ["CHR"]],
    [("Reactions", 'k0'), ["15"]],
    # [("Material", 'D'), ["7.5e-17"]],
    ]


output_folder = "LFP_CC/conc_dep_cond"
config_file = 'params_system_net_SS.cfg'
material_file = 'params_LFP_ss.cfg'


run_params_mpet(config_file, material_file, system_properties, material_properties, output_folder)
