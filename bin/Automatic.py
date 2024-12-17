import os
import subprocess
import itertools
import configparser
import time


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
            thick = param_mat[('Particles', 'thickness')]
            std = 0.2 * float(thick)
            new_mat['Particles']['std_thickness'] = str(std)
            nicename_mat.append(f"{key[1]}={val}")
        with open(material_file, "w") as f:
            new_mat.write(f)

        for combin_sys in combinations_system:
            params_sys = dict(zip(keys_system, combin_sys))
            new_sys = cfg_sys
            nicename_sys = []
            for key, val in params_sys.items():
                new_sys[key[0]][key[1]] = val
                thick = param_mat[('Particles', 'thickness')]
                
                if thick == "600e-9":
                    new_sys['Sim Params']['Npart_c'] = "5"
                    mean_c = 2.5 * float(thick)
                    new_sys['Particles']['mean_c'] = str(mean_c)
                    new_sys['Particles']['stddev_c'] = str(mean_c*0.25)
                elif thick == "87e-9":
                    new_sys['Sim Params']['Npart_c'] = "50"
                    mean_c = float(thick)
                    new_sys['Particles']['mean_c'] = str(mean_c)
                    new_sys['Particles']['stddev_c'] = str(mean_c*0.25)
                else:
                    new_sys['Sim Params']['Npart_c'] = "100"
                    mean_c = 2.5 * float(thick)
                    new_sys['Particles']['mean_c'] = str(mean_c)
                    new_sys['Particles']['stddev_c'] = str(mean_c*0.25)
                    
                nicename_sys.append(f"{key[1]}={val}")
            with open(config_file, "w") as f:
                new_sys.write(f)

            new_folder_name = "-".join(nicename_mat + nicename_sys)
            new_folder_path = os.path.join(cwd, output_folder, new_folder_name)
            if os.path.exists(new_folder_path):
                print(f"Simulation {ind+1} of {num_mat * num_sys} already exists")
                ind += 1
                continue
                # os.rename(new_folder_path, os.path.join(cwd, output_folder,
                #                                         (new_folder_name + 'old')))
                
            os.chdir(cwd)
            minutes = 0
            seconds = minutes * 60
            print(f"Waiting for {minutes} min")
            time.sleep(seconds)

            run_MPET(cwd, os.path.join(cwd, "configs", config_file))

            os.chdir("history")
            folders = os.listdir()
            folders.sort()
            last_folder = folders[0]
            
            
            # if os.path.exists(new_folder_path):
            #     os.rename(new_folder_path, os.path.join(cwd, output_folder,
            #                                             (new_folder_name + 'old')))
            os.rename(last_folder, new_folder_path)
            os.chdir(cwd)
            os.chdir("configs")
            ind += 1
            print(f"Simulation {ind} of {num_mat * num_sys} completed")


system_properties = [
    # [("Electrolyte","c01"), ["0.1","0.5","1","2","3","4","5","10"]],
    # [("Particles","cs1_c"), ["0.1","0.3","0.5"]],
    [("Particles","cs1_c"), ["0.5","0.1"]],
    [("Particles","cs2_c"), ["0.001"]],
    # [("Electrolyte","c01"), ["1"]],
    # [("Electrolyte","c02"), ["1000"]],
    # [("Particles","mean_c"), ["60e-9"]],
    # [("Particles","mean_c"), ["plat20","plat40","cub80"]],
    # [("Particles","stddev_c"), ["10e-9"]],
    # [("Sim Params","Crate"), ["0.1","0.2","0.5","1"]],
    [("Sim Params","Crate"), ["0"]],
    # [("Sim Params","segments"), ["[(1, 42)]","[(0.1, 420)]"]],
    ]

material_properties = [
    # [("Reactions", 'k0_2'), ["0.1","0.5","1","5","10"]],
    [("Reactions", 'k0_1'), ["200"]],
    [("Reactions", 'k012_ratio'), ["2e3"]],
    # [("Material", 'Omega_c'), ["0","2e-20","6e-20","-4e-20"]],
    [("Material", 'Omega_b'), ["1e-20"]],
    [("Material", 'Omega_c'), ["-3e-20"]],
    [("Material", 'v_2'), ["0.075"]],
    [("Material", 'kappa12'), ["10e-10"]],
    [("Material", 'kappa1'), ["10e-10"]],
    # [("Material", 'B2'), ["1e9"]],
    # [("Material", 'cwet_1'), ["0.01","0.1","0.25","0.5","0.75","0.99"]],
    # [("Material", 'cwet_2'), ["0.1"]],
    # [("Material", 'dgammadc_2'), ["50e-30"]],
    # [("Material", 'dgammadc_1'), ["40e-30"]],
    # [("Reactions", 'Rfilm'), ["0"]],
    [("Particles", 'thickness'), ["20e-9"]],
    # [("Particles", 'std_thickness'), ["2e-9"]],
    ]


output_folder = "Li_extr/non_farad_exch2"
config_file = 'params_system_mutlicat.cfg'
material_file = 'params_LNFP.cfg'


# hours = 3
# minutes = hours * 60
# seconds = minutes * 60
# print(f"Waiting for {hours} hours")
# time.sleep(seconds)

run_params_mpet(config_file, material_file, system_properties, material_properties, output_folder)
