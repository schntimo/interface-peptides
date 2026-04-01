import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trp
import csv

all_folders_int = []

# %% Inputs

n_ver_interface = 10
full_path = './Results/'

# %% Inputs

all_folders_int.append(full_path)

N_strat = 4 # for interface simulations

profiles = []
dG_1 = []
dG_2 = []

for i_seq, path_int in enumerate(all_folders_int):
    values_int = []
    values_dd = []
    for version in range(1,n_ver_interface+1):
        version_sim_error = 0
        Filepath_prefix = path_int + 'V' + str(version) + '/'
        filepaths = []
        for i in range(N_strat-1,-1,-1):
        # for i in range(N_strat):
            filepaths.append(Filepath_prefix + f'W{i+1}/colvar_out.abf_v1.pmf')
        
        # Read data from the file into NumPy arrays
        data = np.array([]).reshape(0,2)
        for i in range(0, len(filepaths)):
            try:
                data_temp = np.genfromtxt(filepaths[i], skip_header=1)
                if len(data_temp) > 2:
                    if i > 0:
                        data_temp2 = data_temp[1:,:]
                        data_temp2[:,1] = data_temp2[:,1]-(data_temp[0,1]-data[-1,1])
                    else:
                        data_temp2 = data_temp
                    data = np.concatenate((data, data_temp2), axis=0)
                else:
                    print(F'SIM ERROR AT {filepaths[-1]} \n')
                    version_sim_error = 1
                    data = np.array([[0, 0],[0, 0]])
                    
            except:
                print(F'SIM ERROR AT {filepaths[-1]} \n')
                version_sim_error = 1
                data = np.array([[0, 0],[0, 0]])
        
        x_values = data[:, 0]
        y_values = data[:, 1]
        

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
        plt.xlabel('Distance [A]')
        plt.ylabel('PMF [kcal/mol]')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        dF_dense_dilute = y_values[0]-y_values[-1]
        dF_dense_inter = y_values[0]-min(y_values)
        
        dG_1.append(dF_dense_inter)
        dG_2.append(dF_dense_dilute)
        
        if version_sim_error == 0:
            values_int.append(dF_dense_inter)
            values_dd.append(dF_dense_dilute)
            
        profiles.append(y_values-np.mean(y_values[:1]))
        
        
        
        # Chain 2 - same code 
        filepaths = []
        # for i in range(N_strat-1,-1,-1):
        for i in range(N_strat):
            filepaths.append(Filepath_prefix + f'W{i+1}/colvar_out.abf_v2.pmf')
        
        # Read data from the file into NumPy arrays
        data = np.array([]).reshape(0,2)
        for i in range(0, len(filepaths)):
            try:
                data_temp = np.genfromtxt(filepaths[i], skip_header=1)
                if len(data_temp) > 2:
                    if i > 0:
                        data_temp2 = data_temp[1:,:]
                        data_temp2[:,1] = data_temp2[:,1]-(data_temp[0,1]-data[-1,1])
                    else:
                        data_temp2 = data_temp
                    data = np.concatenate((data, data_temp2), axis=0)
                else:
                    print(F'SIM ERROR AT {filepaths[-1]} \n')
                    version_sim_error = 1
                    data = np.array([[0, 0],[0, 0]])
                    
            except:
                print(F'SIM ERROR AT {filepaths[-1]} \n')
                version_sim_error = 1
                data = np.array([[0, 0],[0, 0]])
             
        
        x_values = data[:, 0]
        y_values = data[:, 1]
    
        profiles.append(y_values-np.mean(y_values[:1]))

        plt.figure(figsize=(10, 6))
        plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
        plt.xlabel('Distance [A]')
        plt.ylabel('PMF [kcal/mol]')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        dF_dense_dilute = y_values[0]-y_values[-1]
        dF_dense_inter = y_values[0]-min(y_values)
        
        dG_1.append(dF_dense_inter)
        dG_2.append(dF_dense_dilute)

profiles = np.array(profiles)

print(f'dG_1 = {np.mean(dG_1):.2f} +- {np.std(dG_1):.2f}')
        

# %% Plotting

import matplotlib
import matplotlib.colors as mcolors

matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 12  # Adjust the size as needed

color_m = mcolors.CSS4_COLORS['black']
color_std = mcolors.CSS4_COLORS['grey']

mean_profile = np.mean(profiles, axis=0)
std_profile = np.std(profiles, axis=0)

ylim = [-3.5, 9.8]

mean_profile = np.mean(profiles, axis=0)
std_profile = np.std(profiles, axis=0)

color_m = mcolors.CSS4_COLORS['black']
color_std = mcolors.CSS4_COLORS['grey']

fig, axes = plt.subplots(figsize=(6, 3))

plt.plot(x_values/10, mean_profile, color=color_m, linewidth=2)
plt.fill_between(x_values/10, mean_profile - std_profile, mean_profile + std_profile, color=color_std, alpha=0.2)
plt.xlabel('x [nm]', labelpad=0.2)
plt.ylim([ylim[0], ylim[1]])
plt.title(r'$N_{\text{prot}}$ = 128')
plt.ylabel('PMF [kcal/mol]', labelpad=0.2)

plt.xlim(0, np.max(x_values/10))

plt.subplots_adjust(wspace=0.3, bottom=0.2)

# plt.tight_layout()
plt.savefig('pmf_large.png', dpi=400, transparent=True)
plt.show()
    
           
