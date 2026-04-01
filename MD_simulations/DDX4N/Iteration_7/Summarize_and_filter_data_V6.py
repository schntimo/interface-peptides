"""
Summarizes simulation data and generate csv file

If there are outliers or if equal or more than 50% of the simulations have failed, the data is flagged

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid as trp
import csv

# %% Inputs

path_folders_interface = 'Y:/DATA/SIMULATION_FILES/DDX4_Run_V1/Iteration_6/Simulation_files_interface/'
path_folders_homo = 'Y:/DATA/SIMULATION_FILES/DDX4_Run_V1/Iteration_6/Simulation_files_homo/'
path_sequence_files = './It_6_ddx4_sequences.txt'
filename_out = "results_DDX4_it6.csv"

n_ver_interface = 3
n_ver_homo = 8

Range_Ref_B = np.array([65,70])

all_folders_int = []
for i in range(1,100+1):
    full_path = path_folders_interface + 'Iter_6_' + str(i) + '/'
    all_folders_int.append(full_path)

all_folders_homo = []
for i in range(1,100+1):
    full_path = path_folders_homo + 'Iter_6_' + str(i) + '/'
    all_folders_homo.append(full_path)
    
    
std_limit_interface = 0.5 # in kcal/mol (FOR ENERGIES, LOG p reported)
std_limit_inout = 1 # in kcal/mol
std_imit_homo = 1e15 # in nm^3

T = 300
N_strat = 4 # for interface simulations
R = 8.314
Na = 6.022025e23
kcaltoJ = 4184
kT_kcalmol = R*T/kcaltoJ

show_plots = True
printing = False
write_csv = False


# %% Read sequences

# Read sequences
sequences = []
sim_names = []
with open(path_sequence_files, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        sequences.append(parts[1])
        sim_names.append(parts[0])


# %% Code interface

int_mean = np.zeros([np.size(all_folders_int)])
int_std = np.zeros([np.size(all_folders_int)])  
int_flags = np.zeros([np.size(all_folders_int)])  
dd_mean = np.zeros([np.size(all_folders_int)])  
dd_std = np.zeros([np.size(all_folders_int)])  
dd_flags = np.zeros([np.size(all_folders_int)])  

data_read_flag = np.zeros([np.size(all_folders_int)])  


for i_seq, path_int in enumerate(all_folders_int):
    values_int = []
    values_dd = []
    for version in range(1,n_ver_interface+1):
        version_sim_error = 0
        Filepath_prefix = path_int + 'V' + str(version) + '/'
        filepaths = []
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
        
        
        if show_plots == True:
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
            plt.xlabel('Distance [A]')
            plt.ylabel('PMF [kcal/mol]')
            plt.title(f'{i_seq+1}: {sequences[i_seq]}')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        dF_dense_dilute = y_values[0]-y_values[-1]
        dF_dense_inter = y_values[0]-min(y_values)
        
        if version_sim_error == 0:
            values_int.append(dF_dense_inter)
            values_dd.append(dF_dense_dilute)
        
        
        # Chain 2 - same code 
        filepaths = []
        for i in range(N_strat-1,-1,-1):
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
    

        if show_plots == True:
            plt.figure(figsize=(10, 6))
            plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
            plt.xlabel('Distance [A]')
            plt.ylabel('PMF [kcal/mol]')
            plt.title(f'{i_seq+1}: {sequences[i_seq]}')
            plt.grid(True)
            plt.legend()
            plt.show()
        
        dF_dense_dilute = y_values[0]-y_values[-1]
        dF_dense_inter = y_values[0]-min(y_values)
        
        if version_sim_error == 0:
            values_int.append(dF_dense_inter)
            values_dd.append(dF_dense_dilute)
            
            
    int_mean[i_seq] = np.mean(values_int)
    int_std[i_seq] = np.std(values_int)
    int_flags[i_seq] = np.std(values_int) > std_limit_interface
    
    dd_mean[i_seq] = np.mean(values_dd)
    dd_std[i_seq] = np.std(values_dd)
    dd_flags[i_seq] = np.std(values_dd) > std_limit_inout
    
    if len(values_int) <= 2*n_ver_interface/2:
        data_read_flag[i_seq] = 1 
        print(F"\n DATA READ FLAG AT {filepaths[-1]} \n'")        
    
    if printing == True:
        print(f'sequence = {sequences[i_seq]}')
        print(f'int_mean = {int_mean[i_seq]} +- {int_std[i_seq]}')
        print(f'dd_mean = {dd_mean[i_seq]} +- {dd_std[i_seq]}')
        print('\n')
        
        
# %% Code homotypic 

B_mean = np.zeros([np.size(all_folders_homo)])
B_std = np.zeros([np.size(all_folders_homo)])  
B_flags = np.zeros([np.size(all_folders_homo)]) 

data_read_flag_homo = np.zeros([np.size(all_folders_homo)])  

for i_seq, path_int in enumerate(all_folders_homo):
    data_read_fail_count_homo = 0
    values_B = []
    for version in range(1,n_ver_homo+1):
        filepath = path_int + 'V' + str(version) + '/test_colvar_out.pmf'
        
        # Read data from the file into NumPy arrays
        try:
            data = np.genfromtxt(filepath, skip_header=1)
            if len(data) > 2:
                x_values = data[:, 0]
                y_values = data[:, 1]
                
                Max_integrate = Range_Ref_B[0]
                PMF = y_values
                x_coord = x_values
                PMF_ref = np.mean(PMF[(x_coord > Range_Ref_B[0]) & (x_coord < Range_Ref_B[1])])
                PMF_norm = (PMF-PMF_ref)*kcaltoJ/(T*R)
                
                Mayer_f = np.exp(-PMF_norm) - 1
                x_coord_ext = np.append(0, x_coord)
                Mayer_f = np.append(-1, Mayer_f)
                
                Integral = Mayer_f*x_coord_ext**2
    
                Integral_calc = Integral[x_coord_ext < Max_integrate]
                x_coord_ext_calc = x_coord_ext[x_coord_ext < Max_integrate]
    
                Integral_value = trp(Integral_calc, x_coord_ext_calc)
                B = -2*np.pi*Integral_value/1000
                
                if show_plots == True:
                    plt.figure(figsize=(10, 6))
                    plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Data')
                    plt.xlabel('Distance [A]')
                    plt.ylabel('PMF [kcal/mol]')
                    plt.title(f'{i_seq+1}: {sequences[i_seq]}')
                    plt.grid(True)
                    plt.legend()
                    plt.show()
                
                values_B.append(B)         
                
            else:
                print(F'SIM ERROR AT {filepath} \n')
                data_read_fail_count_homo += 1
       
        except:
            print(F'SIM ERROR AT {filepath} \n')
            data_read_fail_count_homo += 1
            
        
    B_mean[i_seq] = np.mean(values_B)
    B_std[i_seq] = np.std(values_B)
    B_flags[i_seq] = np.std(values_B) > std_imit_homo
    
    if data_read_fail_count_homo >= n_ver_homo/2:
        data_read_flag_homo[i_seq] = 1 
        print(F'\n DATA READ FLAG AT {filepath} \n')        
    

    if printing == True:
        print(f'sequence = {sequences[i_seq]}')
        print(f'B_mean = {B_mean[i_seq]} +- {B_std[i_seq]}')
        print('\n')
        
        
# %% Print out data:
    
if write_csv == True:
    with open(filename_out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        row = ['SEQUENCE', 'dF_dI (mean)', 'dF_dI (std)', 'dF_dd (mean)', 'dF_dd (std)',
               'B (mean)', 'B (std)', 'int_flag', 'dd_flag', 'B_flag', 'dr_flag_int', 'dr_flag_homo',
               ]
        writer.writerow(row)
        
        for i, seq in enumerate(sequences):
            row = [seq, int_mean[i], int_std[i], dd_mean[i], dd_std[i], B_mean[i], B_std[i], int_flags[i],
                   dd_flags[i], B_flags[i], data_read_flag[i], data_read_flag_homo[i],
                   ]
            writer.writerow(row)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
