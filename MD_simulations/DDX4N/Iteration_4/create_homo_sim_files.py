# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:59:41 2024

@author: schntimo
"""

import os
import fnmatch


# %% Inputs

path_template = './Simulation_templates/Homo'
path_sequence_files = './It_4_ddx4_sequences.txt'
path_folder_sims = './Simulation_files_homo'


# %% Code

amino_acid_table = {'M': 1, 'G': 2, 'K': 3, 'T': 4, 'R': 5, 'A': 6, 'D': 7, 'E': 8, 'Y': 9, 'V': 10, 
                    'L': 11, 'Q': 12, 'W': 13, 'F': 14, 'S': 15, 'H': 16, 'N': 17, 'P': 18, 'C': 19, 'I': 20}

# Read sequences
sequences = []
sim_names = []
with open(path_sequence_files, 'r') as file:
    for line in file:
        parts = line.strip().split(': ')
        sequences.append(parts[1])
        sim_names.append(parts[0])
    
    
for i, seq in enumerate(sequences):
    # Create simulation folder
    folder_sim = path_folder_sims + '/' + sim_names[i]
    os.system(f'xcopy "{path_template}" "{folder_sim}" /E /I')
    
    # Rename job scripts
    name_batch = 'HI' + sim_names[i][3:]
    target_filename = "job_script.txt"
    for root, dirs, files in os.walk(folder_sim):
        if target_filename in files:
            file_path = os.path.join(root, target_filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
            lines[4] = f'#SBATCH --job-name="{name_batch}"\n'
            
            with open(file_path, 'w') as file:
                    file.writelines(lines)
                    
    # Change structure files
    AA_numbers = [amino_acid_table[char] for char in seq]*2
    for root, dirs, files in os.walk(folder_sim):
        for filename in fnmatch.filter(files, 'N_30*'):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r') as file:
                lines = file.readlines()
        
            new_lines = []
            for ind, line in enumerate(lines[56:116]):
                parts = line.split()
                parts[2] = str(AA_numbers[ind])
                new_lines.append(' '.join(parts) + '\n')
            lines[56:116] = new_lines
            with open(file_path, 'w') as file:
                file.writelines(lines)
                
                
    



