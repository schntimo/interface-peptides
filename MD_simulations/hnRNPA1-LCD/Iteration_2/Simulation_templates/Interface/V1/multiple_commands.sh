#!/bin/bash

# Specify the directory where you want to execute the command in subfolders
directory="."

# Use a for loop to iterate through all subdirectories
for subdir in "$directory"/*; do
    if [ -d "$subdir" ]; then
	
		cd "$subdir"
		pwd
		
        #/cluster/home/schntimo/LAMMPS_installation/lammps-23Jun2022/build/lmp -in input-script_replicate.dat
		#mv input-structure-slab-run_raw.dat input-structure-slab-run.dat
        sbatch job_script.txt
		sleep 1
		
		cd ..
		
    fi
done