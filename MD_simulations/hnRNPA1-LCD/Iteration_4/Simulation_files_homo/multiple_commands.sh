#!/bin/bash

# Specify the directory where you want to execute the command in subfolders
directory="."

# Use a for loop to iterate through all subdirectories
for subdir in "$directory"/*; do
    if [ -d "$subdir" ]; then
	
		cd "$subdir"
		
		sh multiple_commands.sh	
		
		sleep 0.1
		
		cd ..
		
    fi
done