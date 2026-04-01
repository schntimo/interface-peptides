# import mdtraj as md
import numpy as np


# %% INPUTS

No_hn = 16
beads_hn = 168

No_pep = 2
beads_pep = 30

Cutoff = 1.0      #Cutoff in nm for contacts
Cutoff_NL = 2   #Cutoff in nm for neighbor list

Every_X_frame = 1 # Analyze every X frame

Name_output = "INPUT_PLUMED_pep_10A.dat"

# %% Create File

with open(Name_output, "w") as file:
    for res_i in range(1, beads_hn+1):
        to_write = f"bead{res_i}: GROUP ATOMS="
        file.write(to_write)
        for mol_i in range(0, No_hn-1):
            ID = res_i+beads_hn*mol_i
            to_write = f"{ID},"
            file.write(to_write)
        ID = res_i+beads_hn*(No_hn-1)
        to_write = f"{ID}\n"
        file.write(to_write)
    
    
    for res_i_p in range(beads_hn+1, beads_hn+1+beads_pep):
        to_write = f"bead{res_i_p}: GROUP ATOMS="
        file.write(to_write)
        for mol_i_p in range(0, No_pep-1):
            ID = beads_hn*No_hn + (res_i_p-beads_hn)+beads_pep*mol_i_p
            to_write = f"{ID},"
            file.write(to_write)
        ID = beads_hn*No_hn + (res_i_p-beads_hn)+beads_pep*(No_pep-1)
        to_write = f"{ID}\n"
        file.write(to_write)
    file.write("\n")
    
    
    for res_i_hn in range(1, beads_hn+1):
        for res_i_pep in range(beads_hn+1, beads_hn+1+beads_pep):
            to_write = f"in{res_i_hn}_{res_i_pep}: COORDINATION GROUPA=bead{res_i_hn} GROUPB=bead{res_i_pep} R_0={Cutoff:.1f} NLIST NL_CUTOFF={Cutoff_NL:.1f} NL_STRIDE=1\n"
            file.write(to_write)
    file.write("\n")

    
    to_write="PRINT ARG="
    file.write(to_write)
    for res_i_hn in range(1, beads_hn+1):
        for res_i_pep in range(beads_hn+1, beads_hn+1+beads_pep):
            if res_i_hn == beads_hn and res_i_pep == beads_hn+beads_pep:
                to_write = f"in{res_i_hn}_{res_i_pep} STRIDE={Every_X_frame} FILE=all_bead_pepprot_colvar\n"
            else:
                to_write = f"in{res_i_hn}_{res_i_pep},"
            file.write(to_write)
    file.write("\n")
    
    
