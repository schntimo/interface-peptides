import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.colors as mcolors

import MDAnalysis as mda
from MDAnalysis.analysis import pca, rms, align, distances
from MDAnalysis.lib.distances import augment_coordinates
from MDAnalysis.analysis.lineardensity import LinearDensity

from tqdm import tqdm

from scipy.signal import find_peaks, find_peaks_cwt

import pickle

# %% Inputs

# N_pep = 0
# top_file =  '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/0_pep/final_structure_LCDslab.dat'
# path_traj_files = ['\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/0_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed2/hnRNPA1/0_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed3/hnRNPA1/0_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed4/hnRNPA1/0_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed5/hnRNPA1/0_pep/pulltraj.lammpstrj',
#                    ]

# N_pep = 1
# top_file =  '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/1_pep/final_structure_LCDslab.dat'
# path_traj_files = ['\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/1_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed2/hnRNPA1/1_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed3/hnRNPA1/1_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed4/hnRNPA1/1_pep/pulltraj.lammpstrj',
#                    '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed5/hnRNPA1/1_pep/pulltraj.lammpstrj',
#                    ]

N_pep = 2
top_file =  '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/2_pep/final_structure_LCDslab.dat'
path_traj_files = ['\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed1\\hnRNPA1/2_pep/pulltraj.lammpstrj',
                   '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed2/hnRNPA1/2_pep/pulltraj.lammpstrj',
                   '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed3/hnRNPA1/2_pep/pulltraj.lammpstrj',
                   '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed4/hnRNPA1/2_pep/pulltraj.lammpstrj',
                   '\\\\d.ethz.ch\\groups\\chab\\icb\\Arosio\\Users\\schntimo\\_Doktorat\\ProjectFiles\\LCD_Design\\Reviews\\Organization_simulations\\V1\\seed5/hnRNPA1/2_pep/pulltraj.lammpstrj',
                   ]


range_frames = [0, 100000]    # all frames
stride = 1
n_prot_mol = 16

nbins = 100  # even number

# %% Code

def count_peaks(x_full, inst_rho_full, plotting=False, label='n/a'):

    # peaks, _ = find_peaks(inst_rho_full, distance=10, width=None, height=max(inst_rho_full)*1/2, prominence=max(inst_rho_full)*1/2)
    peaks = find_peaks_cwt(inst_rho_full, widths=10)
    No_peaks = len(peaks)
                    
    if plotting == True:
        
        plt.figure()
        plt.plot(x_full, inst_rho_full, 'bo')
        plt.xlabel('x [-]')
        plt.ylabel(r'$\rho$ [g/mL]')
        plt.title(label)
        plt.text(0.95, 0.95, f'N_peaks = {No_peaks}', verticalalignment='top', horizontalalignment='right',
         transform=plt.gca().transAxes, fontsize=12, color='black', fontweight='bold')
        plt.show()
        
    return No_peaks

dall_prot = []
dall_pep = []
dall_pos = []
dall_neg = []
dall_aro = []
dall_ali = []

for path_traj in path_traj_files:

    u = mda.Universe(top_file, path_traj, topology_format="DATA", format="LAMMPSDUMP")  
    box = u.dimensions
    box_center = box[:3] / 2 
    
    #define atom groups
    prot_all = u.residues[:n_prot_mol].atoms
    pep_all = u.residues[n_prot_mol:].atoms
    prot_pos = prot_all.select_atoms('type 3 or type 5')
    prot_neg = prot_all.select_atoms('type 7 or type 8')
    prot_aro = prot_all.select_atoms('type 9 or type 13 or type 14')
    prot_ali = prot_all.select_atoms('type 1 or type 2 or type 6 or type 10 or type 11 or type 18 or type 20')
    
    def density_along_z(z_coord, masses, Lz=box[2], nbins=nbins, area=box[0]*box[1]):                              
        mass_per_bin, edges = np.histogram(z_coord, bins=nbins, range=(0.0, Lz), weights=masses)
        dz = Lz / nbins
        rho = mass_per_bin / (area * dz)*1.66054  # Da/A^3 to g/mL                
        centers = 0.5 * (edges[1:] + edges[:-1])
        return rho, centers
    
    dens_prot = []
    dens_pep = []
    dens_pos = []
    dens_neg = []
    dens_aro =[]
    dens_ali = []
    
    accepted = 0
    total = 0
    for frame in tqdm(u.trajectory[range_frames[0]:range_frames[1]][::stride]):
        
        # get median z position -> shift & wrap to make slab whole
        protein_coord_raw = prot_all.positions
        median_z_coord = np.median(protein_coord_raw[:,2])
        
        u.atoms.translate(box_center - np.array([0, 0, median_z_coord]))
        u.atoms.wrap()
        
        # repeat now for slab COM --> leads to slab COM in box center
        protein_coord_t1 = prot_all.center_of_mass()
        u.atoms.translate(box_center - np.array([0, 0, protein_coord_t1[2]]))
        u.atoms.wrap()
        
        profile, centers = density_along_z(prot_all.positions[:,2], prot_all.masses)
        
        # Get density profiles (total_protein, total_peptide, protein_pos, protein_neg, protein_aro)
        No_droplets = count_peaks(centers, profile, plotting=False, label=None)
        
        if No_droplets == 1:
            dens_prot.append(density_along_z(prot_all.positions[:,2], prot_all.masses)[0])
            dens_pep.append(density_along_z(pep_all.positions[:,2], pep_all.masses)[0])
            dens_pos.append(density_along_z(prot_pos.positions[:,2], prot_pos.masses)[0])
            dens_neg.append(density_along_z(prot_neg.positions[:,2], prot_neg.masses)[0])
            dens_aro.append(density_along_z(prot_aro.positions[:,2], prot_aro.masses)[0])
            dens_ali.append(density_along_z(prot_ali.positions[:,2], prot_ali.masses)[0])
            
            accepted += 1
        total += 1
        
    print(f'{accepted/total*100} % accepted')
      
    
   
    # Half profile
    dp_prot = ((np.array(dens_prot).mean(axis=0) + np.array(dens_prot).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    dp_pep = ((np.array(dens_pep).mean(axis=0) + np.array(dens_pep).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    dp_pos = ((np.array(dens_pos).mean(axis=0) + np.array(dens_pos).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    dp_neg = ((np.array(dens_neg).mean(axis=0) + np.array(dens_neg).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    dp_aro = ((np.array(dens_aro).mean(axis=0) + np.array(dens_aro).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    dp_ali = ((np.array(dens_ali).mean(axis=0) + np.array(dens_ali).mean(axis=0)[::-1])/2)[:int(nbins/2)][::-1]
    
    centers = density_along_z(prot_all.positions[:,2], prot_all.masses)[1][:int(nbins/2)]

    dall_prot.append(dp_prot)
    dall_pep.append(dp_pep)
    dall_pos.append(dp_pos)
    dall_neg.append(dp_neg)
    dall_aro.append(dp_aro)
    dall_ali.append(dp_ali)
    
d_prot = np.array(dall_prot).mean(axis=0) 
d_pep = np.array(dall_pep).mean(axis=0) 
d_pos = np.array(dall_pos).mean(axis=0) 
d_neg = np.array(dall_neg).mean(axis=0) 
d_aro = np.array(dall_aro).mean(axis=0) 
d_ali = np.array(dall_ali).mean(axis=0) 

# %% save_stuff

data_final = np.vstack([centers, d_prot, d_pep, d_pos, d_neg, d_aro, d_ali])
np.save('comp_' + str(N_pep) + '.npy', data_final)







