import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.colors as mcolors

import MDAnalysis as mda
from MDAnalysis.analysis import pca, rms, align, distances
from MDAnalysis.lib.distances import augment_coordinates
from MDAnalysis.analysis.lineardensity import LinearDensity


from tqdm import tqdm

import pickle

# %% Inputs 

prot_name = 'prot'
top_file =  './seed1/production.tpr'
path_traj_files = ['./seed1/production.xtc',
                   './seed2/production.xtc',
                   './seed3/production.xtc',
                   './seed4/production.xtc',
                   './seed5/production.xtc',
                   './seed6/production.xtc',
                   './seed7/production.xtc',
                   './seed8/production.xtc',
                   './seed9/production.xtc',
                   './seed10/production.xtc',
                   './seed11/production.xtc',
                   './seed12/production.xtc',
                   './seed13/production.xtc',
                   './seed14/production.xtc',
                   './seed15/production.xtc',
                   './seed16/production.xtc',                   
                   ]

folder_eval = './'

first_step_eval = 40e6   # ps  1e6 = 1 us
stride = 1
n_prot_mol = 16

nbins = 100  # even number

# %% Code (main analysis)

dall_prot = []
dall_pep = []
all_time = []

for i_sim, path_traj in enumerate(path_traj_files):
    
    u = mda.Universe(top_file, path_traj) 
      
    #define atom groups
    prot_all = u.segments[0].atoms
    pep_all = u.segments[-1].atoms

    dens_prot = []
    dens_pep = []
    
    time_frame = []
    pos_pep1 = []
    pos_pep2 = []
    pos_prot = []
    idx_prot = [600, 630] # residue range (AA)
    time_pos = []

    for frame in tqdm(u.trajectory[::stride]):
        
        box = u.dimensions
        box_center = box[:3] / 2 
        
        def density_along_z(z_coord, masses, Lz=box[2], nbins=nbins, area=box[0]*box[1]):                              
            mass_per_bin, edges = np.histogram(z_coord, bins=nbins, range=(0.0, Lz), weights=masses)
            dz = Lz / nbins
            rho = mass_per_bin / (area * dz)*1.66054  # Da/A^3 to g/mL                
            centers = 0.5 * (edges[1:] + edges[:-1])
            return rho, centers
        
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
        
                        
        rho_z_prot = density_along_z(prot_all.positions[:,2], prot_all.masses)[0]
        rho_z_pep = density_along_z(pep_all.positions[:,2], pep_all.masses)[0]
        
        rho_z_prot_half = ((rho_z_prot + rho_z_prot[::-1])/2)[:int(nbins/2)][::-1]
        rho_z_pep_half = ((rho_z_pep + rho_z_pep[::-1])/2)[:int(nbins/2)][::-1]
        
        time_frame.append(frame.time)
        dens_prot.append(rho_z_prot_half)
        dens_pep.append(rho_z_pep_half)

        # For convergence analysis
        pos_pep1.append(pep_all.residues[:30].center_of_mass()[-1])
        pos_pep2.append(pep_all.residues[30:].center_of_mass()[-1])
        pos_prot.append(prot_all.residues[idx_prot[0]:idx_prot[1]].center_of_mass()[-1])
        time_pos.append(u.trajectory.time)
        
    plt.figure()
    plt.plot(np.array(time_pos)/1e6, pos_pep1, 'b-', label='pep1')
    plt.plot(np.array(time_pos)/1e6, pos_pep2, 'r-', label='pep2')
    plt.plot(np.array(time_pos)/1e6, pos_prot, 'g-', label='prot_segment')
    plt.xlabel('time ')
    plt.ylabel('x [A]')
    plt.title(f'Sim # {i_sim+1}')
    plt.legend()
    plt.grid()
    plt.ylim([0, box[2]])
    plt.show()
             
    centers = density_along_z(prot_all.positions[:,2], prot_all.masses)[1][:int(nbins/2)]

    dall_prot.append(dens_prot)
    dall_pep.append(dens_pep)
    all_time.append(time_frame)
    

# %% save_stuff

avg_prot = []
avg_pep = []

for idx_run, (t, prot, pep) in enumerate(zip(all_time, dall_prot, dall_pep)):
    
    idx_start = np.where(np.array(t) > first_step_eval)[0][0]
    
    print(f'{idx_start/len(t)*100:.1f} % of trajectory for equilibration')
    
    avg_prot.append(np.array(prot)[idx_start:].mean(axis=0))
    avg_pep.append(np.array(pep)[idx_start:].mean(axis=0))
    
    
d_prot = np.array(avg_prot).mean(axis=0)
d_pep = np.array(avg_pep).mean(axis=0)


matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Verdana']
matplotlib.rcParams['font.size'] = 10  # Adjust the size as needed

max_x = 14

# Plot 1--------------
fig, ax1 = plt.subplots(figsize=(4.5, 2.8))

# First axis (left) — protein
ax1.plot(centers/10, d_prot, color='k', label='Protein')
ax1.set_xlabel('x [nm]')
ax1.set_ylabel('Protein concentration [g/mL]', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.spines['left'].set_color('k')

# Second axis (right) — peptide, in red
ax2 = ax1.twinx()
ax2.plot(centers/10, d_pep, color='red', linestyle='--', label='Peptide')
ax2.set_ylabel('Peptide concentration [g/mL]', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.spines['right'].set_color('red')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
ax1.set_xlim([0,max_x])

fig.tight_layout()
fig.savefig(folder_eval + 'Profile' + '_' + prot_name + '.png', dpi=300)



# %% Binning & plotting (maximum peptide density)

time_bin_width = 1e6      # ps 1e6 = 1us

# get max time
t_max_traj = [max(t) for t in all_time]

max_t = np.max(t_max_traj)
edges_time = np.arange(0, np.max(t_max_traj), time_bin_width)
centers_time = edges_time[:-1] + time_bin_width/2


all_com_binned = []
all_cde_binned = []
for i_traj, t_dat in enumerate(all_time):
    
    idx = np.digitize(t_dat, edges_time, right=False) - 1
    idx = np.clip(idx, 0, len(edges_time) - 2)
    
    COM_pep = np.sum(np.array(dall_pep[i_traj])*centers, axis=1)/np.sum(np.array(dall_pep[i_traj]), axis=1)
    cde = np.array(dall_pep[i_traj])[:,0]
    
    sums_com = np.bincount(idx, weights=COM_pep, minlength=len(edges_time)-1)
    counts_com = np.bincount(idx, minlength=len(edges_time)-1)
    means_com = sums_com / counts_com
    means_com[counts_com==0] = np.nan
   
    sums_cde = np.bincount(idx, weights=cde, minlength=len(edges_time)-1)
    counts_cde = np.bincount(idx, minlength=len(edges_time)-1)
    means_cde = sums_cde / counts_cde
    means_cde[counts_cde==0] = np.nan
    
    all_com_binned.append(means_com)
    all_cde_binned.append(means_cde)
    
binned_mean_com = np.nanmean(np.array(all_com_binned), axis=0)
binned_std_com = np.nanstd(np.array(all_com_binned), axis=0)

cd_mean = np.nanmean(np.array(all_cde_binned), axis=0)
cd_std = np.nanstd(np.array(all_cde_binned), axis=0)

plt.figure(figsize=(4.5, 2.8))
plt.plot(centers_time/1e6, binned_mean_com, label="Mean", color="blue")
plt.fill_between(centers_time/1e6, binned_mean_com - binned_std_com/len(path_traj_files)**0.5, binned_mean_com + binned_std_com/len(path_traj_files)**0.5, alpha=0.3, color="blue")
plt.xlabel("Time [us]")
plt.ylabel("Peptide COM position [nm]")

# plt.legend()
plt.tight_layout()
plt.savefig(folder_eval + 'Convergence' + '_' + prot_name + '.png', dpi=300)
plt.show()

plt.figure(figsize=(4.5, 2.8))
plt.plot(centers_time/1e6, cd_mean, color="red")
plt.fill_between(centers_time/1e6, cd_mean - cd_std/len(path_traj_files)**0.5, cd_mean + cd_std/len(path_traj_files)**0.5, alpha=0.3, color="red", label='Dense')
plt.xlabel("Time [us]")
plt.ylabel("Dense peptide \nconcentration [g/mL]")
# plt.legend()
plt.tight_layout()
plt.savefig(folder_eval + 'Convergence_2' + '_' + prot_name + '.png', dpi=300)
plt.show()


# %% Save all relevant results


all_results = [dens_prot, dens_prot, dens_prot, d_prot, d_pep, centers, centers_time, binned_mean_com, binned_std_com, len(path_traj_files), cd_mean, cd_std, first_step_eval]

with open(prot_name + '_results.pkl', "wb") as f:
    pickle.dump(all_results, f)





