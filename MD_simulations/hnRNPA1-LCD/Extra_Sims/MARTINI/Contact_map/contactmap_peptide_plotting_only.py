# import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import AutoMinorLocator
import matplotlib
from matplotlib.ticker import FixedLocator

# %% Inputs

beads_hn = 140




No_pep = 2
beads_pep = 30

Min_colorbar_input = 0
step = 0.1

filename_out = 'contact_map_pep.png'
 
# plotting stuff
fontsize = 13

tick_intervals_x = 5
small_step_x = 1
tick_intervals_y = 20
small_step_y = 2

matplotlib.rcParams['font.serif'] = ['Times New Roman']

# Set font properties globally
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = fontsize  # Adjust the size as needed

bead_inter_DHH1C_CG_11_no4_wt_contactmap_norm = np.load('./cmap.npy')
bead_inter_DHH1C_CG_11_no4_wt_average = np.load('./cmap_ave.npy')

# %% PLOT

Max_colorbar = np.percentile(bead_inter_DHH1C_CG_11_no4_wt_contactmap_norm, 99.5)

Min_colorbar = Min_colorbar_input

# Histogram contact_frequencies
nbins = 20
plt.figure()
plt.hist(bead_inter_DHH1C_CG_11_no4_wt_average/No_pep, bins=nbins, color='blue', edgecolor='black')
plt.xlabel('Contacts')
plt.ylabel('counts')
plt.title('Histogram Contact maps')

# Contact map
colmap = "viridis"

fig = plt.figure(figsize=(10, 10))

gw = int(np.floor(0.5 + 100 * fig.get_figwidth()))
gh = int(np.floor(0.5 + 100 * fig.get_figheight()))
gs = plt.GridSpec(gh, gw)
gs.update(hspace=0.0, wspace=0.0)

axes00 = fig.add_subplot(gs[50:600, 50:470])
axes01 = fig.add_subplot(gs[50:600, 510:540])
axes03 = fig.add_subplot(gs[650:770, 50:470])

a = axes00.matshow(bead_inter_DHH1C_CG_11_no4_wt_contactmap_norm.T,cmap=colmap,aspect='auto')

cbar = fig.colorbar(a,cax=axes01)
cbar.mappable.set_clim(Min_colorbar,Max_colorbar)
cbar.set_ticks(np.arange(Min_colorbar,Max_colorbar+0.01,step))

norm=plt.Normalize(Min_colorbar,4)
cmap=plt.get_cmap(colmap)

axes00.xaxis.set_label_coords(0.5, 1.14)
axes00.yaxis.set_label_coords(-0.15, 0.5)
axes00.set_xlabel("Residue # peptide",fontsize=fontsize, labelpad=0)
axes00.set_ylabel("Residue # protein",fontsize=fontsize)
axes00.set_xlim([0,beads_pep-1])
axes00.set_ylim([0,beads_hn-1])
major_x_ticks = np.concatenate(([0], np.arange(tick_intervals_x-1,beads_pep,tick_intervals_x)))
major_y_ticks = np.concatenate(([0], np.arange(tick_intervals_y-1,beads_hn,tick_intervals_y)))
minor_x_ticks = np.arange((beads_pep-1) % small_step_x, beads_pep, small_step_x)
minor_y_ticks = np.arange((beads_hn-1) % small_step_y, beads_hn, small_step_y)

axes00.set_xticks(major_x_ticks)
axes00.set_yticks(major_y_ticks)
axes00.xaxis.set_major_locator(FixedLocator(major_x_ticks ))
axes00.yaxis.set_major_locator(FixedLocator(major_y_ticks ))
axes00.xaxis.set_minor_locator(FixedLocator(minor_x_ticks))
axes00.yaxis.set_minor_locator(FixedLocator(minor_y_ticks))
axes00.set_xticklabels(major_x_ticks+1)
axes00.set_yticklabels(major_y_ticks+1)


axes01.tick_params(direction='out', which='major',length=5, width=1.0, labelsize=fontsize)
axes01.set_ylabel('Number of contacts', fontsize=fontsize, labelpad=5)

axes00.grid(which="both", linewidth=0.5)

# # Plot total contacts
axes03.plot(np.arange(1,beads_pep+1), np.sum(bead_inter_DHH1C_CG_11_no4_wt_contactmap_norm, axis=1), marker='o', linestyle='-', color='k')
axes03.set_xticks(major_x_ticks+1)
axes03.set_ylim(0, 25)
plt.ylabel('Total contacts')
# plt.xlabel('Residue # peptide')
# axes00.set_xticklabels("")

#Finish up
plt.tight_layout()
# fig.savefig(filename_out, dpi=600, transparent=True)


