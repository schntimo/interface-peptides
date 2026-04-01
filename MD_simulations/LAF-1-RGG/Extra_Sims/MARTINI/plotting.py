import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.colors as mcolors

import pickle

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Verdana']
matplotlib.rcParams['font.size'] = 11  # Adjust the size as needed


# %% Inputs

path = './laf1_results.pkl'

size_dens_plot = (6.5, 2.8)
size_conv1 = (3.3, 2.8)


# %% LAF-1

name = 'prot'
max_x = 15

with open(path, "rb") as f:
    loaded_list = pickle.load(f)

centers = loaded_list[5]
d_prot = loaded_list[3]
d_pep = loaded_list[4]

# Plot 1--------------
fig, ax1 = plt.subplots(figsize=size_dens_plot)

# First axis (left) — protein
ax1.plot(centers/10, d_prot, color='k', label='Protein')
ax1.set_xlabel('x [nm]')
ax1.set_ylabel(r'$\rho_\text{protein}$ [g/mL]', color='k')
ax1.tick_params(axis='y', labelcolor='k')
ax1.spines['left'].set_color('k')


ax2 = ax1.twinx()
ax2.plot(centers/10, d_pep, color='red', linestyle='--', label='Peptide')
ax2.set_ylabel(r'$\rho_\text{peptide}$ [g/mL]', color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.spines['right'].set_color('red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False, ncol=2)
ax1.set_xlim([0,max_x])

fig.tight_layout()
# fig.savefig(name + 'dens_profile.png', dpi=100,  transparent=True)

# Convergence 1

centers_time = loaded_list[6]
cd_mean = loaded_list[10]
cd_std = loaded_list[11]
n_repeats = loaded_list[9]

plt.figure(figsize=size_conv1)
plt.plot(centers_time/1e6, cd_mean, color="red")
plt.fill_between(centers_time/1e6, cd_mean - cd_std/n_repeats**0.5, cd_mean + cd_std/n_repeats**0.5, alpha=0.3, color="red", label='Dense')
plt.xlabel(r'Time [$\mu$s]')
plt.ylabel(r'$\rho_\text{peptide}$ [g/mL]')
plt.title('Dense phase concentration (x=0)', fontsize=11)
# plt.legend()
plt.tight_layout()
# plt.savefig(name + 'conv_pep_den.png', dpi=100, transparent=True)
plt.show()

# Convergence 2
binned_mean_com = loaded_list[7]
binned_std_com = loaded_list[8]

plt.figure(figsize=size_conv1)
plt.plot(centers_time/1e6, binned_mean_com/10, label="Mean", color="blue")
plt.fill_between(centers_time/1e6, binned_mean_com/10 - binned_std_com/10/n_repeats**0.5, binned_mean_com/10 + binned_std_com/10/n_repeats**0.5, alpha=0.3, color="blue")
plt.xlabel(r'Time [$\mu$s]')
plt.ylabel("x [nm]")
plt.title('Peptide center-of-mass position', fontsize=11)
# plt.legend()
plt.tight_layout()
# plt.savefig(name + 'conv_com.png', dpi=100, transparent=True)
plt.show()


