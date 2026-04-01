# import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.colors as mcolors

from tqdm import tqdm

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Verdana']
matplotlib.rcParams['font.size'] = 12  # Adjust the size as needed

# %% Inputs 

paths = ['./comp_0.npy',
        './comp_1.npy',
        './comp_2.npy',
        ]

data = np.array([np.load(path) for path in paths])
             
# %% Plotting function

def plot_profile(x_data, y_data, name, title=None, color='k', limits=[0,10],
                 labels=[r'$N_{\text{pep}} = 0$', r'$N_{\text{pep}} = 2$', r'$N_{\text{pep}} = 4$'],
                 linetypes=['-','-.','--'], normalize_dense = True):
    
    
    plt.figure(figsize=(4.2,2.5))
    for idx, x in enumerate(x_data):
        if normalize_dense:
            plt.plot(x/10, y_data[idx]/y_data[idx][0], label=labels[idx], color=color, linestyle=linetypes[idx], linewidth=2.5)
        else:
            plt.plot(x/10, y_data[idx], label=labels[idx], color=color, linestyle=linetypes[idx], linewidth=2.5)
    
    plt.xlabel('x [nm]')
    if normalize_dense:
        plt.ylabel(r'$\rho$ / $\rho_{\text{dense}}$ ')
    else:
        plt.ylabel(r'$\rho$ [g/mL]')
    plt.title(title, fontsize=11)
    plt.tight_layout()
    plt.xlim(limits)
    plt.legend(frameon=False, fontsize=11, loc='upper right')
    plt.savefig('./Eval_files/Plots_organization/' + name, dpi=300, transparent=True)  
    plt.show()
    
def plot_all_profiles(x, y_data, name, limits=[0,10],
                      labels=['Positive', 'Negative', 'Aromatic', 'Aliphatic'],
                      colors=['b','r','g','k']):
               
    plt.figure(figsize=(4.2,3))
    for idx, y in enumerate(y_data):
        plt.plot(x/10, y/y[0], label=labels[idx], color=colors[idx], linestyle='-', linewidth=2.2)
    
    plt.xlabel('x [nm]')
    plt.ylabel(r'$\rho$ / $\rho_{\text{dense}}$ ')
    # plt.title(r'$N_{\text{pep}} = 2$')
    plt.tight_layout()
    plt.xlim(limits)
    plt.ylim([0, 1.2])
    plt.legend(frameon=False, fontsize=11, loc='upper right')
    plt.show()

# %% 

plot_all_profiles(data[1,0], 
                  data[1,3:7],
                  'all.png',
                  limits = [0,12],
                  )

#pos
plot_profile(data[:,0], 
             data[:,3], 
             'pos.png',
             title='Positive (K,R)', 
             color='b',
             limits = [0,12],
             )

#neg
plot_profile(data[:,0], 
             data[:,4], 
             'neg.png',
             title='Negative (D,E)', 
             color='r',
             limits = [0,12],
             )

#aro
plot_profile(data[:,0], 
             data[:,5], 
             'aro.png',
             title='Aromatic (Y,W,F)', 
             color='g',
             limits = [0,12],
             )

#ali
plot_profile(data[:,0], 
             data[:,6], 
             'ali.png',
             title='Aliphatic (G,A,V,I,L,M,P)', 
             color='k',
             limits = [0,12],
             )



