
import numpy as np
import pickle
import torch
from pep_des.utils import *
import matplotlib.pyplot as plt
from pep_des.preprocessing.feature_calculation import extract_linear_features

# %% Get results

name="example_results"
names_exp = ['example_results']

objectives= np.load(f"./results_example/obj_{name}.npy")
sequences=np.load(f"./results_example/seq_{name}.npy")
objectives = np.delete(objectives,np.where(np.char.find(sequences,["XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"])==0),0)
sequences = np.delete(sequences,np.where(np.char.find(sequences,["XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"])==0),0)
reason=np.load(f"./results_example/reason_{name}.npy")

with open(r"./trained_models/example.pkl", "rb") as input_file:
    tt = pickle.load(input_file)
dl = tt["loader"]
x, y = dl.get_data()


sequences_explore_list = []
objectives_explore_list = []
for name in names_exp:
    sequences_explore_temp=np.load(f"./results_example/sequences_exp_{name}.npy")
    objectives_explore_temp=np.load(f"./results_example/objectives_exp_{name}.npy")
    obj_exp_temp = np.delete(objectives_explore_temp,np.isnan(objectives_explore_temp)[:,0],0)
    seq_exp_temp = np.delete(sequences_explore_temp,np.isnan(objectives_explore_temp)[:,0],0)
    obj_exp_temp = np.delete(objectives_explore_temp,np.where(np.char.find(sequences_explore_temp,["XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"])==0),0)
    seq_exp_temp = np.delete(sequences_explore_temp,np.where(np.char.find(sequences_explore_temp,["XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"])==0),0)
    
    sequences_explore_list.append(seq_exp_temp)
    objectives_explore_list.append(obj_exp_temp)
    
seq_exp = np.concatenate(sequences_explore_list)
obj_exp = np.concatenate(objectives_explore_list)  
    
int1 = interface_space(y)
int2 = interface_space(objectives)
int3 = interface_space(obj_exp)

obj_filtered_selected = well_spaced_points(int2, 25, int1).data
seq_filtered_selected = sequences[[np.where(int2 == obj_filtered_selected[i])[0][0] for i in range(len(obj_filtered_selected))]]

exploration_points = well_spaced_points(int3, 75, int1).data
exploration_sequences = seq_exp[[np.where(int3 == exploration_points[i])[0][0] for i in range(len(exploration_points))]]

# %% Check if prediction is correct

lin_features = extract_linear_features(seq_filtered_selected)
with torch.no_grad():
    pred = interface_space(tt["scalers"][1].inverse_transform(tt["model"](torch.tensor(tt["scalers"][0].transform(lin_features),dtype=torch.float32))))
    if any((np.round(pred,3)==np.round(obj_filtered_selected,3))[0] == False):
        print("FALSE")
        
# %% Plot results

plt.plot(np.exp(-obj_filtered_selected[:,0]),-obj_filtered_selected[:,1],"o", alpha=1, label="Pareto Front", markersize=6.5, zorder=20)
plt.plot(np.exp(-exploration_points[:,0]),-exploration_points[:,1],"^", alpha=1, label="Selected Exploration", markersize=5)
plt.plot(np.exp(-int1[:,0]),-int1[:,1],"o", label="Simulation data", markersize=2)

plt.legend()
plt.xlabel("$p_{int}^*$")
plt.ylabel(r"$B_2^*$")
plt.xscale('log')
plt.tight_layout()
plt.grid()

# %% Generate sequence file for next MD simulations

it1_seq = np.hstack((seq_filtered_selected, exploration_sequences))
it1_obj = np.vstack((obj_filtered_selected, exploration_points))

with open("new_sequences.txt", 'w') as file:
    for i, string in enumerate(it1_seq, start=1):
        file.write(f"Iter_2_{i}: {string}\n")






