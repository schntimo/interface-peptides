from pep_des.preprocessing import InterfaceLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from pep_des.model import Net, train_model
import torch
import matplotlib.pyplot as plt
from pep_des.utils import *
import matplotlib

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['font.size'] = 12  

torch.set_default_dtype(torch.float32)

# %% Inputs

dl = InterfaceLoader(path_results="./pep_des/data/LAF_1/results_LAF1_ini_it1.csv", delimiter=";")
seed = 42 

# %% Load data

x, y = dl.get_data()
X_train, X_holdout, y_train, y_holdout = train_test_split(x, y, test_size = 0.2, random_state=seed)
scaler_x = MinMaxScaler((-1,1)).fit(X_train)
X_train_scaled = scaler_x.transform(X_train)
X_holdout_scaled = scaler_x.transform(X_holdout)
scaler_y = MinMaxScaler((-1,1)).fit(y_train)
y_train_scaled = scaler_y.transform(y_train)
y_holdout_scaled = scaler_y.transform(y_holdout)
X_train_scaled_small, X_test, y_train_small, y_test = train_test_split(X_train_scaled, y_train_scaled, test_size = 0.2,random_state=seed)

# %% Optimize learning rate / weight decay

learnr = 5e-4     
weigth_d = 2e-3   

torch.manual_seed(123)
model = Net(len(X_train[0]),50,50,3)
optimizer = torch.optim.Adam(model.parameters(), lr=learnr, weight_decay=weigth_d)
mod_trained,hist_train,hist_val = train_model(X_train_scaled_small,y_train_small,model, epochs=10000, optimizer=optimizer, seq_model=True, verbose=True, loss=torch.nn.SmoothL1Loss())

plt.figure()
plt.plot(hist_train, label='training_loss')
plt.plot(hist_val, label='validation_loss')
plt.xlabel('epoch #')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 5))

with torch.no_grad():
    preds1 = mod_trained(torch.tensor(X_test, dtype=torch.float32)).detach()
    preds2 = mod_trained(torch.tensor(X_train_scaled_small, dtype=torch.float32)).detach()

main_title = f"Observed vs Predicted lr = {learnr} wd = {weigth_d}"
text = [r"$\Delta G_1$", r"$\Delta G_2$", r"$B_2^*$"]
sub_titles = [f"{text[i]} \n RMSE: {round(mean_squared_error(preds1[:, i], y_test[:, i]), 3)}, R2: {round(r2_score(y_test[:, i], preds1[:, i]), 3)}" for i in range(3)]

fig.suptitle(main_title, fontsize=14)

for i, ax in enumerate(axes):
    ax.scatter(y_test[:, i], preds1[:, i], label="Test Set")
    ax.scatter(y_train_small[:, i], preds2[:, i], alpha=0.2, label="Train Set")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_xticks(range(int(min_val), int(max_val) + 1,2))
    ax.set_yticks(range(int(min_val), int(max_val) + 1,2))
    ax.axline((1, 1), slope=1, color="black")
    ax.set_title(sub_titles[i], fontsize=10)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout to accommodate the main title
plt.show()

# %% Evaluate model performance

torch.manual_seed(seed)
model = Net(len(X_train[0]),50,50,3)
optimizer = torch.optim.Adam(model.parameters(), lr=learnr, weight_decay=weigth_d)
mod_trained,hist_train,hist_val = train_model(X_train_scaled,y_train_scaled,model, epochs=10000, optimizer=optimizer, seq_model=True, verbose=True)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.7))

with torch.no_grad():
    preds1 = mod_trained(torch.tensor(X_holdout_scaled, dtype=torch.float32)).detach()
    preds2 = mod_trained(torch.tensor(X_train_scaled, dtype=torch.float32)).detach()

main_title = "Observed vs Predicted"
r2 = r"$R^2$"
sub_titles = [f"{text[i]}" for i in range(3)]

for i, ax in enumerate(axes):
    ax.scatter(preds2[:, i], y_train_scaled[:, i], alpha=0.8, label="Train", color='grey')
    ax.scatter(preds1[:, i], y_holdout_scaled[:, i], label="Test", color='magenta', edgecolor='k')
    ax.set_xlabel(f'Predicted', labelpad=-1)
    ax.set_ylabel(f'True', labelpad=-10)
    min_val = min(ax.get_xlim()[0], ax.get_ylim()[0])
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.text(0.5, 0.1, f'MSE: {round(mean_squared_error(preds1[:, i], y_holdout_scaled[:, i]), 13):.1e} \n{r2}: {round(r2_score(y_holdout_scaled[:, i], preds1[:, i]), 3)}', transform=ax.transAxes)
    ax.set_xticks(range(int(min_val), int(max_val) + 1,2))
    ax.set_yticks(range(int(min_val), int(max_val) + 1,2))
    ax.axline((1, 1), slope=1, color="black", zorder=-1)
    ax.set_title(sub_titles[i], wrap=True)
    ax.legend(frameon=False)

plt.tight_layout()  # Adjust the layout to accommodate the main title
plt.savefig("Nodelim_all_train_obspred_l1.png", dpi=400)
plt.show()

# %% Train production model

torch.manual_seed(seed)
model = Net(len(X_train[0]),50,50,3)
optimizer = torch.optim.Adam(model.parameters(), lr=learnr, weight_decay=weigth_d)
mod_trained,hist_train,hist_val = train_model(scaler_x.transform(x),scaler_y.transform(y),model, epochs=10000, optimizer=optimizer, seq_model=True, verbose=True, loss=torch.nn.SmoothL1Loss())

from torch.nn.utils import prune
parameters_to_prune = ((mod_trained.f1, "bias"), (mod_trained.f1, "weight"),(mod_trained.f2, "weight"), (mod_trained.f2, "bias"),(mod_trained.input, "weight"), (mod_trained.input, "bias"))
prune.global_unstructured(
    parameters_to_prune, pruning_method=ThresholdPruning, threshold=0.001
)
print(sum(torch.nn.utils.parameters_to_vector(mod_trained.buffers()) == 0))

export_model_interface(mod_trained, scaler_x, scaler_y,dl, path="trained_models/example.pkl")


