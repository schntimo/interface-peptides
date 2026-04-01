"""
Author: Marco Bühler
"""

import numpy as np
import json
from numpy.ma import vstack
from numpy.random import randint
from scipy.spatial.distance import cdist
import pickle
import torch
import pathlib
from torch.nn.utils import prune

filepath = str(pathlib.Path(__file__).resolve().parent)


def get_parameters() -> dict:
    """Loads a json file with all the amino acid parameters"""
    with open(filepath + "/preprocessing/amino_acid_params.json") as f:
        file = json.load(f)
    return file

amino_params = get_parameters()


def get_seq_from_pyo(ohe):
    return get_seq_from_mat(convert_pyo_to_np(ohe))


def get_best_seq(dl, y_scaler, bound=None, variable=0):
    """Get the best sequence in the training data"""
    _, y = dl.get_data(linear=True)
    y_scaled = y_scaler.transform(y[:, 2:])
    seq = dl.sequences_clean
    if bound:
        y_selected = y_scaled[y_scaled[:, variable] < bound]
        seq_selected = seq[y_scaled[:, variable] < bound]
    else:
        y_selected = y_scaled
        seq_selected = seq
    return seq_selected[np.argmin(y_selected[:, variable])]

def get_theoretical_bounds():
    # Theoretical lower bounds
    SHD = [-233.73047289, -0.026056388]
    SPD = [0, 1.79699228e02]
    SND = [-1.79699228e02, 0]
    SPND = [-1.79699228e02, 1.79699228e02]

    DS_0 = [-3.90202800e01, -4.35000000e-03]
    DS_1 = [-1.46309737e02, 1.46309738e02]
    DS_2 = [-2.92326931e03, -3.25887500e-01]
    DS_3 = [-1.64232680e04, 1.64232680e04]

    DA_0 = [0, 3.00000000e01]
    DA_1 = [-1.12500000e02, 1.12500000e02]
    DA_2 = [0, 2.24750000e03]
    DA_3 = [-1.26281250e04, 1.26281250e04]

    DP_0 = [0, 3.0000e01]
    DP_1 = [-1.12500000e02, 1.12500000e02]
    DP_2 = [0, 2.2475e03]
    DP_3 = [-1.26281250e04, 1.26281250e04]

    DN_0 = [-3.0000000e01, 0]
    DN_1 = [-1.1250000e02, 1.12500000e02]
    DN_2 = [-2.2475000e03, 0]
    DN_3 = [-1.2628125e04, 1.26281250e04]

    DM_0 = [1.71150000e03, 5.58600000e03]
    DM_1 = [-1.45293750e04, 1.45293750e04]
    DM_2 = [1.28219875e05, 4.18484500e05]
    DM_3 = [0, 1.63092234e06]  # Due to symmetry constraint lb = 0
    theoretical_lb = np.array(
        [
            *[0 for _ in range(20)],
            SHD[0],
            SPD[0],
            SND[0],
            SPND[0],
            DS_0[0],
            DS_1[0],
            DS_2[0],
            DS_3[0],
            DA_0[0],
            DA_1[0],
            DA_2[0],
            DA_3[0],
            DP_0[0],
            DP_1[0],
            DP_2[0],
            DP_3[0],
            DN_0[0],
            DN_1[0],
            DN_2[0],
            DN_3[0],
            DM_0[0],
            DM_1[0],
            DM_2[0],
            DM_3[0],
        ]
    )
    theoretical_ub = np.array(
        [
            *[30 for _ in range(20)],
            SHD[1],
            SPD[1],
            SND[1],
            SPND[1],
            DS_0[1],
            DS_1[1],
            DS_2[1],
            DS_3[1],
            DA_0[1],
            DA_1[1],
            DA_2[1],
            DA_3[1],
            DP_0[1],
            DP_1[1],
            DP_2[1],
            DP_3[1],
            DN_0[1],
            DN_1[1],
            DN_2[1],
            DN_3[1],
            DM_0[1],
            DM_1[1],
            DM_2[1],
            DM_3[1],
        ]
    )
    return theoretical_lb, theoretical_ub


def export_model_interface(trained_model, x_scaler, y_scaler, dl, path="pyomo_predictor.pkl"):
    """Exports the things needed to run the pyomo model in a single pickle file"""
    theoretical_lb, theoretical_ub = get_theoretical_bounds()
    file = {}
    lb = x_scaler.transform(theoretical_lb.reshape(1, -1)).ravel()
    ub = x_scaler.transform(theoretical_ub.reshape(1, -1)).ravel()
    file["model"] = trained_model
    file["scalers"] = [x_scaler, y_scaler]
    file["bounds"] = [lb, ub]
    file["loader"] = dl
    with open(path, "wb") as output_file:
        pickle.dump(file, output_file)


def convert_pyo_to_np(mat) -> np.ndarray:
    """Convert a pyomo one hot matrix into a numpy mat"""
    pep_len = int(len(mat) / 20)
    np_mat = np.zeros((pep_len, 20), dtype=int)
    for i in range(pep_len):
        for j in range(20):
            np_mat[i, j] = mat[i, j].value
    return np_mat

def interface_space(x):
    obj = np.log(np.exp(-x[:, 0]) + 100 * np.exp(-x[:, 1]))
    return np.hstack((obj.reshape(-1, 1), x[:, 2:]))


def get_seq_from_mat(mat: np.ndarray) -> list[str]:
    """Get the sequence corresponding to the one hot matrix"""
    vec = np.argmax(mat, 1)
    amino_acid_table = list(amino_params["letters"].keys())
    return [amino_acid_table[i] for i in vec]


def create_ohe_from_seq(seq: list[str]) -> np.ndarray:
    """Create a one hot matrix from a sequence"""
    amino_acid_table = amino_params["letters"]
    pep_vec = [amino_acid_table[i] - 1 for i in seq]
    mat = np.zeros((len(seq), 20))
    for i in range(len(seq)):
        mat[i, pep_vec[i]] = int(1)
    return mat


# adapted from https://stackoverflow.com/a/50264422
def well_spaced_points(points: np.ndarray, num_points: int, pre_picked_points: np.ndarray = None) -> np.ndarray:
    """
    Pick `num_points` well-spaced points from `points` array.

    :param points: An m x n array of m n-dimensional points.
    :param num_points: The number of points to pick.
    :rtype: ndarray
    :return: A num_points x n array of points from the original array.
    """
    if pre_picked_points is not None:
        current_point_index = randint(0, num_points)
        picked_points = np.array([points[current_point_index]])
        picked_points = vstack((picked_points, pre_picked_points))
        remaining_points = vstack((points[:current_point_index], points[current_point_index + 1 :]))
        num_points = num_points + len(picked_points)
    # pick a random point
    else:
        current_point_index = randint(0, num_points)
        picked_points = np.array([points[current_point_index]])
        remaining_points = vstack((points[:current_point_index], points[current_point_index + 1 :]))
    # while there are more points to pick
    while picked_points.shape[0] < num_points:
        # find the furthest point to the current point
        distance_pk_rmn = cdist(picked_points, remaining_points, metric="cityblock")
        min_distance_pk = distance_pk_rmn.min(axis=0)
        i_furthest = np.argmax(min_distance_pk)
        # add it to picked points and remove it from remaining
        picked_points = vstack((picked_points, remaining_points[i_furthest]))
        remaining_points = vstack((remaining_points[:i_furthest], remaining_points[i_furthest + 1 :]))
    if pre_picked_points is not None:
        picked_points = np.delete(picked_points, slice(0, len(pre_picked_points)), 0)
        picked_points = np.delete(picked_points, 0, 0)
        return picked_points
    else:
        return picked_points


# taken from https://stackoverflow.com/a/40239615
def is_pareto_efficient(costs: np.ndarray, return_mask: bool = False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


class ThresholdPruning(prune.BasePruningMethod):
    """Neural Network Pruning"""

    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
