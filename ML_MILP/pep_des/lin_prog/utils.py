"""
Author: Marco Bühler
"""

from pep_des.utils import amino_params
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def create_features(ohe, scaler) -> list[float]:
    """
    Create our features from the OHE matrix
    We also have to account for the relaxation problems, where multiple fractions of AA can
    occupy the same position
    """

    Features_counts = [(sum(ohe[:, i])) for i in range(20)]
    SHD = 0
    SPD = 0
    SND = 0
    peptide_length = int(len(ohe) / 20)
    for idx1 in range(peptide_length):
        eps1 = sum([ohe[idx1, i] * amino_params["eps_mpipi"][i] for i in range(20)])
        pos_1 = sum([ohe[idx1, i] * amino_params["pos_charges"][i] for i in range(20)])
        neg_1 = sum([ohe[idx1, i] * amino_params["neg_charges"][i] for i in range(20)])
        for idx2 in range(idx1 + 1, peptide_length):
            eps2 = sum([ohe[idx2, i] * amino_params["eps_mpipi"][i] for i in range(20)])
            pos_2 = sum([ohe[idx2, i] * amino_params["pos_charges"][i] for i in range(20)])
            neg_2 = sum([ohe[idx2, i] * amino_params["neg_charges"][i] for i in range(20)])
            SPD += (pos_1 + pos_2) * (idx2 - idx1) ** (-1)
            SND += (neg_1 + neg_2) * (idx2 - idx1) ** (-1)
            SHD += (eps1 + eps2) * (idx2 - idx1) ** (-1)

    SPND = 0
    for idx1 in range(peptide_length):
        pos_charge = sum([ohe[idx1, i] * amino_params["pos_charges"][i] for i in range(20)])
        for idx2 in range(30):
            if not idx1 == idx2:
                neg_charge = sum([ohe[idx2, i] * amino_params["neg_charges"][i] for i in range(20)])
                t = (pos_charge + neg_charge) * abs((idx2 - idx1)) ** (-1)
                SPND += t

    eps_seq = [sum([ohe[j, i] * amino_params["eps_mpipi"][i] for i in range(20)]) for j in range(peptide_length)]
    pos_seq = [sum([ohe[j, i] * amino_params["pos_charges"][i] for i in range(20)]) for j in range(peptide_length)]
    neg_seq = [sum([ohe[j, i] * amino_params["neg_charges"][i] for i in range(20)]) for j in range(peptide_length)]
    mw_seq = [sum([ohe[j, i] * amino_params["molecular_weight"][i] for i in range(20)]) for j in range(peptide_length)]
    A_yes_no = [sum([ohe[i, k] if k in [8, 12, 13] else 0 for k in range(20)]) for i in range(peptide_length)]

    distances = np.arange(-14.5, 15.5)
    DS_0 = sum(eps_seq[i] for i in range(peptide_length))
    DS_1 = sum(eps_seq[i] * distances[i] for i in range(peptide_length))
    DS_2 = sum(eps_seq[i] * distances[i] ** 2 for i in range(peptide_length))
    DS_3 = sum(eps_seq[i] * distances[i] ** 3 for i in range(peptide_length))
    DA_0 = sum(A_yes_no[i] for i in range(peptide_length))
    DA_1 = sum(A_yes_no[i] * distances[i] for i in range(peptide_length))
    DA_2 = sum(A_yes_no[i] * distances[i] ** 2 for i in range(peptide_length))
    DA_3 = sum(A_yes_no[i] * distances[i] ** 3 for i in range(peptide_length))
    DP_0 = sum(pos_seq[i] for i in range(peptide_length))
    DP_1 = sum(pos_seq[i] * distances[i] for i in range(peptide_length))
    DP_2 = sum(pos_seq[i] * distances[i] ** 2 for i in range(peptide_length))
    DP_3 = sum(pos_seq[i] * distances[i] ** 3 for i in range(peptide_length))
    DN_0 = sum(neg_seq[i] for i in range(peptide_length))
    DN_1 = sum(neg_seq[i] * distances[i] for i in range(peptide_length))
    DN_2 = sum(neg_seq[i] * distances[i] ** 2 for i in range(peptide_length))
    DN_3 = sum(neg_seq[i] * distances[i] ** 3 for i in range(peptide_length))
    DM_0 = sum(mw_seq[i] for i in range(peptide_length))
    DM_1 = sum(mw_seq[i] * distances[i] for i in range(peptide_length))
    DM_2 = sum(mw_seq[i] * distances[i] ** 2 for i in range(peptide_length))
    DM_3 = sum(mw_seq[i] * distances[i] ** 3 for i in range(peptide_length))
    features = [
        *Features_counts,
        SHD,
        SPD,
        SND,
        SPND,
        DS_0,
        DS_1,
        DS_2,
        DS_3,
        DA_0,
        DA_1,
        DA_2,
        DA_3,
        DP_0,
        DP_1,
        DP_2,
        DP_3,
        DN_0,
        DN_1,
        DN_2,
        DN_3,
        DM_0,
        DM_1,
        DM_2,
        DM_3,
    ]
    # Functionality for different preprocessing steps
    for i in range(len(features)):
        if scaler["x"].__class__ == StandardScaler().__class__:
            if scaler["x"].with_mean:
                features[i] -= scaler["x"].mean_[i]
            if scaler["x"].with_std:
                features[i] /= scaler["x"].scale_[i]
        elif scaler["x"].__class__ == MinMaxScaler().__class__:
            features[i] *= scaler["x"].scale_[i]
            features[i] += scaler["x"].min_[i]
        else:
            raise NotImplementedError
    return features
