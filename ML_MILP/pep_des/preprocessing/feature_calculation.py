"""
Author: Marco Bühler

Contains multiple functions to calculate different feature sets
"""

import numpy as np
from pep_des.utils import amino_params

distance = np.arange(-14.5, 15.5)


def get_composition(seq_num):
    return [[seq.count(i) for i in range(20)] for seq in seq_num]


def get_scd(seq_num):
    """added later, for post-optimization analysis only"""
    SCD = []
    for seq in seq_num:
        SCD_ = 0
        for idx1, res1 in enumerate(seq):
            for idx2, res2 in enumerate(seq[idx1 + 1 :], start=idx1 + 1):
                inv = (idx2 - idx1) ** (0.5)
                SCD_ += (amino_params["charges"][res1] * amino_params["charges"][res2]) * inv / len(seq)
        SCD.append(SCD_)
    return SCD


def get_shannon_entropy(seq_num):
    """Return the shannon entropy"""
    comp = get_composition(seq_num)
    shannon = []
    for i in range(len(seq_num)):
        shannon.append(-1 * sum([(c * np.log2(c)) if c != 0 else 0 for c in comp[i]]))
    return shannon


def flip_sequences(seq_num, condition="eps_mpipi", exponent=3):
    "Flips the sequences to be invariant to flips during training of the ML algorithm"
    assert condition in [
        "eps_mpipi",
        "molecular_weight",
    ], f"condition has to be either 'eps_mpipi' or 'molecular_weight' got {condition} "

    for i, seq in enumerate(seq_num):
        condition_array = np.array([amino_params[condition][res] for res in seq])
        if sum(condition_array * distance**exponent) < 0:
            seq_num[i] = seq_num[i][::-1]
    return seq_num


def get_decorators(seq_num):
    """Returns SND, SPD, SHD, SPND"""
    (SND, SPD, SHD, SPND) = ([] for i in range(4))
    for seq in seq_num:
        SND_, SPD_, SHD_ = 0, 0, 0
        for idx1, res1 in enumerate(seq):
            for idx2, res2 in enumerate(seq[idx1 + 1 :], start=idx1 + 1):
                inv = (idx2 - idx1) ** (-1)
                SHD_ += (amino_params["eps_mpipi"][res1] + amino_params["eps_mpipi"][res2]) * inv
                SPD_ += (amino_params["pos_charges"][res1] + amino_params["pos_charges"][res2]) * inv
                SND_ += (amino_params["neg_charges"][res1] + amino_params["neg_charges"][res2]) * inv
        SND.append(SND_)
        SPD.append(SPD_)
        SHD.append(SHD_)

        x = 0
        for idx1, res1 in enumerate(seq):
            for idx2, res2 in enumerate(seq):
                if not idx1 == idx2:
                    t = (amino_params["pos_charges"][res1] + amino_params["neg_charges"][res2]) * abs(idx2 - idx1) ** (
                        -1
                    )
                    x += t
        SPND.append(x)
    return SND, SPD, SHD, SPND

  
        
def get_eps_moment(seq_num):
    (DS_0, DS_1, DS_2, DS_3) = ([] for i in range(4))
    for seq in seq_num:
        eps_seq = np.array([amino_params["eps_mpipi"][res] for res in seq])

        DS_0.append(sum(eps_seq))
        DS_1.append(sum(eps_seq * distance))
        DS_2.append(sum(eps_seq * distance**2))
        DS_3.append(sum(eps_seq * distance**3))
    return DS_0, DS_1, DS_2, DS_3


def get_aro_moment(seq_num):
    (DA_0, DA_1, DA_2, DA_3) = ([] for i in range(4))
    for seq in seq_num:
        A_yes_no = np.array([(res == 8) | (res == 12) | (res == 13) for res in seq])
        DA_0.append(sum(A_yes_no))
        DA_1.append(sum(A_yes_no * distance))
        DA_2.append(sum(A_yes_no * distance**2))
        DA_3.append(sum(A_yes_no * distance**3))
    return DA_0, DA_1, DA_2, DA_3


def get_pos_moment(seq_num):
    (DP_0, DP_1, DP_2, DP_3) = ([] for i in range(4))
    for seq in seq_num:
        pos_seq = np.array([amino_params["pos_charges"][res] for res in seq])
        DP_0.append(sum(pos_seq))
        DP_1.append(sum(pos_seq * distance))
        DP_2.append(sum(pos_seq * distance**2))
        DP_3.append(sum(pos_seq * distance**3))
    return DP_0, DP_1, DP_2, DP_3


def get_neg_moment(seq_num):
    (DN_0, DN_1, DN_2, DN_3) = ([] for i in range(4))
    for seq in seq_num:
        neg_seq = np.array([amino_params["neg_charges"][res] for res in seq])
        DN_0.append(sum(neg_seq))
        DN_1.append(sum(neg_seq * distance))
        DN_2.append(sum(neg_seq * distance**2))
        DN_3.append(sum(neg_seq * distance**3))
    return DN_0, DN_1, DN_2, DN_3


def get_mw_moment(seq_num):
    (DM_0, DM_1, DM_2, DM_3) = ([] for i in range(4))
    for seq in seq_num:
        mw_seq = np.array([amino_params["molecular_weight"][res] for res in seq])
        DM_0.append(sum(mw_seq))
        DM_1.append(sum(mw_seq * distance))
        DM_2.append(sum(mw_seq * distance**2))
        DM_3.append(sum(mw_seq * distance**3))
    return DM_0, DM_1, DM_2, DM_3


def get_seq_num(inp_seq):
    """Returns the ordinal encoding of the amino acid sequence"""
    return [[amino_params["letters"][letter] - 1 for letter in string] for string in inp_seq]


def extract_linear_features(inp_seq: list[str] = None, flip_condition: str = "molecular_weight", exponent: int = 3):
    """Main function to extract the data needed for the machine learning pipeline"""
    assert (
        type(inp_seq) is list or np.ndarray
    ), "Provide a list of input sequences, or a single sequence in the format [str]"
    seq_num = get_seq_num(inp_seq)
    seq_num = flip_sequences(seq_num, flip_condition, exponent)
    features_counts = get_composition(seq_num)
    SND, SPD, SHD, SPND = get_decorators(seq_num)
    DS_0, DS_1, DS_2, DS_3 = get_eps_moment(seq_num)
    DA_0, DA_1, DA_2, DA_3 = get_aro_moment(seq_num)
    DP_0, DP_1, DP_2, DP_3 = get_pos_moment(seq_num)
    DN_0, DN_1, DN_2, DN_3 = get_neg_moment(seq_num)
    DM_0, DM_1, DM_2, DM_3 = get_mw_moment(seq_num)
    features = np.array(
        [
            *np.array(features_counts).T,
            SHD,  # index: 20
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
    ).transpose()
    return features
