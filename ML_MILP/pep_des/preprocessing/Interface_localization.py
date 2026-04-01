"""
Author: Marco Bühler
"""

import numpy as np
import csv
from pep_des.utils import get_parameters
from pep_des.preprocessing.feature_calculation import extract_linear_features, get_seq_num


class InterfaceLoader:
    def __init__(self, path_results, delimiter) -> None:
        self.path_results = path_results
        self.kT_kcalmol = 8.314 * 300 / 4184
        self.C = 20
        self.delimiter = delimiter
        self._load_data()
        self._load_data_raw_with_std()

    def _load_data(self) -> None:
        with open(self.path_results, "r") as file:
            reader = csv.reader(file, delimiter=self.delimiter)
            next(reader)
            data = [row for row in reader]
        sequences = [row[0] for row in data]
        observations = np.array([[float(row[1]), float(row[3]), float(row[5])] for row in data])
        flags_combined = np.array([np.sum(np.array(row[7:], dtype=float)) for row in data])
        indices_errors = np.where(flags_combined != 0)[0]
        self.sequences_clean = np.delete(sequences, indices_errors)
        values_raw = np.array([row[1:5] for row in data], dtype=float)
        self.values_clean = np.delete(values_raw, indices_errors, axis=0)
        self.observations_clean = np.delete(observations, indices_errors, axis=0)

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        features = extract_linear_features(self.sequences_clean)
        seq_num = get_seq_num(self.sequences_clean)
        distance = np.arange(-14.5, 15.5)
        amino_params = get_parameters()
        for i, seq in enumerate(seq_num):
            condition_array = np.array([amino_params["molecular_weight"][res] for res in seq])
            if sum(condition_array * distance**3) < 0:
                self.sequences_clean[i] = self.sequences_clean[i][::-1]
        second_virial = (
            (-1) * np.sign(self.observations_clean[:, 2]) * np.log10(1 + np.abs(self.observations_clean[:, 2] / self.C))
        )  # MINIMIZE

        return (
            features,
            np.array(
                [
                    self.observations_clean[:, 0] / self.kT_kcalmol,
                    -self.observations_clean[:, 1] / self.kT_kcalmol,
                    second_virial,
                ]
            ).T,
        )
    
    def _load_data_raw_with_std(self) -> None:
        with open(self.path_results, "r") as file:
            reader = csv.reader(file, delimiter=self.delimiter)
            next(reader)
            data = [row for row in reader]
        sequences = [row[0] for row in data]
        observations_raw = np.array([[float(row[1]), float(row[3]), float(row[5])] for row in data])
        observations_raw_std = np.array([[float(row[2]), float(row[4]), float(row[6])] for row in data])
        flags_combined = np.array([np.sum(np.array(row[7:], dtype=float)) for row in data])
        indices_errors = np.where(flags_combined != 0)[0]
        self.sequences_clean = np.delete(sequences, indices_errors)
        values_raw = np.array([row[1:5] for row in data], dtype=float)
        self.values_clean = np.delete(values_raw, indices_errors, axis=0)
        self.observations_raw_clean = np.delete(observations_raw, indices_errors, axis=0)
        self.observations_raw_std_clean = np.delete(observations_raw_std, indices_errors, axis=0)

    def get_data_raw_with_std(self) -> tuple[np.ndarray, np.ndarray]:
        features = extract_linear_features(self.sequences_clean)
        seq_num = get_seq_num(self.sequences_clean)
        distance = np.arange(-14.5, 15.5)
        amino_params = get_parameters()
        for i, seq in enumerate(seq_num):
            condition_array = np.array([amino_params["molecular_weight"][res] for res in seq])
            if sum(condition_array * distance**3) < 0:
                self.sequences_clean[i] = self.sequences_clean[i][::-1]
        # second_virial = (
        #     (-1) * np.sign(self.observations_clean[:, 2]) * np.log10(1 + np.abs(self.observations_clean[:, 2] / self.C))
        # )  # MINIMIZE

        return (
            features,
            np.array(
                [
                    self.observations_raw_clean[:, 0] / self.kT_kcalmol,  # in kbT
                    -self.observations_raw_clean[:, 1] / self.kT_kcalmol,
                    self.observations_raw_clean[:, 2],  # in nm3
                    
                    self.observations_raw_std_clean[:, 0] / self.kT_kcalmol,  # in kbT
                    self.observations_raw_std_clean[:, 1] / self.kT_kcalmol,
                    self.observations_raw_std_clean[:, 2],  # in nm3
                    
                ]
            ).T,
        )
