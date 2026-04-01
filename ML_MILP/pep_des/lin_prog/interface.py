"""
Author: Marco Bühler
"""

import tempfile
import torch, torch.onnx
import numpy as np
import pyomo.environ as pe
from omlt import OmltBlock
from omlt.neuralnet import ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds, load_onnx_neural_network_with_bounds
from pep_des.utils import amino_params, create_ohe_from_seq, get_seq_from_pyo
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .utils import create_features
from pep_des.preprocessing import extract_linear_features


class PyomoInterface:
    def __init__(
        self,
        n_steps: int,
        n_threads: int,
        pep_length: int = 30,
    ) -> None:
        self.n_steps = n_steps
        self.n_threads = n_threads
        self.pep_length = pep_length

    def create_new_NN_formulation(self, mod_dict: dict = None) -> None:
        """Takes the pickled trained model object and extracts the formulation"""
        scaler_X, scaler_y = mod_dict["scalers"]
        self.inp_shape = scaler_X.n_features_in_
        
        x_dummy = torch.tensor(mod_dict["bounds"][0], dtype=torch.float32).reshape(1, -1)
        self.nn_model = mod_dict["model"]  # get the model from the dict
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            torch.onnx.export(
                mod_dict["model"],
                x_dummy,
                f,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )
            input_bounds = {}
            for i in range(self.inp_shape):
                input_bounds[(i)] = (
                    mod_dict["bounds"][0][i],
                    mod_dict["bounds"][1][i],
                )
            write_onnx_model_with_bounds(f.name, None, input_bounds)  # Create an onnx model
            network_definition = load_onnx_neural_network_with_bounds(f.name)
            
        self.formulation = ReluBigMFormulation(network_definition)
        self.scaler = {"x": scaler_X, "y": scaler_y}  # Scalers for further use
        self.dl = mod_dict["loader"]  # The dataloader for further use

    def _create_pyo_model(self) -> pe.ConcreteModel:
       
        # Initialize a pyomo env
        m = pe.ConcreteModel()
        m.predictor = OmltBlock()
        m.predictor.build_formulation(self.formulation)

        # Create the Range sets needed, in pyomo the last int is INCLUSIVE
        m.amino_acids = pe.RangeSet(0, 19)
        m.position = pe.RangeSet(0, self.pep_length - 1)

        # The OHE matrix with the binary variables
        m.ohe_param = m.position * m.amino_acids
        m.ohe = pe.Var(m.ohe_param, initialize=0, within=pe.Binary)
        features = create_features(m.ohe, self.scaler)

        # Aggrescan
        aggr_values = [
            sum([m.ohe[j, i] * amino_params["aggrescan_weights"][i] for i in range(20)]) for j in range(self.pep_length)
        ]

        end_caps = [(-1.412 + -1.836) / 2, (-0.931 + -1.24) / 2]
        aggr_values.insert(0, end_caps[0])
        aggr_values.append(end_caps[1])
        hst = -0.02
        a4v = []
        a4v.append(sum(aggr_values[i] for i in range(5)) / 5 - m.ohe[0, 17] * 7)
        for i in range(self.pep_length - 2):
            a4v.append(sum(aggr_values[i + j] for j in range(5)) / 5 - m.ohe[i + 1, 17] * 7)
        a4v.append(sum(aggr_values[-j] for j in range(1, 6)) / 5 - m.ohe[self.pep_length - 1, 17] * 7)

        aggregation_windows = [[a4v[i + j] for j in range(5)] for i in range(self.pep_length - 4)]
        m.window_slider = pe.RangeSet(0, self.pep_length - 5) # remember: list inclusive
        m.window_size = pe.RangeSet(0, 4)
        m.ismin_mat = pe.Var(m.window_slider * m.window_size, within=pe.Binary)
        m.aggr_min_vec = pe.Var(m.window_slider, within=pe.Reals, bounds=(-10, 10))
        m.min_constraint, m.c1_list, m.c2_list, m.c3_list = (pe.ConstraintList() for _ in range(4))
        m.M = pe.Param(initialize=10.0, mutable=False)

        for i in range(self.pep_length - 4):
            m.min_constraint.add(expr=sum(m.ismin_mat[i, t] for t in m.window_size) == 4)
            for j in range(5):
                m.c1_list.add(m.aggr_min_vec[i] >= aggregation_windows[i][j] - m.M * m.ismin_mat[i, j])
                m.c2_list.add(m.aggr_min_vec[i] <= aggregation_windows[i][j])

            m.c3_list.add(m.aggr_min_vec[i] <= hst)

    
        # Amino acid constraint
        m.amino_acid_limit = pe.ConstraintList()
        for i in range(self.pep_length):
            m.amino_acid_limit.add(sum(m.ohe[i, a] for a in m.amino_acids) == 1)

        # Connect the inputs
        def connect_inputs(m, i):
            return features[i] == m.predictor.inputs[i]

        m.input_constr = pe.Constraint(range(self.inp_shape), rule=connect_inputs)

        # Symmetry constraint to limit one feature to pos values only
        constraint_feature = features[-1]
        if self.scaler["x"].__class__ == StandardScaler().__class__:
            if self.scaler["x"].with_std:
                constraint_feature *= self.scaler["x"].scale_[-1]
            if self.scaler["x"].with_mean:
                constraint_feature += self.scaler["x"].mean_[-1]
        elif self.scaler["x"].__class__ == MinMaxScaler().__class__:
            constraint_feature -= self.scaler["x"].min_[-1]
            constraint_feature /= self.scaler["x"].scale_[-1]
        else:
            raise NotImplementedError

        m.mw_constraint = pe.Constraint(expr=constraint_feature >= 0)
        return m

    def _inverse_scale_y(self, x1, x2, x3):
        """Inverse scale the predictions"""
        return self.scaler["y"].inverse_transform(np.array([x1, x2, x3]).reshape(1, -1))[0]

    def _check_prediction(self) -> int:
        """
        Checks the obtained values with the full neural network

        Returns
        -------
        0: If everything is ok
        1: If the objectives do not correspond to the NN predictions
        2: If we are outside the convex envelope for the objective
        """
        seq = get_seq_from_pyo(self.m.ohe)
        control_feat = self.scaler["x"].transform(extract_linear_features([seq]).reshape(1, -1))
        with torch.no_grad():
            preds = self.nn_model(torch.tensor(control_feat, dtype=torch.float32)).numpy()
        x1 = pe.value(self.m.predictor.outputs[0])
        x2 = pe.value(self.m.predictor.outputs[1])
        x3 = pe.value(self.m.predictor.outputs[2])
        print(preds)
        print(x1, x2, x3)
        print(seq)
        if (
            round(float(preds[0][0]), 3) != round(x1, 3)
            or round(float(preds[0][1]), 3) != round(x2, 3)
            or round(float(preds[0][2]), 3) != round(x3, 3)
        ):
            print("=" * 40)
            print("Prediction is Wrong")
            print("=" * 40)
            return 1
        elif x1 > self.s_a_vec[-1] or x2 > self.s_b_vec[-1]:  # Outside the convex envelope
            print("=" * 40)
            print("Outside the linearisation")
            print("=" * 40)
            return 2
        else:  # All good
            return 0

    def _init_pyomo(self) -> None:
        self.m = self._create_pyo_model()
        self.solver = pe.SolverFactory("gurobi", io_format="python")
        self.solver.options["threads"] = self.n_threads
        self.solver.options["Presolve"] = 2
        self.solver.options["PreSparsify"] = 2

        # Add the convex envelope for log(exp(-A)+100*exp(-B))
        _, y = self.dl.get_data()
        a_min, a_max = y[:, 0].min(), y[:, 0].max()
        b_min, b_max = y[:, 1].min(), y[:, 1].max() # potentially adapt if convex envelope error occurs
        self.s_a_vec = np.linspace(a_min, a_max, 30)
        self.s_b_vec = np.linspace(b_min, b_max, 30)
        self.m.convex_obj = pe.Var(initialize=0.1, within=pe.Reals)

        # We need to undo the scaling for the objective
        y1 = self.m.predictor.outputs[0]
        y2 = self.m.predictor.outputs[1]
        if self.scaler["y"].__class__ == StandardScaler().__class__:
            if self.scaler["y"].with_std:
                y1 *= self.scaler["y"].scale_[0]
                y2 *= self.scaler["y"].scale_[1]
            if self.scaler["y"].with_mean:
                y1 += self.scaler["y"].mean_[0]
                y2 += self.scaler["y"].mean_[1]
        elif self.scaler["y"].__class__ == MinMaxScaler().__class__:
            y1 -= self.scaler["y"].min_[0]
            y1 /= self.scaler["y"].scale_[0]
            y2 -= self.scaler["y"].min_[1]
            y2 /= self.scaler["y"].scale_[1]
        else:
            raise NotImplementedError

        self.m.convex_envelope_constraint = pe.ConstraintList()
        for s_a in self.s_a_vec:
            for s_b in self.s_b_vec:
                self.m.convex_envelope_constraint.add(
                    expr=self.m.convex_obj
                    >= np.log(np.exp(-s_a) + 100 * np.exp(-s_b))
                    + -np.exp(-s_a) / (np.exp(-s_a) + 100 * np.exp(-s_b)) * (y1 - s_a)
                    + -100 * np.exp(-s_b) / (np.exp(-s_a) + 100 * np.exp(-s_b)) * (y2 - s_b)
                )

    def get_bounds(self) -> list:
        """Get the bounds and steps for the epsilon constraint method"""
        self._init_pyomo()
        self.m.O_f1 = pe.Objective(expr=self.m.convex_obj, sense=pe.minimize)
        self.m.O_f2 = pe.Objective(expr=self.m.predictor.outputs[2], sense=pe.minimize)

        self.m.O_f1.activate()
        self.m.O_f2.deactivate()
        self.solver.solve(self.m, tee=True)
        seq = get_seq_from_pyo(self.m.ohe)
        print(seq)
        assert self._check_prediction() == 0, "Bound Prediction is Wrong"

        f2_max = pe.value(self.m.predictor.outputs[2])

        self.m.O_f1.deactivate()
        self.m.O_f2.activate()
        self.solver.solve(self.m, tee=True)
        seq = get_seq_from_pyo(self.m.ohe)
        print(seq)
        assert self._check_prediction() == 0, "Bound Prediction is Wrong"

        f2_min = pe.value(self.m.predictor.outputs[2])

        steps = np.linspace(f2_min, f2_max, self.n_steps)
        print(steps)
        self.m.del_component(self.m.O_f1)
        self.m.del_component(self.m.O_f2)

        return steps

    def _get_onnx_output(self):
        """Returns the individual objectives"""
        x1 = pe.value(self.m.predictor.outputs[0])
        x2 = pe.value(self.m.predictor.outputs[1])
        x3 = pe.value(self.m.predictor.outputs[2])
        return x1, x2, x3

    def _check_failed(self):
        """If check failed, return nan and X-Seq"""
        return np.nan, np.nan, np.nan, ["X"] * self.pep_length

    def _solve_and_check(self, warmstart=True):
        """Solves the model and runs checks"""
        try:
            res = self.solver.solve(self.m, tee=True, warmstart=warmstart)
            x1, x2, x3 = self._get_onnx_output()
            seq = get_seq_from_pyo(self.m.ohe)
            if res.solver.termination_condition == "infeasibleOrUnbounded":
                "When initial solution is infeasible"
                x1, x2, x3, seq = self._check_failed()
                reason = 3
        except:
            x1, x2, x3, seq = self._check_failed()
            reason = 3
        if not np.isnan(x1):
            reason = self._check_prediction()
            if reason > 0:
                x1, x2, x3, seq = self._check_failed()
        y_unscaled = self._inverse_scale_y(x1, x2, x3)

        return y_unscaled, seq, reason

    def exploit_pyomo(self, bound: float) -> tuple[str, float, float]:
        """Run the epsilon constraint exploitation"""
        try:
            self.m.Obj.activate()
        except:
            self.m.eps = pe.Param(mutable=True, within=pe.Reals)
            self.m.C_e = pe.Constraint(expr=self.m.predictor.outputs[2] <= self.m.eps)
            self.m.Obj = pe.Objective(expr=self.m.convex_obj, sense=pe.minimize)
        self.m.eps = bound
        self.solver.options["MIPFocus"] = 2
        self.solver.options["NoRelHeurTime"] = 0
        y_unscaled, seq, reason = self._solve_and_check()
        return seq, y_unscaled[0], y_unscaled[1], y_unscaled[2], reason

    def explore_pyomo(
        self,
        positions: list[int],
    ) -> tuple[str, float, float, float, int]:
        self._init_pyomo()
        # Get a random amino acid for the positions
        amino_acids = np.random.choice(20, len(positions), replace=True).tolist()
        self.m.amino_constr_list = pe.ConstraintList()

        for i, pos in enumerate(positions):
            self.m.amino_constr_list.add(self.m.ohe[pos, amino_acids[i]] == 1)

        """Add the objective"""
        weight = np.random.rand(1)
        self.m.Obj = pe.Objective(
            expr=weight * self.m.convex_obj + (1 - weight) * self.m.predictor.outputs[2],
            sense=pe.minimize,
        )
        y_unscaled, seq, reason = self._solve_and_check(warmstart=False)
        return (seq, y_unscaled[0], y_unscaled[1], y_unscaled[2], weight, positions)



















