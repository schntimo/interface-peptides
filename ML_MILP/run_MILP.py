"""
Author: Marco Bühler
"""

from pep_des.lin_prog.interface import PyomoInterface
from pep_des.utils import *
import pickle
import argparse
from joblib import Parallel, delayed
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--n_steps")
parser.add_argument("--n_threads")
parser.add_argument("--output_name")
parser.add_argument("--parallel")
parser.add_argument("--peptide_length")

feature_parser = parser.add_mutually_exclusive_group(required=False)
feature_parser.add_argument("--explore", dest="explore", action="store_true")
feature_parser.add_argument("--no-explore", dest="explore", action="store_false")
parser.set_defaults(feature=True)
args = parser.parse_args()

explore = args.explore
n_steps = int(args.n_steps)
n_threads = int(args.n_threads)
out_name = str(args.output_name)
parallel = bool(int(args.parallel))
peptide_length = int(args.peptide_length) if args.peptide_length else 30

if not explore:
    print("EXPLOITATION")
    lp = PyomoInterface(n_steps=n_steps, n_threads=n_threads, pep_length=peptide_length)
    print("SOLVING WITH NEURAL NETWORKS")
    with open(r"trained_models/example.pkl", "rb") as input_file:
        model = pickle.load(input_file)
    _ = lp.create_new_NN_formulation(model)
    start_time = time.time()
    steps = lp.get_bounds()
    seq_arr = []
    obj_arr = []
    reason_arr = []
    for i in steps:
        temp_seq, temp_x1, temp_x2, temp_x3, reason_temp = lp.exploit_pyomo(i)
        seq_arr.append("".join(temp_seq))
        obj_arr.append((temp_x1, temp_x2, temp_x3))
        reason_arr.append(reason_temp)
    print(seq_arr)
    print(obj_arr)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"ELAPSED TIME: {elapsed_time}")

    np.save(f"./results_example/seq_{out_name}.npy", seq_arr)
    np.save(f"./results_example/obj_{out_name}.npy", obj_arr)
    np.save(f"./results_example/reason_{out_name}.npy", reason_arr)

else:
    print("EXPLORATION")
    lp = PyomoInterface(
        n_steps=n_steps,
        n_threads=n_threads,
    )
    with open(r"trained_models/example.pkl", "rb") as input_file:
        model = pickle.load(input_file)
    _ = lp.create_new_NN_formulation(model)
    n_mod = {}
    for i in range(n_steps):
        n_mod[i] = PyomoInterface(
            n_steps=n_steps,
            n_threads=6,
        )

    def fun(mod, form):
        _ = mod.create_new_NN_formulation(form)
        n_changes = int(np.random.randint(2, 21, 1)[0])
        random_indeces = np.random.choice(a=30, size=n_changes, replace=False)
        return mod.explore_pyomo(positions=random_indeces)

    results = Parallel(n_jobs=int(n_threads / 6), backend="loky")(delayed(fun)(n_mod[i], model) for i in range(n_steps))
    seq_arr = ["".join(results[i][0]) for i in range(n_steps)]
    obj_arr = np.array([results[i][1:4] for i in range(n_steps)])
    weights = np.array([results[i][4] for i in range(n_steps)])
    random_indeces = np.array([results[i][5] for i in range(n_steps)], dtype=object)
    print(seq_arr)
    print(obj_arr)
    print(weights)
    print(random_indeces)
    np.save(f"./results_example/sequences_exp_{out_name}.npy", seq_arr)
    np.save(f"./results_example/objectives_exp_{out_name}.npy", obj_arr)
    np.save(f"./results_example/weights_exp_{out_name}.npy", weights)
    np.save(f"./results_example/indeces_exp_{out_name}.npy", random_indeces)

