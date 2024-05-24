import sys

import iisignature
import numpy as np
import torch.optim

from server.run_environment import run
from src.strategies import *
from src.fbm_sampling import augmented_paths

SERVER = True

PENALTY = 0.1  # 0.0
CONSTRAINT = None  # (-1,1)
INITIAL = 0.0

result_path = "./results_server/" if SERVER else "./results_local/"

jobs = dict()

learning_rate = 0.1

if SERVER:
    batch_size = 512
    n_batches = 2 ** 10
    validation_size = 1024 * 16 * 8

    epochs = 31
    MC_ = 2 ** 20
    MAX_GB = 32
else:
    batch_size = 1024
    validation_size = 1024 * 8

    epochs = 2
    MC_ = validation_size
    MAX_GB = 8

RESTARTS = 3

params = {
    "penalty": PENALTY,
    "initial": INITIAL,
    "constraint": CONSTRAINT,

    "restarts": RESTARTS,
    "steps_per_restart": 3,
    "batch_size": batch_size,
    "n_batches": n_batches,
    "validation_size": validation_size,

    "learning_rate": learning_rate,

    "optim": "Adam",

    # "optim": "Adagrad",
    # "optim_initial_accumulator_value": 0.1,
    # "optim_epsilon": 1e-07,

    # " optim": "LBFGS",
    # "optim_history_size": 20,
    # "optim_line_search_fn": "strong_wolfe",
    # "optim_tolerance_change": 0.0,
    # "optim_max_iter": 50,

    "nn_dropout": 0.0,

    "max_gb": MAX_GB,
    "MC_": MC_,
    "sig_comp": 'tX',

    "save_model": True,
}

JOBS = [
    {"dscrt_train": dscrt, "dscrt": dscrt, "N": N, "H": H, **mod_,
     "epochs": epochs + N * 10}
    for dscrt in [1000]
    for N in [1, 2, 3, 4, 5]
    for H in [1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0 / 3, 1.0 / 2, 3.0 / 4, 7.0 / 8, 1.0]
    for mod_ in [
        {"space": 'log', "nn_hidden": 2},
        {"space": 'sig', "nn_hidden": 0}
    ]
]

pfs = {
    "time": lambda time_, path_: time_,
    "path": lambda time_, path_: path_,
}


def update_params(restart_, epochs_, *args):
    if epochs_ == 50:
        # return {"batch_size": batch_size * 8}
        return {}
    else:
        return {}


def rde_model(Y, U, dX):
    return dX[:, 1] + U * dX[:, 0]


def build_strategy(N, space, sig_comp, **kwargs):
    if space == 'log':
        strat_ = [DNNStrategy(iisignature.logsiglength(len(sig_comp), N), **kwargs)]
    elif space == 'sig':
        strat_ = [DNNStrategy(iisignature.siglength(len(sig_comp), N), **kwargs)]

    if CONSTRAINT:
        strat_ = strat_ + [ConstraintLayer(constraint=kwargs["constraint"])]

    return torch.nn.Sequential(*strat_)


def new_batch(MC, steps, H, **kwargs):
    time, _, X = augmented_paths(MC=MC, H=H, path_functions=[pfs["time"], pfs["path"]], steps=steps)
    X = torch.tensor(X, requires_grad=False, dtype=torch.float64)
    return time, X


def loss_fn(time, Y, U, X, **kwargs):
    payoff = 0.5 * (Y[:, :-1] ** 2 + PENALTY * U[:, :-1] ** 2)
    l_ = torch.mean(payoff) * (time[-1] - time[0])
    v = torch.var(torch.mean(payoff, dim=1) * (time[-1] - time[0]))
    return l_, v


if __name__ == "__main__":
    run(sys.argv, JOBS, build_strategy, rde_model, loss_fn, new_batch, params, update_params, result_path)
