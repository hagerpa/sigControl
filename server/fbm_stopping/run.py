import sys

import numpy as np
import torch.optim

from server.fbm_tracking.run import new_batch, build_strategy
from server.run_environment import run
from src.strategies import *

SERVER = True

result_path = "./results_server/" if SERVER else "./results_local/"

jobs = dict()

learning_rate = .1

if SERVER:
    batch_size = 512
    n_batches = 2 ** 10
    validation_size = 1024 * 16 * 4

    epochs = 101
    MC_ = 2 ** 20
    MAX_GB = 16
else:
    batch_size = 1024
    validation_size = 1024 * 8

    epochs = 2
    MC_ = validation_size
    MAX_GB = 8

RESTARTS = 3

params = {
    "initial": 0.0,
    "constraint": None,

    "restarts": RESTARTS,
    "steps_per_restart": 3,
    "epochs": epochs,
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
}

JOBS = [
    {"dscrt_train": dscrt, "dscrt": dscrt, "N": N, "H": H, **mod_}
    for dscrt in [1000]
    for N in [1, 2, 3, 4, 5]
    # for sig_comp in MODES
    for H in np.arange(11) / 10.0
    for mod_ in [
        {"space": 'log', "nn_hidden": 2},
        {"space": 'sig', "nn_hidden": 0}
    ]
]

pfs = {
    "time": lambda time_, path_: time_,
    "path": lambda time_, path_: path_,
}


def rde_model(Y, U, dX):
    return U ** 2 * dX[:, 0]


def loss_fn(time, Y, U, X, **kwargs):
    dZ = X[:, 1:, 1] - X[:, :-1, 1]
    dt = X[:, 1:, 0] - X[:, :-1, 0]
    res = torch.sum(torch.exp(- torch.cumsum(U[:, :-1] ** 2 * dt, dim=1)) * dZ, dim=1)
    loss = - torch.mean(res)
    with torch.no_grad():
        var = torch.var(res)
    return loss, var


model_break_points = ()  # (100,)


def update_params(restarts_, epochs_, optim, best_model):
    """
    if epochs_ == 30:
        return {
            "batch_size": batch_size  # * 8,
        }
    elif epochs_ == 60:
        return {
            "batch_size": batch_size  # * 16,
        }
    elif epochs_ == 70:
        return {
            "batch_size": batch_size  # * 32,
        }
    elif epochs_ == 100:
        return {
            "batch_size": 1024 * 8,
            "dscrt_train": 1000,

            "learning_rate": 0.001,

            #"optim": "LBFGS",
            #"learning_rate": learning_rate,
            #"optim_history_size": 20,
            #"optim_line_search_fn": "strong_wolfe",
            #"optim_tolerance_change": 0.0,
            #"optim_max_iter": 50,

            "nn_dropout": 0.0
        }
    else:
        return {}
    """
    if epochs_ == 40:
        return {
            "batch_size": batch_size  # * 8,
        }
    else:
        return {}


if __name__ == "__main__":
    run(sys.argv, JOBS, build_strategy, rde_model, loss_fn, new_batch, params, update_params, result_path,
        model_break_points)
